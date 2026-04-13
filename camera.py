#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AI摄像头流检测主程序
支持多路RTSP/RTMP流并发检测、AI推理、告警、FFmpeg推送AI结果流
"""

import atexit
import faulthandler
import logging
import multiprocessing
import os
import signal
import subprocess
import sys
import threading
import time
from collections import deque
from pathlib import Path
from typing import Dict, Optional
from urllib.parse import urlparse

import cv2
import numpy as np

# ── 全局 NVENC 会话计数器 ──
_nvenc_lock = threading.Lock()
_nvenc_count = 0
_NVENC_MAX_SESSIONS = 10  # 当前按 10 路压测，超出后自动回退 libx264

def _terminate_subprocess(proc: Optional[subprocess.Popen], timeout: float = 3.0):
    if proc is None:
        return

    try:
        if proc.stdin and not proc.stdin.closed:
            proc.stdin.close()
    except Exception:
        pass

    try:
        if os.name != 'nt':
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        else:
            proc.terminate()
        proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        try:
            if os.name != 'nt':
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            else:
                proc.kill()
        except Exception:
            pass
        try:
            proc.wait(timeout=2.0)
        except Exception:
            pass
    except Exception:
        try:
            proc.kill()
            proc.wait(timeout=2.0)
        except Exception:
            pass

def _nvenc_acquire() -> bool:
    """尝试获取一个 NVENC 会话槽位，成功返回 True"""
    global _nvenc_count
    with _nvenc_lock:
        if _nvenc_count < _NVENC_MAX_SESSIONS:
            _nvenc_count += 1
            return True
        return False

def _nvenc_release():
    """释放一个 NVENC 会话槽位"""
    global _nvenc_count
    with _nvenc_lock:
        _nvenc_count = max(0, _nvenc_count - 1)

# 项目根目录加入路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from config.algorithm_config import CameraConfig, VideoCodec
from config.config_manager import ConfigManager
from nan.logger_config import setup_logging
from stream.enhanced_video_processor import (
    EnhancedVideoStreamProcessor,
    VideoStreamConfig,
    VideoStreamStatus,
)
from stream.capture_process import CaptureProxy
from inference.unified_scheduler import UnifiedInferenceScheduler
from alert.alert_system import AlertSystem, create_count_threshold_rule, AlertLevel
from tracking.simple_tracker import SimpleTracker


def _enable_fault_logging(log_dir: str = 'log'):
    try:
        os.makedirs(log_dir, exist_ok=True)
        crash_log = os.path.join(log_dir, 'fatal_trace.log')
        fault_file = open(crash_log, 'a', encoding='utf-8')
        faulthandler.enable(file=fault_file, all_threads=True)
        logging.info(f"faulthandler 已启用: {crash_log}")
        return fault_file
    except Exception as e:
        logging.error(f"启用 faulthandler 失败: {e}")
        return None


class StreamProcessor:
    """单路输入流：拉流 → 推理 → 绘框 → 输出推流 → 告警"""

    def __init__(self, stream_cfg: dict, config: CameraConfig, inference_scheduler):
        self.stream_cfg = stream_cfg
        self.config = config
        self.inference_scheduler = inference_scheduler
        self.inference_engine = inference_scheduler
        self._owns_inference_engine = False

        self.name = stream_cfg.get('name', 'unknown')
        self.input_url = stream_cfg.get('input_url') or stream_cfg.get('rtsp_url') or stream_cfg.get('rtmp_url', '')
        self.output_url = (
            stream_cfg.get('output_url')
            or stream_cfg.get('output_rtsp')
            or stream_cfg.get('output_rtmp')
            or ''
        ) if getattr(config, 'push_enabled', True) else ''
        self.stream_tracking_key = stream_cfg.get('stream_id') or self.name or self.input_url

        self.is_running = False
        self._stop_event = threading.Event()

        self.alert_system = AlertSystem(config)
        result_path = os.path.join(getattr(config, 'output_directory', './res'), self.name)
        self.alert_system.initialize_alert_handler(stream_cfg, result_path)
        self._setup_alert_rules()

        self.pipe: Optional[subprocess.Popen] = None
        self._detected_resolution: Optional[tuple] = None
        self._push_ffmpeg_resolution: Optional[tuple] = None
        self._push_reset_needed = False
        self._stream_codec: Optional[str] = None  # 本流实际使用的编码器
        self._using_nvenc = False  # 本流是否占用了 NVENC 槽位
        self._ffmpeg_restart_backoff = 2.0  # FFmpeg重启退避时间

        self.video_processor = None
        self.capture_proxy: Optional[CaptureProxy] = None
        self._processor_thread: Optional[threading.Thread] = None
        self._push_thread: Optional[threading.Thread] = None

        self._frame_id = 0
        self._last_infer_frame_id = -1
        self._last_applied_result_frame_id = -1
        self._pending_infer = False
        self._crash_trace_enabled = bool(getattr(self.config, 'crash_trace_enabled', False))
        self._last_detection_overlays = []
        self._last_tracking_summary = {
            'track_count': 0,
            'track_ids': [],
            'classes': [],
        }
        self._latest_input_frame: Optional[np.ndarray] = None
        self._latest_rendered_frame: Optional[np.ndarray] = None
        self._latest_frame_lock = threading.Lock()
        self._latest_render_lock = threading.Lock()
        self._render_queue = deque(maxlen=max(16, int(getattr(self.config, 'result_max_back_frames', 30) or 30)))
        self._render_queue_lock = threading.Lock()
        self._frame_buffer = deque(maxlen=max(8, int(getattr(self.config, 'result_max_back_frames', 30) or 30)))
        self._result_overlays_by_frame: Dict[int, list] = {}
        self._output_delay_frames = max(
            0,
            int(getattr(self.config, 'output_delay_frames', 0) or 0),
        )
        self._frame_ready_event = threading.Event()
        self._last_capture_status = 'init'
        self._last_frame_ts = 0.0
        self._last_processed_ts = 0.0
        self._last_push_ts = 0.0
        self._last_infer_result_ts = 0.0
        self._last_infer_result_count = 0
        self._last_target_seen_ts = 0.0
        self._last_motion_level = 0.0
        self._motion_prev_small: Optional[np.ndarray] = None
        self._tracker = SimpleTracker(
            max_missed=max(5, int(getattr(self.config, 'push_fps', self.config.fps) * 2)),
            min_iou=float(getattr(self.config, 'tracking_match_iou', 0.3) or 0.3),
            max_predict_gap_ms=float(getattr(self.config, 'max_predict_gap_ms', 200.0) or 200.0),
        ) if bool(getattr(self.config, 'tracking_enabled', False)) else None

        logging.info(f"[{self.name}] StreamProcessor 初始化完成")

    def _setup_alert_rules(self):
        cooldown = float(getattr(self.config, 'alarm_interval_seconds', 10.0))
        threshold = float(getattr(self.config, 'alarm_target_threshold', 1))
        rule = create_count_threshold_rule(
            rule_id="alarm_any_detection",
            threshold=threshold,
            description="检测到目标",
            level=AlertLevel.MEDIUM,
            cooldown=cooldown,
        )
        self.alert_system.add_rule(rule)
    def _build_capture_options(self) -> str:
        return self.config.get_capture_options(self.input_url)


    def _on_capture_status(self, stream_id: str, status: str):
        """拉流子进程状态回调"""
        status_map = {
            'connecting': VideoStreamStatus.CONNECTING,
            'connected': VideoStreamStatus.CONNECTED,
            'interrupted': VideoStreamStatus.INTERRUPTED,
            'reconnecting': VideoStreamStatus.RECONNECTING,
            'error': VideoStreamStatus.ERROR,
        }
        vs = status_map.get(status)
        if vs:
            self._on_status_change(stream_id, vs)

    def start(self):
        self.is_running = True
        self._stop_event.clear()
        self.inference_scheduler.ensure_stream(self.stream_tracking_key)

        self._detected_resolution = self.config.get_default_resolution()
        logging.info(
            f"[{self.name}] 启动时跳过阻塞分辨率探测，先使用默认分辨率: "
            f"{self._detected_resolution[0]}x{self._detected_resolution[1]}"
        )

        if self.output_url:
            self._open_ffmpeg(self.output_url)

        capture_options = self._build_capture_options()
        logging.info(f"[{self.name}] 拉流参数: {capture_options}")

        self.capture_proxy = CaptureProxy(
            stream_id=f"{self.name}_{int(time.time())}",
            stream_url=self.input_url,
            pull_device=self.config.get_pull_device(),
            capture_options=capture_options,
            frame_width=self._detected_resolution[0],
            frame_height=self._detected_resolution[1],
            frame_callback=self._on_frame,
            status_callback=self._on_capture_status,
        )
        self.capture_proxy.start()

        self._processor_thread = threading.Thread(
            target=self._process_frames_loop,
            daemon=True,
            name=f"Process-{self.name}",
        )
        self._processor_thread.start()
        self._push_thread = threading.Thread(target=self._push_loop, daemon=True, name=f"Push-{self.name}")
        self._push_thread.start()

        logging.info(f"[{self.name}] 启动完成")

    def stop(self):
        logging.info(f"[{self.name}] 停止中...")
        self.is_running = False
        self._stop_event.set()
        self._frame_ready_event.set()

        if self.capture_proxy:
            self.capture_proxy.stop()
            self.capture_proxy = None

        if self.video_processor:
            self.video_processor.stop()
            self.video_processor = None

        self.inference_scheduler.reset_stream_tracking(self.stream_tracking_key)
        self._last_detection_overlays = []
        self._last_tracking_summary = {
            'track_count': 0,
            'track_ids': [],
            'classes': [],
        }
        self._last_infer_frame_id = -1
        self._last_applied_result_frame_id = -1
        if self._tracker is not None:
            self._tracker.reset()
        with self._latest_frame_lock:
            self._latest_input_frame = None
        with self._latest_render_lock:
            self._latest_rendered_frame = None
        with self._render_queue_lock:
            self._render_queue.clear()
        self._frame_buffer.clear()
        self._result_overlays_by_frame.clear()

        self.inference_scheduler.remove_stream(self.stream_tracking_key)
        self._close_ffmpeg(release_nvenc=True)
        if self._owns_inference_engine:
            try:
                self.inference_engine.cleanup()
            except Exception as e:
                logging.error(f"[{self.name}] 清理独立推理引擎失败: {e}")
        logging.info(f"[{self.name}] 已停止")

    def _trace_stage(self, fid: int, stage: str, **kwargs):
        if not getattr(self, '_crash_trace_enabled', False):
            return
        extras = []
        for key, value in kwargs.items():
            extras.append(f"{key}={value}")
        suffix = f" {' '.join(extras)}" if extras else ''
        logging.info(f"[{self.name}] fid={fid} stage={stage}{suffix}")
        for handler in logging.getLogger().handlers:
            try:
                handler.flush()
            except Exception:
                pass

    def _on_frame(self, stream_id: str, frame: np.ndarray):
        self._last_frame_ts = time.time()
        try:
            frame = np.ascontiguousarray(frame, dtype=np.uint8)
            fh, fw = frame.shape[:2]
            actual_resolution = (fw, fh)
            if self._detected_resolution != actual_resolution:
                previous_resolution = self._detected_resolution
                self._detected_resolution = actual_resolution
                if previous_resolution:
                    logging.info(
                        f"[{self.name}] 首帧确认真实分辨率: {fw}x{fh} "
                        f"(原启动值 {previous_resolution[0]}x{previous_resolution[1]})"
                    )
                else:
                    logging.info(f"[{self.name}] 首帧确认真实分辨率: {fw}x{fh}")
                if self.output_url and self._push_ffmpeg_resolution and self._push_ffmpeg_resolution != actual_resolution:
                    self._push_reset_needed = True
            with self._latest_frame_lock:
                self._latest_input_frame = frame
            self._frame_ready_event.set()
        except Exception as e:
            logging.error(f"[{self.name}] 接收帧异常: {e}")

    def _take_latest_input_frame(self) -> Optional[np.ndarray]:
        with self._latest_frame_lock:
            frame = self._latest_input_frame
            self._latest_input_frame = None
        return frame

    def _store_latest_rendered_frame(self, frame: np.ndarray):
        frame = np.ascontiguousarray(frame, dtype=np.uint8)
        with self._latest_render_lock:
            self._latest_rendered_frame = frame

    def _enqueue_rendered_frame(self, frame: np.ndarray):
        frame = np.ascontiguousarray(frame, dtype=np.uint8)
        with self._latest_render_lock:
            self._latest_rendered_frame = frame
        with self._render_queue_lock:
            self._render_queue.append(frame)

    def _take_next_rendered_frame(self) -> Optional[np.ndarray]:
        with self._render_queue_lock:
            if not self._render_queue:
                return None
            return self._render_queue.popleft()

    def _take_latest_rendered_frame(self) -> Optional[np.ndarray]:
        with self._latest_render_lock:
            frame = self._latest_rendered_frame
            self._latest_rendered_frame = None
        return frame

    def _process_frames_loop(self):
        logging.info(f"[{self.name}] 处理线程启动")
        while self.is_running and not self._stop_event.is_set():
            self._frame_ready_event.wait(1.0)
            if self._stop_event.is_set():
                break
            frame = self._take_latest_input_frame()
            if frame is None:
                self._frame_ready_event.clear()
                continue
            if self._latest_input_frame is None:
                self._frame_ready_event.clear()
            self._process_frame(frame)
        logging.info(f"[{self.name}] 处理线程结束")

    def _estimate_motion_level(self, frame: np.ndarray) -> float:
        if not bool(getattr(self.config, 'motion_detection_enabled', True)):
            return 100.0
        try:
            width = int(getattr(self.config, 'motion_resize_width', 160))
            height = int(getattr(self.config, 'motion_resize_height', 90))
            small = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
            gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
            if self._motion_prev_small is None:
                self._motion_prev_small = gray
                return 100.0
            diff = cv2.absdiff(gray, self._motion_prev_small)
            self._motion_prev_small = gray
            return float(diff.mean())
        except Exception:
            return 100.0

    def _current_inference_interval(self, frame: np.ndarray) -> int:
        base_interval = max(1, int(getattr(self.config, 'detection_inference_interval', 5)))
        idle_interval = max(base_interval, int(getattr(self.config, 'inference_idle_interval', base_interval)))
        hold_seconds = float(getattr(self.config, 'inference_active_hold_seconds', 2.0) or 2.0)
        motion_threshold = float(getattr(self.config, 'motion_threshold', 3.5) or 3.5)

        motion_level = self._estimate_motion_level(frame)
        self._last_motion_level = motion_level
        active_recently = self._last_target_seen_ts > 0 and (time.time() - self._last_target_seen_ts) <= hold_seconds
        if active_recently or motion_level >= motion_threshold:
            return base_interval
        return idle_interval

    def _process_frame(self, frame: np.ndarray):
        try:
            self._frame_id += 1
            fid = self._frame_id
            interval = self._current_inference_interval(frame)
            self._frame_buffer.append((fid, frame.copy()))
            if self._tracker is not None:
                self._tracker.predict(frame)
                self._last_detection_overlays = self._tracker.get_active_tracks()

            latest_result = self.inference_scheduler.get_latest_result(self.stream_tracking_key)
            if latest_result:
                result_fid = int(latest_result.get('frame_id', 0) or 0)
                result_ts = float(latest_result.get('result_ts', 0.0) or 0.0)
                result_age = time.time() - result_ts if result_ts > 0 else 0.0
                frame_lag = max(0, fid - result_fid)
                max_result_age = float(getattr(self.config, 'max_infer_result_age', 1.0) or 1.0)
                max_frame_lag = max(1, int(getattr(self.config, 'max_infer_frame_lag', 5) or 5))
                if (
                    result_fid > self._last_applied_result_frame_id
                    and result_age <= max_result_age
                    and frame_lag <= max_frame_lag
                ):
                    self._process_infer_results(latest_result.get('results', {}) or {}, result_fid)
                    self._last_applied_result_frame_id = result_fid
                else:
                    # 记录推理结果被丢弃的原因，便于调试
                    if result_fid <= self._last_applied_result_frame_id:
                        logging.debug(f"[{self.name}] 丢弃旧推理结果: result_fid={result_fid} <= last_applied={self._last_applied_result_frame_id}")
                    elif result_age > max_result_age:
                        logging.debug(f"[{self.name}] 丢弃过期推理结果: result_age={result_age:.2f}s > max={max_result_age}s, fid={fid}, result_fid={result_fid}")
                    elif frame_lag > max_frame_lag:
                        logging.debug(f"[{self.name}] 丢弃滞后推理结果: frame_lag={frame_lag} > max={max_frame_lag}, fid={fid}, result_fid={result_fid}")

            detection_dict = {}

            # 异步推理：提交新请求
            if self.inference_scheduler.is_loaded() and (fid - self._last_infer_frame_id) >= interval:
                self._last_infer_frame_id = fid
                self._pending_infer = self.inference_scheduler.submit_frame(
                    stream_key=self.stream_tracking_key,
                    frame=frame,
                    algo_id=None,
                    frame_id=fid,
                )

            alert_target_info = None
            track_count = int(self._last_tracking_summary.get('track_count', 0))
            if track_count > 0:
                overlay_classes = sorted({str(o.get('class_name', '')).strip() for o in self._last_detection_overlays if str(o.get('class_name', '')).strip()})
                class_text = ','.join(overlay_classes) if overlay_classes else 'unknown'
                detection_dict["alarm_any_detection"] = float(track_count)
                alert_target_info = {
                    'classes': class_text,
                    'class_name': class_text,
                    'count': track_count,
                    'track_count': track_count,
                    'track_ids': list(self._last_tracking_summary.get('track_ids', [])),
                    'tracking_enabled': bool(getattr(self.config, 'tracking_enabled', False)),
                }

            rendered_frame = frame.copy()
            if self._last_detection_overlays:
                rendered_frame = self._draw_detection_overlays(rendered_frame, self._last_detection_overlays)
            rendered_frame = self._draw_ai_badge(rendered_frame, fid=fid)
            self._enqueue_rendered_frame(rendered_frame)
            self._last_processed_ts = time.time()

            if self.alert_system.alert_handler:
                self.alert_system.alert_handler.collect_clip_frame(rendered_frame)
                if detection_dict and alert_target_info:
                    self.alert_system.process_frame_alerts(rendered_frame, detection_dict, target_info=alert_target_info)

        except Exception as e:
            logging.error(f"[{self.name}] 帧处理异常: {e}")

    def _process_infer_results(self, results, fid):
        """处理推理结果，更新检测覆盖层"""
        raw_detections = []
        total = 0
        class_names = set()
        for aid, res in results.items():
            cnt = self._count_detections(res)
            total += cnt
            model_detections = self._extract_raw_detections(res, aid, fid=fid)
            raw_detections.extend(model_detections)
            for det in model_detections:
                c = str(det.get('class_name', '')).strip()
                if c:
                    class_names.add(c)
        track_ids = set()
        tracking_enabled = bool(getattr(self.config, 'tracking_enabled', False))
        if self._tracker is not None:
            self._tracker.update(raw_detections, frame=None)
            overlays = self._tracker.get_active_tracks()
            for overlay in overlays:
                tid = overlay.get('track_id')
                if tid not in (None, ''):
                    try:
                        track_ids.add(int(tid))
                    except Exception:
                        track_ids.add(tid)
        else:
            overlays = self._build_overlays_from_detections(raw_detections)

        alarm_count = len(track_ids) if tracking_enabled and track_ids else total
        if alarm_count > 0:
            self._last_target_seen_ts = time.time()
            logging.info(f"[{self.name}] 检测到目标: alarm_count={alarm_count}, track_ids={sorted(track_ids, key=lambda x: str(x))}, classes={sorted(class_names)}")
        self._last_infer_result_ts = time.time()
        self._last_infer_result_count = int(alarm_count)
        self._last_tracking_summary = {
            'track_count': int(len(track_ids)) if tracking_enabled else int(total),
            'track_ids': sorted(track_ids, key=lambda x: str(x)),
            'classes': sorted(class_names),
        }
        self._last_detection_overlays = overlays

    def _on_status_change(self, stream_id: str, status: VideoStreamStatus):
        self._last_capture_status = status.value
        logging.info(f"[{self.name}] 流状态: {status.value}")
        if status == VideoStreamStatus.INTERRUPTED:
            self.inference_scheduler.reset_stream_tracking(self.stream_tracking_key)
            self._last_detection_overlays = []
            self._last_tracking_summary = {
                'track_count': 0,
                'track_ids': [],
                'classes': [],
            }
            self._last_infer_frame_id = -1
            self._last_applied_result_frame_id = -1
            if self._tracker is not None:
                self._tracker.reset()
            with self._latest_frame_lock:
                self._latest_input_frame = None
            with self._latest_render_lock:
                self._latest_rendered_frame = None
            with self._render_queue_lock:
                self._render_queue.clear()
            self._frame_buffer.clear()
            self._result_overlays_by_frame.clear()
        elif status in (VideoStreamStatus.READING, VideoStreamStatus.CONNECTED):
            self._push_reset_needed = True

    def _on_error(self, stream_id: str, error_msg: str):
        logging.error(f"[{self.name}] 流错误: {error_msg}")

    def _count_detections(self, results) -> int:
        try:
            r = results[0] if isinstance(results, (list, tuple)) and results else results
            if r is not None and hasattr(r, 'boxes') and r.boxes is not None:
                return int(len(r.boxes))
        except Exception:
            pass
        return 0

    def _extract_raw_detections(self, results, algo_id: str, fid: Optional[int] = None):
        detections = []
        try:
            r = results[0] if isinstance(results, (list, tuple)) and results else results
            if r is None or not hasattr(r, 'boxes') or r.boxes is None:
                return detections
            boxes = r.boxes
            if len(boxes) == 0:
                return detections
            if fid is not None:
                self._trace_stage(fid, 'tensor_extract_start', algo_id=algo_id)
            xyxy = boxes.xyxy.cpu().numpy() if hasattr(boxes.xyxy, 'cpu') else np.asarray(boxes.xyxy)
            confs = boxes.conf.cpu().numpy() if hasattr(boxes.conf, 'cpu') else np.asarray(boxes.conf)
            clss = boxes.cls.cpu().numpy() if hasattr(boxes.cls, 'cpu') else np.asarray(boxes.cls)
            if fid is not None:
                self._trace_stage(fid, 'tensor_extract_end', algo_id=algo_id, box_count=len(xyxy))
            names = getattr(r, 'names', {})
            color = self._color_for_model(algo_id)
            raw_detections = []
            for i in range(len(xyxy)):
                x1, y1, x2, y2 = map(int, xyxy[i])
                conf = float(confs[i])
                cls_id = int(clss[i])
                label = names.get(cls_id, str(cls_id)) if isinstance(names, dict) else str(cls_id)
                detections.append({
                    'xyxy': (x1, y1, x2, y2),
                    'color': color,
                    'class_name': str(label),
                    'algo_id': str(algo_id),
                    'confidence': conf,
                })
        except Exception as e:
            logging.debug(f"[{self.name}] 提取绘框信息异常: {e}")
        return detections

    def _build_overlays_from_detections(self, detections):
        overlays = []
        for det in detections or []:
            overlays.append({
                'xyxy': det['xyxy'],
                'text': f"{det['algo_id']}:{det['class_name']} {det['confidence']:.2f}",
                'color': det['color'],
                'class_name': det['class_name'],
                'algo_id': det['algo_id'],
                'confidence': det['confidence'],
                'track_id': None,
            })
        return overlays

    def _find_frame_in_buffer(self, fid: int) -> Optional[np.ndarray]:
        for frame_id, frame in reversed(self._frame_buffer):
            if int(frame_id) == int(fid):
                return frame
        return None

    def _get_replay_frames(self, fid: int):
        replay_frames = []
        for frame_id, frame in self._frame_buffer:
            if int(frame_id) >= int(fid):
                replay_frames.append((int(frame_id), frame))
        return replay_frames

    def _select_overlays_for_frame(self, fid: int):
        if not self._result_overlays_by_frame:
            return []
        candidate_fids = [result_fid for result_fid in self._result_overlays_by_frame.keys() if int(result_fid) <= int(fid)]
        if not candidate_fids:
            return []
        selected_fid = max(candidate_fids)
        return list(self._result_overlays_by_frame.get(int(selected_fid), []) or [])

    def _emit_ready_frames(self):
        while len(self._frame_buffer) > self._output_delay_frames:
            frame_id, frame = self._frame_buffer.popleft()
            overlays = self._select_overlays_for_frame(frame_id)
            self._last_detection_overlays = overlays
            track_ids = []
            class_names = []
            for overlay in overlays:
                track_id = overlay.get('track_id')
                if track_id not in (None, ''):
                    track_ids.append(track_id)
                class_name = str(overlay.get('class_name', '')).strip()
                if class_name:
                    class_names.append(class_name)
            self._last_tracking_summary = {
                'track_count': int(len(track_ids) if track_ids else len(overlays)),
                'track_ids': list(track_ids),
                'classes': sorted(set(class_names)),
            }

            rendered_frame = frame.copy()
            if overlays:
                rendered_frame = self._draw_detection_overlays(rendered_frame, overlays)
            rendered_frame = self._draw_ai_badge(rendered_frame, fid=frame_id)
            self._enqueue_rendered_frame(rendered_frame)

            stale_keys = [key for key in self._result_overlays_by_frame.keys() if int(key) < int(frame_id) - self._output_delay_frames]
            for key in stale_keys:
                self._result_overlays_by_frame.pop(key, None)
    def _draw_ai_badge(self, frame: np.ndarray, fid: Optional[int] = None) -> np.ndarray:
        try:
            badge_text = "AI"
            ts_text = time.strftime('%H:%M:%S')
            if fid is not None:
                ts_text = f"{ts_text} F{fid}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.8
            thickness = 2
            margin = 12
            (text_w, text_h), baseline = cv2.getTextSize(badge_text, font, font_scale, thickness)
            (ts_w, ts_h), ts_baseline = cv2.getTextSize(ts_text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
            h, w = frame.shape[:2]
            box_w = max(text_w + 24, ts_w + 24)
            box_h = text_h + ts_h + baseline + ts_baseline + 28
            x1 = w - box_w - margin
            y1 = h - box_h - margin
            x2 = w - margin
            y2 = h - margin
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 140, 255), -1)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 2)
            cv2.putText(frame, badge_text, (x1 + 12, y1 + text_h + 8), font, font_scale, (255, 255, 255), thickness)
            cv2.putText(frame, ts_text, (x1 + 12, y2 - ts_baseline - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
        except Exception as e:
            logging.debug(f"[{self.name}] 绘制AI标识异常: {e}")
        return frame

    def _draw_detection_overlays(self, frame: np.ndarray, overlays) -> np.ndarray:
        try:
            for overlay in overlays or []:
                x1, y1, x2, y2 = overlay['xyxy']
                color = overlay['color']
                text = overlay['text']
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, text, (x1, max(y1 - 5, 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        except Exception as e:
            logging.debug(f"[{self.name}] 绘制缓存框异常: {e}")
        return frame

    def _draw_detections(self, frame: np.ndarray, results, algo_id: str) -> np.ndarray:
        overlays = self._build_overlays_from_detections(
            self._extract_raw_detections(results, algo_id)
        )
        return self._draw_detection_overlays(frame, overlays)

    def _color_for_model(self, algo_id: str):
        palette = [
            (0, 255, 0),
            (0, 165, 255),
            (255, 0, 0),
            (255, 255, 0),
            (255, 0, 255),
            (0, 255, 255),
        ]
        idx = sum(ord(c) for c in str(algo_id)) % len(palette)
        return palette[idx]

    def _open_ffmpeg(self, output_url: str):
        self.output_url = output_url
        w, h = self._detected_resolution or self.config.get_default_resolution()
        fps = getattr(self.config, 'push_fps', self.config.fps)
        self._push_ffmpeg_resolution = (w, h)
        output_scheme = (urlparse(output_url).scheme or 'rtsp').lower()

        def _build_cmd(codec: str, hw: bool):
            cmd = [
                'ffmpeg', '-y', '-hide_banner', '-loglevel', 'warning', '-nostats',
                '-fflags', 'nobuffer', '-flags', 'low_delay',
                '-f', 'rawvideo', '-vcodec', 'rawvideo', '-pix_fmt', 'bgr24',
                '-s', f'{w}x{h}', '-r', str(fps), '-i', '-',
                '-an',
                '-c:v', codec,
            ]
            if hw:
                cmd += ['-preset', getattr(self.config, 'encoding_preset', 'p4'), '-tune', 'll']
            else:
                cmd += ['-preset', 'ultrafast', '-tune', 'zerolatency']
            cmd += [
                '-g', str(max(getattr(self.config, 'gop_size', 50), fps)),
                '-keyint_min', str(max(1, fps)),
                '-bf', '0',
                '-b:v', getattr(self.config, 'bitrate', '4M'),
                '-maxrate', getattr(self.config, 'max_bitrate', '6M'),
                '-bufsize', getattr(self.config, 'buffer_size', '8M'),
                '-pix_fmt', 'yuv420p',
                '-flush_packets', '1',
            ]
            if output_scheme == 'rtsp':
                cmd += ['-rtsp_transport', 'tcp', '-muxdelay', '0', '-muxpreload', '0', '-f', 'rtsp', output_url]
            elif output_scheme == 'rtmp':
                cmd += ['-flvflags', 'no_duration_filesize', '-f', 'flv', output_url]
            else:
                raise ValueError(f"不支持的推流协议: {output_scheme}")
            return cmd

        # 确定本流使用的编码器（带 NVENC 槽位管理）
        codec = self._resolve_push_codec()
        hw = codec == VideoCodec.H264_NVENC.value

        codecs_to_try = [codec]
        if hw:
            codecs_to_try.append(VideoCodec.LIBX264.value)

        for attempt, try_codec in enumerate(codecs_to_try):
            try_hw = try_codec == VideoCodec.H264_NVENC.value
            cmd = _build_cmd(try_codec, try_hw)
            logging.info(f"[{self.name}] FFmpeg命令(尝试{attempt + 1}, codec={try_codec}): {' '.join(cmd)}")
            try:
                pipe = subprocess.Popen(
                    cmd,
                    stdin=subprocess.PIPE,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.PIPE,
                    bufsize=0,
                    start_new_session=(os.name != 'nt'),
                )
            except Exception as e:
                logging.error(f"[{self.name}] 启动FFmpeg失败: {e}")
                if try_hw and self._using_nvenc:
                    _nvenc_release()
                    self._using_nvenc = False
                continue
            time.sleep(0.3)
            if pipe.poll() is not None:
                err = ''
                try:
                    err = pipe.stderr.read().decode('utf-8', errors='ignore')[:500]
                except Exception:
                    pass
                logging.error(f"[{self.name}] FFmpeg启动即退出(codec={try_codec}): {err}")
                try:
                    pipe.kill()
                except Exception:
                    pass
                # NVENC 失败，释放槽位，回退 libx264
                if try_hw and self._using_nvenc:
                    _nvenc_release()
                    self._using_nvenc = False
                    logging.warning(f"[{self.name}] NVENC编码失败，回退libx264")
                continue
            # 启动成功
            self.pipe = pipe
            self._stream_codec = try_codec
            self._ffmpeg_restart_backoff = 2.0  # 成功后重置退避
            logging.info(f"[{self.name}] FFmpeg推流启动成功(codec={try_codec}) -> {output_url}")
            return

        logging.error(f"[{self.name}] FFmpeg推流进程启动失败（所有编码器均失败）")

    def _resolve_push_codec(self) -> str:
        """决定本流使用的编码器，带 NVENC 会话槽位管理"""
        # 如果已经确定了回退编码器，继续使用
        if self._stream_codec == VideoCodec.LIBX264.value:
            return VideoCodec.LIBX264.value

        push_device = self.config.get_push_device()
        want_nvenc = False
        if push_device == 'cpu':
            return VideoCodec.LIBX264.value
        elif push_device == 'gpu':
            want_nvenc = True
        elif self.config.is_auto_codec_enabled():
            want_nvenc = True
        else:
            codec_val = self.config.get_video_codec()
            want_nvenc = (codec_val == VideoCodec.H264_NVENC.value)

        if want_nvenc:
            # 如果本流已经持有 NVENC 槽位，直接用
            if self._using_nvenc:
                return VideoCodec.H264_NVENC.value
            # 尝试获取 NVENC 槽位
            if _nvenc_acquire():
                self._using_nvenc = True
                logging.info(f"[{self.name}] 获取NVENC槽位成功")
                return VideoCodec.H264_NVENC.value
            else:
                logging.warning(f"[{self.name}] NVENC槽位已满，使用libx264")
                self._stream_codec = VideoCodec.LIBX264.value
                return VideoCodec.LIBX264.value

        return VideoCodec.LIBX264.value

    def _close_ffmpeg(self, release_nvenc: bool = False):
        if self.pipe:
            try:
                _terminate_subprocess(self.pipe, timeout=3.0)
            except Exception:
                pass
            self.pipe = None
        if release_nvenc and self._using_nvenc:
            _nvenc_release()
            self._using_nvenc = False
            self._stream_codec = None  # 下次重新决定编码器

    def _read_ffmpeg_stderr(self, pipe: Optional[subprocess.Popen]) -> str:
        if not pipe or not pipe.stderr:
            return ''
        try:
            if pipe.poll() is None:
                return ''
            err = pipe.stderr.read().decode('utf-8', errors='ignore').strip()
            return err[:1000]
        except Exception:
            return ''

    def _describe_ffmpeg_failure(self) -> str:
        pipe = self.pipe
        if not pipe:
            return 'FFmpeg进程不存在'

        reasons = []
        return_code = pipe.poll()
        if return_code is not None:
            reasons.append(f'returncode={return_code}')
            err = self._read_ffmpeg_stderr(pipe)
            if err:
                reasons.append(f'stderr={err}')
        if not pipe.stdin or pipe.stdin.closed:
            reasons.append('stdin已关闭')
        return '; '.join(reasons) if reasons else ''

    def _restart_ffmpeg(self) -> bool:
        self._close_ffmpeg(release_nvenc=True)
        if self.output_url:
            self._open_ffmpeg(self.output_url)
            return self.pipe is not None
        return False

    def _build_pipeline_snapshot(self) -> str:
        now = time.time()
        frame_age = (now - self._last_frame_ts) if self._last_frame_ts else -1.0
        process_age = (now - self._last_processed_ts) if self._last_processed_ts else -1.0
        push_age = (now - self._last_push_ts) if self._last_push_ts else -1.0
        infer_age = (now - self._last_infer_result_ts) if self._last_infer_result_ts else -1.0
        with self._render_queue_lock:
            rendered_ready = bool(self._render_queue)
        return (
            f"capture_status={self._last_capture_status}, "
            f"frame_id={self._frame_id}, "
            f"frame_age={frame_age:.1f}s, "
            f"process_age={process_age:.1f}s, "
            f"push_age={push_age:.1f}s, "
            f"infer_age={infer_age:.1f}s, "
            f"last_alarm_count={self._last_infer_result_count}, "
            f"rendered_ready={rendered_ready}, "
            f"codec={self._stream_codec or 'unknown'}, "
            f"output_url={self.output_url or '(none)'}"
        )

    def _check_ffmpeg_health(self) -> bool:
        if not self.pipe:
            return False
        if self.pipe.poll() is not None:
            return False
        if not self.pipe.stdin or self.pipe.stdin.closed:
            return False
        return True

    def _push_loop(self):
        logging.info(f"[{self.name}] 推流线程启动")
        fps = getattr(self.config, 'push_fps', self.config.fps)
        interval = 1.0 / fps
        next_push_time = time.perf_counter()
        last_frame: Optional[np.ndarray] = None
        repeated_frame_count = 0
        max_repeat_frames = 1
        stale_repeat_window = max(0.3, interval * 2.5)
        frame_count = 0

        while self.is_running:
            if self._push_reset_needed:
                last_frame = None
                repeated_frame_count = 0
                self._push_reset_needed = False

            now = time.perf_counter()
            sleep_time = next_push_time - now
            if sleep_time > 0:
                time.sleep(min(sleep_time, 0.02))
                continue
            if sleep_time < -interval * 3:
                next_push_time = now

            frame = None
            next_frame = self._take_next_rendered_frame()
            got_new_frame = next_frame is not None

            if got_new_frame:
                frame = next_frame
                last_frame = next_frame
                repeated_frame_count = 0
            elif (
                last_frame is not None
                and repeated_frame_count < max_repeat_frames
                and self._last_processed_ts
                and (time.time() - self._last_processed_ts) <= stale_repeat_window
            ):
                frame = last_frame
                repeated_frame_count += 1
            else:
                next_push_time = now + interval
                time.sleep(0.005)
                continue

            if not self.output_url:
                next_push_time += interval
                continue

            if not self._check_ffmpeg_health():
                failure_reason = self._describe_ffmpeg_failure()
                logging.error(f"[{self.name}] output broken pipe snapshot: {self._build_pipeline_snapshot()}")
                logging.warning(f"[{self.name}] output health snapshot: {self._build_pipeline_snapshot()}")
                if failure_reason:
                    logging.warning(f"[{self.name}] FFmpeg异常，尝试重启(退避{self._ffmpeg_restart_backoff:.1f}s)，原因: {failure_reason}")
                else:
                    logging.warning(f"[{self.name}] FFmpeg异常，尝试重启(退避{self._ffmpeg_restart_backoff:.1f}s)")
                time.sleep(self._ffmpeg_restart_backoff)
                if not self._restart_ffmpeg():
                    # 指数退避，最大30秒
                    self._ffmpeg_restart_backoff = min(30.0, self._ffmpeg_restart_backoff * 2)
                    next_push_time = time.perf_counter() + interval
                    continue
                else:
                    self._ffmpeg_restart_backoff = 2.0  # 成功后重置
                    next_push_time = time.perf_counter() + interval
                    continue

            try:
                fh, fw = frame.shape[:2]
                exp = self._push_ffmpeg_resolution
                if exp and (fw, fh) != exp:
                    logging.warning(f"[{self.name}] 帧尺寸不一致 {fw}x{fh} vs {exp}，重启FFmpeg")
                    self._detected_resolution = (fw, fh)
                    self._restart_ffmpeg()
                    next_push_time = time.perf_counter() + interval
                    continue
            except Exception:
                pass

            try:
                frame = np.ascontiguousarray(frame, dtype=np.uint8)
                self.pipe.stdin.write(memoryview(frame))
                frame_count += 1
                self._last_push_ts = time.time()
                next_push_time += interval

                if self.alert_system.alert_handler:
                    self.alert_system.alert_handler.update_latest_push_frame(frame)

                if frame_count % 500 == 0:
                    logging.info(f"[{self.name}] 已推流 {frame_count} 帧")
            except BrokenPipeError:
                logging.error(f"[{self.name}] output broken pipe snapshot: {self._build_pipeline_snapshot()}")
                failure_reason = self._describe_ffmpeg_failure()
                logging.error(f"[{self.name}] 推流管道断开，尝试恢复，原因: {failure_reason}")
                if not self._restart_ffmpeg():
                    self._ffmpeg_restart_backoff = min(30.0, self._ffmpeg_restart_backoff * 2)
                    time.sleep(self._ffmpeg_restart_backoff)
                else:
                    self._ffmpeg_restart_backoff = 2.0
                next_push_time = time.perf_counter() + interval
            except Exception as e:
                logging.error(f"[{self.name}] 推流写入失败: {e}")
                logging.error(f"[{self.name}] output write-failure snapshot: {self._build_pipeline_snapshot()}")
                time.sleep(0.5)
                next_push_time = time.perf_counter() + interval

        logging.info(f"[{self.name}] 推流线程结束")


class CameraStreamManager:
    """管理所有摄像头流的生命周期"""

    def __init__(self, config: CameraConfig):
        self.config = config
        self.inference_scheduler = UnifiedInferenceScheduler(config)
        self._processors: Dict[str, StreamProcessor] = {}
        self._threads: Dict[str, threading.Thread] = {}
        self._lock = threading.Lock()
        self.running = False

    def start_all(self, stream_list: list):
        self.running = True
        for scfg in stream_list:
            if not scfg.get('enabled', True):
                continue
            self._start_stream(scfg)

    def _start_stream(self, scfg: dict):
        name = scfg.get('name', scfg.get('input_url') or scfg.get('rtsp_url') or scfg.get('rtmp_url', 'unknown'))
        with self._lock:
            if name in self._processors:
                logging.warning(f"流 [{name}] 已在运行")
                return
            proc = StreamProcessor(scfg, self.config, self.inference_scheduler)
            self._processors[name] = proc

        t = threading.Thread(target=self._run_stream, args=(name, proc), daemon=True, name=f"Stream-{name}")
        with self._lock:
            self._threads[name] = t
        t.start()
        logging.info(f"流 [{name}] 线程已启动")

    def _run_stream(self, name: str, proc: StreamProcessor):
        try:
            proc.start()
            while self.running and proc.is_running:
                time.sleep(1.0)
        except Exception as e:
            logging.error(f"流 [{name}] 运行异常: {e}")
        finally:
            try:
                proc.stop()
            except Exception:
                pass
            with self._lock:
                self._processors.pop(name, None)
                self._threads.pop(name, None)
            logging.info(f"流 [{name}] 线程退出")

    def stop_all(self):
        self.running = False
        with self._lock:
            procs = list(self._processors.values())
        for proc in procs:
            try:
                proc.stop()
            except Exception as e:
                logging.error(f"停止流异常: {e}")
        with self._lock:
            threads = list(self._threads.values())
        for t in threads:
            t.join(timeout=10.0)
        self.inference_scheduler.cleanup()
        logging.info("所有流已停止")

    def get_active_count(self) -> int:
        with self._lock:
            return len(self._processors)


def main():
    setup_logging()
    fault_log_handle = _enable_fault_logging()
    logging.info("=" * 60)
    logging.info("AI摄像头流检测系统启动")
    logging.info("=" * 60)

    config = CameraConfig()
    cfg_mgr = ConfigManager()
    stream_list = cfg_mgr.get_enabled_streams()

    if not stream_list:
        logging.error("未找到任何启用的流配置，请检查 config/config.json")
        return 1

    logging.info(f"共 {len(stream_list)} 路流:")
    for s in stream_list:
        input_url = s.get('input_url') or s.get('rtsp_url') or s.get('rtmp_url', '')
        output_url = s.get('output_url') or s.get('output_rtsp') or s.get('output_rtmp') or '(无推流)'
        logging.info(f"  [{s.get('name')}] {input_url} -> {output_url}")

    if config.push_enabled:
        push_dev = config.get_push_device()
        nvenc_info = f"，NVENC最大并发={_NVENC_MAX_SESSIONS}，超出自动回退libx264" if push_dev == 'gpu' else ''
        logging.info(f"AI输出流已启用，拉流设备={config.get_pull_device()}，推流设备={push_dev}，编码模式={config.get_video_codec()}{nvenc_info}")
    else:
        logging.info("AI输出流已禁用，仅本地处理")

    manager = CameraStreamManager(config)

    def _signal_handler(signum, frame):
        logging.info(f"收到信号 {signum}，开始优雅退出...")
        manager.stop_all()
        sys.exit(0)

    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    manager.start_all(stream_list)
    logging.info("所有流已启动，进入主监控循环（Ctrl+C 退出）")

    try:
        while True:
            active = manager.get_active_count()
            logging.info(f"[主循环] 活跃流数量: {active}/{len(stream_list)}")
            time.sleep(30)
    except KeyboardInterrupt:
        logging.info("收到键盘中断，退出")
    finally:
        manager.stop_all()
        if fault_log_handle is not None:
            try:
                fault_log_handle.flush()
                fault_log_handle.close()
            except Exception:
                pass

    return 0


def _kill_child_processes():
    """主进程退出时强制清理所有子进程，防止孤儿进程残留"""
    try:
        import psutil
        current = psutil.Process(os.getpid())
        children = current.children(recursive=True)
        for child in children:
            try:
                child.kill()
            except Exception:
                pass
        try:
            psutil.wait_procs(children, timeout=3.0)
        except Exception:
            pass
    except ImportError:
        # psutil 不可用时用 os.killpg 兜底
        try:
            os.killpg(os.getpgid(os.getpid()), signal.SIGKILL)
        except Exception:
            pass
    except Exception:
        pass


if __name__ == '__main__':
    multiprocessing.set_start_method('forkserver', force=True)
    atexit.register(_kill_child_processes)
    sys.exit(main())

