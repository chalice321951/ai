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
import queue
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Dict, Optional
from urllib.parse import urlparse

import cv2
import numpy as np

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
from inference.inference_engine import InferenceEngine
from inference.inference_process import InferenceProxy
from alert.alert_system import AlertSystem, create_count_threshold_rule, AlertLevel


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

    def __init__(self, stream_cfg: dict, config: CameraConfig,
                 inference_engine, owns_inference_engine: bool = False):
        self.stream_cfg = stream_cfg
        self.config = config
        self.inference_engine = inference_engine
        self._owns_inference_engine = owns_inference_engine

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

        self.push_queue: queue.Queue = queue.Queue(maxsize=5)

        self.alert_system = AlertSystem(config)
        result_path = os.path.join(getattr(config, 'output_directory', './res'), self.name)
        self.alert_system.initialize_alert_handler(stream_cfg, result_path)
        self._setup_alert_rules()

        self.pipe: Optional[subprocess.Popen] = None
        self._detected_resolution: Optional[tuple] = None
        self._push_ffmpeg_resolution: Optional[tuple] = None
        self._push_reset_needed = False

        self.video_processor: Optional[EnhancedVideoStreamProcessor] = None

        self._frame_id = 0
        self._last_infer_frame_id = -1
        self._crash_trace_enabled = bool(getattr(self.config, 'crash_trace_enabled', False))
        self._last_detection_overlays = []
        self._last_tracking_summary = {
            'track_count': 0,
            'track_ids': [],
            'classes': [],
        }

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

    def start(self):
        self.is_running = True
        self._stop_event.clear()

        try:
            w, h = self.config.auto_detect_and_update_resolution(self.input_url)
            self._detected_resolution = (w, h)
            logging.info(f"[{self.name}] 分辨率: {w}x{h}")
        except Exception as e:
            logging.warning(f"[{self.name}] 分辨率检测失败: {e}，使用默认值")
            self._detected_resolution = self.config.get_default_resolution()

        if self.output_url:
            self._open_ffmpeg(self.output_url)

        w, h = self._detected_resolution
        stream_config = VideoStreamConfig(
            stream_url=self.input_url,
            target_width=w,
            target_height=h,
            expected_fps=self.config.fps,
            pull_device=getattr(self.config, 'pull_device', 'cpu'),
            connection_timeout=10.0,
            read_timeout=5.0,
            frame_timeout=10.0,
            stream_timeout=30.0,
            min_fps_threshold=10.0,
            auto_reconnect=True,
            max_reconnect_attempts=0,
            reconnect_delay=5.0,
            frame_queue_size=3,
            drop_frames_on_full=True,
        )
        stream_id = f"{self.name}_{int(time.time())}"
        self.video_processor = EnhancedVideoStreamProcessor(stream_id, stream_config)
        self.video_processor.add_frame_callback(self._on_frame)
        self.video_processor.add_status_callback(self._on_status_change)
        self.video_processor.add_error_callback(self._on_error)
        self.video_processor.start()

        push_thread = threading.Thread(target=self._push_loop, daemon=True, name=f"Push-{self.name}")
        push_thread.start()

        logging.info(f"[{self.name}] 启动完成")

    def stop(self):
        logging.info(f"[{self.name}] 停止中...")
        self.is_running = False
        self._stop_event.set()

        if self.video_processor:
            self.video_processor.stop()
            self.video_processor = None

        self.inference_engine.reset_stream_tracking(self.stream_tracking_key)
        self._last_detection_overlays = []
        self._last_tracking_summary = {
            'track_count': 0,
            'track_ids': [],
            'classes': [],
        }
        self._last_infer_frame_id = -1

        self._close_ffmpeg()
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
        try:
            self._frame_id += 1
            fid = self._frame_id
            self._trace_stage(fid, 'frame_enter', shape=getattr(frame, 'shape', None), dtype=getattr(frame, 'dtype', None))
            interval = max(1, int(getattr(self.config, 'detection_inference_interval', 5)))

            rendered_frame = frame.copy()
            self._trace_stage(fid, 'frame_copied')
            detection_dict = {}

            if self.inference_engine.is_loaded() and (fid - self._last_infer_frame_id) >= interval:
                self._last_infer_frame_id = fid
                self._trace_stage(fid, 'infer_start', tracking=bool(getattr(self.config, 'tracking_enabled', False)))
                results = self.inference_engine.infer(frame, stream_key=self.stream_tracking_key)
                self._trace_stage(fid, 'infer_end', model_count=len(results))
                overlays = []
                total = 0
                class_names = set()
                track_ids = set()
                self._trace_stage(fid, 'postprocess_start')
                for aid, res in results.items():
                    cnt = self._count_detections(res)
                    total += cnt
                    detection_dict[f"detection_{aid}"] = float(cnt)
                    model_overlays = self._extract_detection_overlays(res, aid, fid=fid)
                    overlays.extend(model_overlays)
                    for overlay in model_overlays:
                        overlay_class = str(overlay.get('class_name', '')).strip()
                        if overlay_class:
                            class_names.add(overlay_class)
                        track_id = overlay.get('track_id')
                        if track_id not in (None, ''):
                            try:
                                track_ids.add(int(track_id))
                            except Exception:
                                track_ids.add(track_id)
                self._trace_stage(fid, 'postprocess_end', overlay_count=len(overlays), total=total)
                tracking_enabled = bool(getattr(self.config, 'tracking_enabled', False))
                alarm_count = len(track_ids) if tracking_enabled and track_ids else total
                detection_dict["alarm_any_detection"] = float(alarm_count)
                if alarm_count > 0:
                    logging.info(f"[{self.name}] 检测到目标: alarm_count={alarm_count}, track_ids={sorted(track_ids, key=lambda item: str(item)) if track_ids else []}, classes={sorted(class_names)}")
                self._last_tracking_summary = {
                    'track_count': int(len(track_ids)) if tracking_enabled else int(total),
                    'track_ids': sorted(track_ids, key=lambda item: str(item)),
                    'classes': sorted(class_names),
                }
                self._last_detection_overlays = overlays

            if self._last_detection_overlays:
                self._trace_stage(fid, 'draw_start', overlay_count=len(self._last_detection_overlays))
                rendered_frame = self._draw_detection_overlays(rendered_frame, self._last_detection_overlays)
                self._trace_stage(fid, 'draw_end')

            alert_target_info = None
            if detection_dict:
                overlay_classes = sorted({str(item.get('class_name', '')).strip() for item in self._last_detection_overlays if str(item.get('class_name', '')).strip()})
                class_text = ','.join(overlay_classes) if overlay_classes else 'unknown'
                alert_target_info = {
                    'classes': class_text,
                    'class_name': class_text,
                    'count': int(detection_dict.get("alarm_any_detection", 0.0)),
                    'track_count': int(self._last_tracking_summary.get('track_count', 0)),
                    'track_ids': list(self._last_tracking_summary.get('track_ids', [])),
                    'tracking_enabled': bool(getattr(self.config, 'tracking_enabled', False)),
                }

            rendered_frame = self._draw_ai_badge(rendered_frame)

            try:
                if self.push_queue.full():
                    try:
                        self.push_queue.get_nowait()
                    except queue.Empty:
                        pass
                self.push_queue.put_nowait(rendered_frame)
            except queue.Full:
                pass

            if self.alert_system.alert_handler:
                alert_frame = rendered_frame.copy()
                self.alert_system.alert_handler.collect_clip_frame(alert_frame)
                if detection_dict:
                    self.alert_system.process_frame_alerts(alert_frame, detection_dict, target_info=alert_target_info)

        except Exception as e:
            logging.error(f"[{self.name}] 帧处理异常: {e}")

    def _on_status_change(self, stream_id: str, status: VideoStreamStatus):
        logging.info(f"[{self.name}] 流状态: {status.value}")
        if status == VideoStreamStatus.INTERRUPTED:
            self.inference_engine.reset_stream_tracking(self.stream_tracking_key)
            self._last_detection_overlays = []
            self._last_tracking_summary = {
                'track_count': 0,
                'track_ids': [],
                'classes': [],
            }
            self._last_infer_frame_id = -1
            try:
                while not self.push_queue.empty():
                    self.push_queue.get_nowait()
            except Exception:
                pass
        elif status == VideoStreamStatus.READING:
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

    def _extract_detection_overlays(self, results, algo_id: str, fid: Optional[int] = None):
        overlays = []
        try:
            r = results[0] if isinstance(results, (list, tuple)) and results else results
            if r is None or not hasattr(r, 'boxes') or r.boxes is None:
                return overlays
            boxes = r.boxes
            if len(boxes) == 0:
                return overlays
            if fid is not None:
                self._trace_stage(fid, 'tensor_extract_start', algo_id=algo_id)
            xyxy = boxes.xyxy.cpu().numpy() if hasattr(boxes.xyxy, 'cpu') else np.asarray(boxes.xyxy)
            confs = boxes.conf.cpu().numpy() if hasattr(boxes.conf, 'cpu') else np.asarray(boxes.conf)
            clss = boxes.cls.cpu().numpy() if hasattr(boxes.cls, 'cpu') else np.asarray(boxes.cls)
            track_values = None
            if hasattr(boxes, 'id') and boxes.id is not None:
                track_values = boxes.id.cpu().numpy() if hasattr(boxes.id, 'cpu') else np.asarray(boxes.id)
            if fid is not None:
                self._trace_stage(fid, 'tensor_extract_end', algo_id=algo_id, box_count=len(xyxy))
            names = getattr(r, 'names', {})
            color = self._color_for_model(algo_id)
            for i in range(len(xyxy)):
                x1, y1, x2, y2 = map(int, xyxy[i])
                conf = float(confs[i])
                cls_id = int(clss[i])
                label = names.get(cls_id, str(cls_id)) if isinstance(names, dict) else str(cls_id)
                track_id = None
                if track_values is not None and i < len(track_values):
                    raw_track_id = track_values[i]
                    if raw_track_id is not None:
                        try:
                            track_id = int(raw_track_id)
                        except Exception:
                            track_id = str(raw_track_id)
                text = f"{algo_id}:{label} {conf:.2f}"
                if track_id is not None:
                    text = f"{algo_id}:ID{track_id} {label} {conf:.2f}"
                overlays.append({
                    'xyxy': (x1, y1, x2, y2),
                    'text': text,
                    'color': color,
                    'class_name': str(label),
                    'algo_id': str(algo_id),
                    'confidence': conf,
                    'track_id': track_id,
                })
        except Exception as e:
            logging.debug(f"[{self.name}] 提取绘框信息异常: {e}")
        return overlays

    def _draw_ai_badge(self, frame: np.ndarray) -> np.ndarray:
        try:
            badge_text = f"AI {self.name}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.8
            thickness = 2
            margin = 12
            (text_w, text_h), baseline = cv2.getTextSize(badge_text, font, font_scale, thickness)
            x1, y1 = margin, margin
            x2 = x1 + text_w + 24
            y2 = y1 + text_h + baseline + 20
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 140, 255), -1)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 2)
            cv2.putText(frame, badge_text, (x1 + 12, y2 - baseline - 10), font, font_scale, (255, 255, 255), thickness)
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
        overlays = self._extract_detection_overlays(results, algo_id)
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

        for attempt in range(2):
            codec = self._resolve_push_codec()
            hw = codec == VideoCodec.H264_NVENC.value
            cmd = _build_cmd(codec, hw)
            logging.info(f"[{self.name}] FFmpeg命令: {' '.join(cmd)}")
            try:
                pipe = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, bufsize=0)
            except Exception as e:
                logging.error(f"[{self.name}] 启动FFmpeg失败: {e}")
                break
            time.sleep(0.2)
            if pipe.poll() is not None:
                err = ''
                try:
                    err = pipe.stderr.read().decode('utf-8', errors='ignore')[:300]
                except Exception:
                    pass
                logging.error(f"[{self.name}] FFmpeg启动即退出(尝试{attempt + 1}): {err}")
                try:
                    pipe.kill()
                except Exception:
                    pass
                if attempt == 0 and ('nvenc' in err.lower() or hw):
                    self.config.video_codec = VideoCodec.LIBX264
                    logging.warning(f"[{self.name}] GPU编码不可用，回退libx264")
                continue
            self.pipe = pipe
            logging.info(f"[{self.name}] FFmpeg推流进程启动成功 -> {output_url}")
            return

        logging.error(f"[{self.name}] FFmpeg推流进程启动失败")

    def _resolve_push_codec(self) -> str:
        push_device = str(getattr(self.config, 'push_device', 'auto')).lower()
        if push_device == 'cpu':
            return VideoCodec.LIBX264.value
        if push_device == 'gpu':
            return VideoCodec.H264_NVENC.value
        if self.config.is_auto_codec_enabled():
            return VideoCodec.H264_NVENC.value
        return self.config.get_video_codec()

    def _close_ffmpeg(self):
        if self.pipe:
            try:
                if self.pipe.stdin and not self.pipe.stdin.closed:
                    self.pipe.stdin.close()
                self.pipe.terminate()
                self.pipe.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self.pipe.kill()
                self.pipe.wait()
            except Exception:
                pass
            self.pipe = None

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
        return '; '.join(reasons) if reasons else 'FFmpeg状态异常但未捕获到stderr'

    def _restart_ffmpeg(self) -> bool:
        self._close_ffmpeg()
        if self.output_url:
            self._open_ffmpeg(self.output_url)
            return self.pipe is not None
        return False

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
        max_repeat_frames = max(1, min(3, fps // 5 or 1))
        frame_count = 0

        while self.is_running:
            if self._push_reset_needed:
                try:
                    while not self.push_queue.empty():
                        self.push_queue.get_nowait()
                    last_frame = None
                    repeated_frame_count = 0
                    self._push_reset_needed = False
                except Exception:
                    pass

            now = time.perf_counter()
            sleep_time = next_push_time - now
            if sleep_time > 0:
                time.sleep(min(sleep_time, 0.02))
                continue
            if sleep_time < -interval * 3:
                next_push_time = now

            frame = None
            got_new_frame = False
            try:
                while True:
                    frame = self.push_queue.get_nowait()
                    got_new_frame = True
            except queue.Empty:
                pass

            if got_new_frame:
                last_frame = frame
                repeated_frame_count = 0
            elif last_frame is not None and repeated_frame_count < max_repeat_frames:
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
                logging.warning(f"[{self.name}] FFmpeg异常，尝试重启，原因: {failure_reason}")
                if not self._restart_ffmpeg():
                    time.sleep(2.0)
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
                next_push_time += interval

                if self.alert_system.alert_handler:
                    self.alert_system.alert_handler.update_latest_push_frame(frame)

                if frame_count % 500 == 0:
                    logging.info(f"[{self.name}] 已推流 {frame_count} 帧")
            except BrokenPipeError:
                failure_reason = self._describe_ffmpeg_failure()
                logging.error(f"[{self.name}] 推流管道断开，尝试恢复，原因: {failure_reason}")
                if not self._restart_ffmpeg():
                    time.sleep(3.0)
                next_push_time = time.perf_counter() + interval
            except Exception as e:
                logging.error(f"[{self.name}] 推流写入失败: {e}")
                time.sleep(0.5)
                next_push_time = time.perf_counter() + interval

        logging.info(f"[{self.name}] 推流线程结束")


class CameraStreamManager:
    """管理所有摄像头流的生命周期"""

    def __init__(self, config: CameraConfig):
        self.config = config
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
            inference_proxy = InferenceProxy(name, self.config)
            proc = StreamProcessor(scfg, self.config, inference_proxy, owns_inference_engine=True)
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
        logging.info(f"AI输出流已启用，拉流设备={getattr(config, 'pull_device', 'cpu')}，推流设备={getattr(config, 'push_device', 'auto')}，编码模式={config.get_video_codec()}")
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

    # 段错误时 atexit 不会触发，用 SIGSEGV 信号处理兜底清理子进程
    def _sigsegv_handler(signum, frame):
        _kill_child_processes()
        # 恢复默认处理，让进程正常 core dump
        signal.signal(signal.SIGSEGV, signal.SIG_DFL)
        os.kill(os.getpid(), signal.SIGSEGV)

    try:
        signal.signal(signal.SIGSEGV, _sigsegv_handler)
    except Exception:
        pass

    sys.exit(main())
