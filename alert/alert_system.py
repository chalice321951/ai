# -*- coding: utf-8 -*-
"""
告警系统模块 - 检测告警、视频剪辑、图片保存
"""
import cv2
import logging
import os
import signal
import subprocess
import threading
import time
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Callable, Any

import numpy as np


class AlertLevel(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class AlertType(Enum):
    COUNT_THRESHOLD = "count_threshold"
    DURATION_THRESHOLD = "duration_threshold"
    CUSTOM = "custom"


@dataclass
class AlertRule:
    rule_id: str
    alert_type: AlertType
    alert_level: AlertLevel
    threshold_value: float
    duration_seconds: float = 0.0
    cooldown_seconds: float = 10.0
    enabled: bool = True
    description: str = ""

    def __post_init__(self):
        if isinstance(self.alert_type, str):
            self.alert_type = AlertType(self.alert_type)
        if isinstance(self.alert_level, str):
            self.alert_level = AlertLevel(self.alert_level)


@dataclass
class AlertEvent:
    event_id: str
    rule_id: str
    alert_type: AlertType
    alert_level: AlertLevel
    current_value: float
    threshold_value: float
    timestamp: float
    message: str
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


def create_count_threshold_rule(rule_id: str, threshold: float, description: str = "",
                                level: AlertLevel = AlertLevel.MEDIUM,
                                cooldown: float = 10.0) -> AlertRule:
    return AlertRule(
        rule_id=rule_id,
        alert_type=AlertType.COUNT_THRESHOLD,
        alert_level=level,
        threshold_value=threshold,
        cooldown_seconds=cooldown,
        description=description or f"数量超过阈值 {threshold}",
    )


class AlertHandler:
    """告警处理器：保存图片、剪辑视频"""

    def __init__(self, stream_cfg: dict, result_path: str, config=None):
        self.stream_cfg = stream_cfg
        self.result_path = result_path
        self.config = config
        os.makedirs(result_path, exist_ok=True)

        self.platform_client = None
        try:
            from nan.camera_server import PlatformApiClient
            self.platform_client = PlatformApiClient(config=config)
        except Exception as e:
            logging.error(f"初始化平台告警客户端失败: {e}")

        alarm_cfg = config.get_alarm_config() if config else {}
        self.clip_seconds = alarm_cfg.get('video_clip_seconds', 10)
        self.buffer_seconds = alarm_cfg.get('video_buffer_seconds', 12)
        self.pre_alert_seconds = alarm_cfg.get('video_pre_alert_seconds', 5)
        self.post_alert_seconds = alarm_cfg.get('video_post_alert_seconds', 5)
        fps = getattr(config, 'fps', 25) if config else 25
        self.clip_fps = fps
        self.max_buffer_frames = self.buffer_seconds * fps

        self.frame_buffer: List[dict] = []
        self.buffer_lock = threading.Lock()
        self.clip_jobs: Dict[str, dict] = {}
        self.clip_jobs_lock = threading.Lock()
        self._latest_push_frame: Optional[np.ndarray] = None
        self.writer_threads: List[threading.Thread] = []
        self._frame_sequence = 0
        self.validation_image_mae_threshold = 8.0
        self.validation_video_mae_threshold = 12.0

    def update_latest_push_frame(self, frame: np.ndarray):
        if frame is not None:
            self._latest_push_frame = frame

    def collect_clip_frame(self, frame: np.ndarray, original_frame: np.ndarray = None,
                           frame_ts: Optional[float] = None):
        """收集帧到循环缓冲区，并推进活跃剪辑任务"""
        try:
            f = frame if frame is not None else original_frame
            if f is None:
                return
            now = float(frame_ts if frame_ts is not None else time.time())
            with self.buffer_lock:
                self._frame_sequence += 1
                current_seq = self._frame_sequence
                self.frame_buffer.append({'frame': f.copy(), 'timestamp': now, 'sequence': current_seq})
                cutoff = now - self.buffer_seconds
                while self.frame_buffer and self.frame_buffer[0]['timestamp'] <= cutoff:
                    self.frame_buffer.pop(0)
                if len(self.frame_buffer) > self.max_buffer_frames:
                    self.frame_buffer.pop(0)

            completed = []
            with self.clip_jobs_lock:
                for tid, job in self.clip_jobs.items():
                    if now <= job['deadline']:
                        if job.pop('skip_next_append', False):
                            continue
                        if current_seq > int(job.get('trigger_sequence', 0) or 0):
                            job['frames'].append(f.copy())
                    else:
                        completed.append(tid)
                done_jobs = [self.clip_jobs.pop(tid) for tid in completed]

            for job in done_jobs:
                self._finalize_clip_job(job)
        except Exception as e:
            logging.error(f"collect_clip_frame 异常: {e}")

    def handle_alert(self, alert_event: AlertEvent, frame: np.ndarray = None,
                     target_info: dict = None, frame_ts: Optional[float] = None):
        try:
            frame_for_alert = frame if frame is not None else self._latest_push_frame
            alert_image_path = self._save_alert_image(alert_event, frame_for_alert)
            if frame_for_alert is not None:
                trigger_ts = float(frame_ts if frame_ts is not None else time.time())
                self._start_clip_job(alert_event, frame_for_alert, alert_image_path, trigger_ts, target_info=target_info)
        except Exception as e:
            logging.error(f"handle_alert 异常: {e}")

    def _save_alert_image(self, event: AlertEvent, frame: np.ndarray) -> str:
        if frame is None:
            return ""
        try:
            ts = time.strftime("%Y%m%d-%H%M%S")
            fname = f"alert_{event.rule_id}_{ts}.jpg"
            path = os.path.join(self.result_path, fname)
            cv2.imwrite(path, frame)
            logging.info(f"告警图片已保存: {path}")
            return path
        except Exception as e:
            logging.error(f"保存告警图片失败: {e}")
            return ""

    def _start_clip_job(
        self,
        event: AlertEvent,
        frame: np.ndarray,
        image_path: str,
        trigger_ts: float,
        target_info: Optional[dict] = None,
    ):
        pre_frames = []
        trigger_sequence = 0
        with self.buffer_lock:
            cutoff = trigger_ts - self.pre_alert_seconds
            for fd in self.frame_buffer:
                fd_ts = float(fd.get('timestamp', 0.0) or 0.0)
                fd_seq = int(fd.get('sequence', 0) or 0)
                if fd_ts > trigger_ts:
                    continue
                if fd_ts >= cutoff and fd_seq > 0:
                    pre_frames.append(fd['frame'].copy())
                    trigger_sequence = max(trigger_sequence, fd_seq)
            if not pre_frames:
                for fd in self.frame_buffer:
                    fd_ts = float(fd.get('timestamp', 0.0) or 0.0)
                    fd_seq = int(fd.get('sequence', 0) or 0)
                    if fd_ts <= trigger_ts:
                        pre_frames.append(fd['frame'].copy())
                        trigger_sequence = max(trigger_sequence, fd_seq)

        if len(pre_frames) > 1:
            pre_frames = pre_frames[:-1]

        job = {
            'target_id': event.rule_id,
            'event': event,
            'class': event.metadata.get('class_name', 'unknown') if event.metadata else 'unknown',
            'frames': pre_frames + [frame.copy()],
            'trigger_index': len(pre_frames),
            'deadline': trigger_ts + self.post_alert_seconds,
            'max_frames': self.buffer_seconds * self.clip_fps,
            'alert_image_path': image_path,
            'skip_next_append': True,
            'trigger_sequence': trigger_sequence,
            'validation_boxes': list((target_info or {}).get('_validation_boxes', []) or []),
        }
        with self.clip_jobs_lock:
            self.clip_jobs[event.rule_id] = job
        logging.info(f"剪辑任务已创建: {event.rule_id}, 预告警帧={len(pre_frames)}")

    def _finalize_clip_job(self, job: dict):
        def _run():
            try:
                frames = job['frames']
                if not frames:
                    return
                target_frames = self.clip_seconds * self.clip_fps
                trigger_index = max(0, min(int(job.get('trigger_index', len(frames) - 1) or 0), len(frames) - 1))
                pre_target = max(0, target_frames // 2)
                post_target = max(0, target_frames - pre_target - 1)

                pre_frames = frames[:trigger_index]
                trigger_frame = frames[trigger_index]
                post_frames = frames[trigger_index + 1:]

                if len(pre_frames) >= pre_target:
                    pre_frames = pre_frames[-pre_target:]
                elif pre_frames:
                    pre_frames = [pre_frames[0]] * (pre_target - len(pre_frames)) + pre_frames
                else:
                    pre_frames = [trigger_frame.copy() for _ in range(pre_target)]

                if len(post_frames) >= post_target:
                    post_frames = post_frames[:post_target]
                elif post_frames:
                    post_frames = post_frames + [post_frames[-1]] * (post_target - len(post_frames))
                else:
                    post_frames = [trigger_frame.copy() for _ in range(post_target)]

                frames = pre_frames + [trigger_frame] + post_frames

                ts = time.strftime("%Y%m%d-%H%M%S")
                clip_path = os.path.join(
                    self.result_path,
                    f"clip_{job['target_id']}_{ts}.mp4"
                )
                video_saved = self._write_video(frames, clip_path)
                logging.info(f"告警视频已保存: {clip_path}")

                # ── MinIO 上传 ───────────────────────────────
                self._validate_alert_assets(
                    alert_image_path=job.get('alert_image_path', ''),
                    alert_video_path=clip_path if video_saved else '',
                    trigger_frame=trigger_frame,
                    validation_boxes=job.get('validation_boxes', []) or [],
                )
                image_url, video_url = self._upload_alert_assets(
                    alert_image_path=job.get('alert_image_path', ''),
                    alert_video_path=clip_path if video_saved else '',
                )
                self._report_alarm_event(job.get('event'), image_url, video_url)
            except Exception as e:
                logging.error(f"_finalize_clip_job 异常: {e}")

        t = threading.Thread(target=_run, daemon=True)
        t.start()
        self.writer_threads.append(t)

    def _validate_alert_assets(
        self,
        alert_image_path: str,
        alert_video_path: str,
        trigger_frame: Optional[np.ndarray],
        validation_boxes: Optional[List[dict]] = None,
    ):
        if trigger_frame is None:
            logging.warning("告警质检跳过: 缺少触发帧")
            return
        try:
            image_ok, image_msg = self._validate_alert_image(alert_image_path, trigger_frame, validation_boxes or [])
            video_ok, video_msg = self._validate_alert_video(alert_video_path, trigger_frame)
            if image_ok and video_ok:
                logging.info(f"告警质检通过: {image_msg}; {video_msg}")
            else:
                logging.error(f"告警质检失败: {image_msg}; {video_msg}")
        except Exception as e:
            logging.error(f"告警质检异常: {e}")

    def _validate_alert_image(self, alert_image_path: str, trigger_frame: np.ndarray, validation_boxes: List[dict]):
        if not alert_image_path or not os.path.exists(alert_image_path):
            return False, "报警图片不存在"
        image = cv2.imread(alert_image_path)
        if image is None:
            return False, "报警图片无法读取"
        mae = self._compute_frame_mae(image, trigger_frame)
        box_ok, box_msg = self._validate_box_pixels(image, validation_boxes)
        ok = mae <= self.validation_image_mae_threshold and box_ok
        return ok, f"图片MAE={mae:.2f}, {box_msg}"

    def _validate_alert_video(self, alert_video_path: str, trigger_frame: np.ndarray):
        if not alert_video_path or not os.path.exists(alert_video_path):
            return False, "报警视频不存在"
        middle_frame = self._read_video_middle_frame(alert_video_path)
        if middle_frame is None:
            return False, "报警视频中间帧无法读取"
        mae = self._compute_frame_mae(middle_frame, trigger_frame)
        ok = mae <= self.validation_video_mae_threshold
        return ok, f"视频中帧MAE={mae:.2f}"

    def _compute_frame_mae(self, frame_a: np.ndarray, frame_b: np.ndarray) -> float:
        if frame_a is None or frame_b is None:
            return float('inf')
        if frame_a.shape[:2] != frame_b.shape[:2]:
            frame_a = cv2.resize(frame_a, (frame_b.shape[1], frame_b.shape[0]))
        diff = np.abs(frame_a.astype(np.float32) - frame_b.astype(np.float32))
        return float(diff.mean())

    def _validate_box_pixels(self, frame: np.ndarray, validation_boxes: List[dict]):
        if frame is None:
            return False, "报警图片为空"
        if not validation_boxes:
            return True, "未提供验框信息"
        height, width = frame.shape[:2]
        matched = 0
        checked = 0
        for box in validation_boxes:
            xyxy = list(box.get('xyxy', []) or [])
            color = np.asarray(list(box.get('color', []) or []), dtype=np.int16)
            if len(xyxy) != 4 or color.shape[0] != 3:
                continue
            x1, y1, x2, y2 = [int(v) for v in xyxy]
            x1 = max(0, min(width - 1, x1))
            x2 = max(0, min(width - 1, x2))
            y1 = max(0, min(height - 1, y1))
            y2 = max(0, min(height - 1, y2))
            if x2 <= x1 or y2 <= y1:
                continue
            checked += 1
            top = frame[y1:min(y1 + 2, height), x1:x2 + 1]
            bottom = frame[max(0, y2 - 1):y2 + 1, x1:x2 + 1]
            left = frame[y1:y2 + 1, x1:min(x1 + 2, width)]
            right = frame[y1:y2 + 1, max(0, x2 - 1):x2 + 1]
            border_pixels = np.concatenate([
                top.reshape(-1, 3),
                bottom.reshape(-1, 3),
                left.reshape(-1, 3),
                right.reshape(-1, 3),
            ], axis=0).astype(np.int16)
            if border_pixels.size == 0:
                continue
            delta = np.abs(border_pixels - color)
            hits = np.all(delta <= 90, axis=1)
            hit_ratio = float(hits.mean()) if hits.size else 0.0
            if hit_ratio >= 0.15:
                matched += 1
        if checked == 0:
            return True, "未提供有效验框信息"
        if matched > 0:
            return True, f"验框通过={matched}/{checked}"
        return False, f"验框失败=0/{checked}"

    def _read_video_middle_frame(self, video_path: str) -> Optional[np.ndarray]:
        cap = None
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap or not cap.isOpened():
                return None
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            middle_index = max(0, frame_count // 2)
            if frame_count > 0:
                cap.set(cv2.CAP_PROP_POS_FRAMES, middle_index)
            ok, frame = cap.read()
            if ok and frame is not None:
                return frame
            return None
        except Exception:
            return None
        finally:
            try:
                if cap is not None:
                    cap.release()
            except Exception:
                pass

    def _upload_alert_assets(self, alert_image_path: str, alert_video_path: str):
        """上传图片/视频到 MinIO。"""
        image_url = ""
        video_url = ""
        try:
            from nan import minio_update

            if alert_image_path and os.path.exists(alert_image_path):
                try:
                    image_url = minio_update.minio_interface(
                        self.stream_cfg, "alarm",
                        os.path.basename(alert_image_path), alert_image_path,
                        minio_config=self.config,
                    )
                    if image_url:
                        logging.info(f"告警图片上传成功: {image_url}")
                except Exception as e:
                    logging.error(f"图片上传失败: {e}")

            if alert_video_path and os.path.exists(alert_video_path):
                try:
                    video_url = minio_update.minio_interface(
                        self.stream_cfg, "clip",
                        os.path.basename(alert_video_path), alert_video_path,
                        minio_config=self.config,
                    )
                    if video_url:
                        logging.info(f"告警视频上传成功: {video_url}")
                except Exception as e:
                    logging.error(f"视频上传失败: {e}")
        except Exception as e:
            logging.error(f"_upload_alert_assets 异常: {e}")
        return image_url, video_url

    def _report_alarm_event(self, event: AlertEvent, image_url: str, video_url: str):
        if event is None:
            return
        if not image_url and not video_url:
            logging.warning("未获取到可上报的媒体 URL，跳过报警接口上报")
            return
        if not self.platform_client:
            logging.warning("平台告警客户端未初始化，跳过报警接口上报")
            return
        try:
            success = self.platform_client.report_alarm(self.stream_cfg, event, image_url, video_url)
            if success:
                logging.info(f"告警上报成功: {event.event_id}")
        except Exception as e:
            logging.error(f"报警接口上报异常: {e}")

    def _write_video(self, frames: List[np.ndarray], output_path: str) -> bool:
        if not frames:
            try:
                if 'pipe' in locals() and pipe is not None and pipe.poll() is None:
                    if os.name != 'nt':
                        os.killpg(os.getpgid(pipe.pid), signal.SIGKILL)
                    else:
                        pipe.kill()
            except Exception:
                pass
            return False
        try:
            h, w = frames[0].shape[:2]
            n = len(frames)
            duration = max(1.0, float(self.clip_seconds))
            input_fps = max(1.0, round(n / duration, 3))

            cmd = [
                'ffmpeg', '-y',
                '-f', 'rawvideo', '-pix_fmt', 'bgr24',
                '-s', f'{w}x{h}',
                '-framerate', str(input_fps),
                '-i', 'pipe:0',
                '-c:v', 'libx264', '-pix_fmt', 'yuv420p',
                '-crf', '23', '-preset', 'ultrafast',
                '-movflags', '+faststart',
                output_path
            ]
            pipe = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                bufsize=0,
                start_new_session=(os.name != 'nt'),
            )
            for f in frames:
                f = np.ascontiguousarray(f, dtype=np.uint8)
                pipe.stdin.write(f.tobytes())
            pipe.stdin.close()
            pipe.wait(timeout=60)
            return pipe.returncode == 0
        except Exception as e:
            logging.error(f"写入视频失败: {e}")
            return False


class AlertSystem:
    """告警系统主类"""

    def __init__(self, config=None):
        self.config = config
        self._rules: Dict[str, AlertRule] = {}
        self._last_alert_times: Dict[str, float] = {}
        self._trigger_times: Dict[str, float] = {}
        self._lock = threading.RLock()
        self.alert_handler: Optional[AlertHandler] = None
        self._last_uninit_warn = 0.0

    def initialize_alert_handler(self, stream_cfg: dict, result_path: str):
        self.alert_handler = AlertHandler(stream_cfg, result_path, self.config)
        logging.info(f"告警处理器初始化完成: {result_path}")

    def add_rule(self, rule: AlertRule):
        with self._lock:
            self._rules[rule.rule_id] = rule
            logging.info(f"告警规则已添加: {rule.rule_id}")

    def process_frame_alerts(self, frame: np.ndarray, detection_dict: dict,
                             original_frame: np.ndarray = None,
                             target_info: dict = None,
                             frame_ts: Optional[float] = None):
        if not self.alert_handler:
            now = time.time()
            if now - self._last_uninit_warn > 300:
                logging.warning("告警处理器未初始化，跳过")
                self._last_uninit_warn = now
            return

        now = time.time()
        with self._lock:
            for rule_id, rule in self._rules.items():
                if not rule.enabled:
                    continue
                val = detection_dict.get(rule_id, 0.0)
                triggered = val >= rule.threshold_value

                if triggered:
                    if rule_id not in self._trigger_times:
                        self._trigger_times[rule_id] = now
                    duration = now - self._trigger_times[rule_id]
                    if duration < rule.duration_seconds:
                        continue
                    last = self._last_alert_times.get(rule_id, 0.0)
                    if now - last < rule.cooldown_seconds:
                        continue
                    self._last_alert_times[rule_id] = now
                    event = AlertEvent(
                        event_id=f"{rule_id}_{int(now)}",
                        rule_id=rule_id,
                        alert_type=rule.alert_type,
                        alert_level=rule.alert_level,
                        current_value=val,
                        threshold_value=rule.threshold_value,
                        timestamp=now,
                        message=f"告警: {rule.description} 当前值={val:.2f}",
                        metadata={'target_info': target_info} if target_info else {},
                    )
                    logging.warning(f"[AlertSystem] 触发告警: {rule_id} val={val:.2f}")
                    try:
                        self.alert_handler.handle_alert(event, frame, target_info, frame_ts=frame_ts)
                    except Exception as e:
                        logging.error(f"告警处理异常: {e}")
                else:
                    self._trigger_times.pop(rule_id, None)
