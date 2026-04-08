# -*- coding: utf-8 -*-
"""
增强版视频流处理器 - 支持RTSP流，集成健康监控与自动重连
"""
import os
import cv2
import time
import threading
import logging
import queue
import contextlib
from urllib.parse import urlparse
from typing import Optional, Callable, List, Any
from dataclasses import dataclass
from enum import Enum

from .stream_health_monitor import StreamHealthMonitor, StreamHealthConfig, StreamHealthStatus

# 全局锁：保护 OPENCV_FFMPEG_CAPTURE_OPTIONS 环境变量设置 + VideoCapture 初始化
# 多路流并发打开时，环境变量是全局的，必须串行设置+打开，避免互相覆盖导致 native 崩溃
_capture_open_lock = threading.Lock()


class VideoStreamStatus(Enum):
    IDLE = "idle"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    READING = "reading"
    INTERRUPTED = "interrupted"
    ERROR = "error"
    RECONNECTING = "reconnecting"


@dataclass
class VideoStreamConfig:
    stream_url: str
    target_width: int = 1920
    target_height: int = 1080
    expected_fps: float = 25.0
    pull_device: str = "cpu"
    auto_detect_resolution: bool = True
    connection_timeout: float = 10.0
    read_timeout: float = 5.0
    frame_timeout: float = 10.0
    stream_timeout: float = 30.0
    min_fps_threshold: float = 10.0
    auto_reconnect: bool = True
    max_reconnect_attempts: int = 0  # 0 = 无限重连
    reconnect_delay: float = 5.0
    frame_queue_size: int = 3
    drop_frames_on_full: bool = True


class EnhancedVideoStreamProcessor:
    """增强版RTSP视频流处理器"""

    @contextlib.contextmanager
    def _suppress_capture_backend_logs(self):
        if os.name != 'nt':
            yield
            return

        stderr_fd = None
        saved_stderr_fd = None
        null_fd = None
        try:
            stderr_fd = os.dup(2)
            saved_stderr_fd = stderr_fd
            null_fd = os.open(os.devnull, os.O_WRONLY)
            os.dup2(null_fd, 2)
            yield
        except Exception:
            yield
        finally:
            if saved_stderr_fd is not None:
                try:
                    os.dup2(saved_stderr_fd, 2)
                except Exception:
                    pass
                try:
                    os.close(saved_stderr_fd)
                except Exception:
                    pass
            if null_fd is not None:
                try:
                    os.close(null_fd)
                except Exception:
                    pass

    def _build_capture_options(self) -> str:
        scheme = (urlparse(self.config.stream_url).scheme or '').lower()
        options = []
        if scheme == 'rtsp':
            options.extend([
                'rtsp_transport;tcp',
                'reorder_queue_size;1024',
                'buffer_size;2097152',
                'max_delay;1000000',
                'stimeout;10000000',
            ])
        if self.config.pull_device == 'gpu':
            options.extend([
                'hwaccel;cuda',
                'hwaccel_output_format;cuda',
            ])
        return '|'.join(options)

    def _open_capture(self):
        options = self._build_capture_options()
        os.environ.setdefault('OPENCV_LOG_LEVEL', 'ERROR')
        os.environ.setdefault('OPENCV_FFMPEG_LOGLEVEL', '0')
        logging.info(f"[{self.stream_id}] FFmpeg拉流参数: {options}")
        with _capture_open_lock:
            os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = options
            with self._suppress_capture_backend_logs():
                if hasattr(cv2, 'CAP_FFMPEG'):
                    cap = cv2.VideoCapture(self.config.stream_url, cv2.CAP_FFMPEG)
                else:
                    cap = cv2.VideoCapture(self.config.stream_url)
        return cap

    def __init__(self, stream_id: str, config: VideoStreamConfig):
        self.stream_id = stream_id
        self.config = config

        self.status = VideoStreamStatus.IDLE
        self.is_running = False
        self._stop_event = threading.Event()
        self._lock = threading.Lock()

        self.cap: Optional[cv2.VideoCapture] = None
        self.last_successful_read = 0.0
        self.consecutive_failures = 0
        self.max_consecutive_failures = 10

        self.frame_queue: queue.Queue = queue.Queue(maxsize=config.frame_queue_size)

        health_cfg = StreamHealthConfig(
            expected_fps=config.expected_fps,
            min_fps_threshold=config.min_fps_threshold,
            frame_timeout=config.frame_timeout,
            stream_timeout=config.stream_timeout,
            auto_reconnect=config.auto_reconnect,
            max_reconnect_attempts=config.max_reconnect_attempts,
            reconnect_delay=config.reconnect_delay,
        )
        self.health_monitor = StreamHealthMonitor(stream_id, health_cfg)
        self.health_monitor.add_status_change_callback(self._on_health_status_change)
        self.health_monitor.add_reconnect_callback(self._on_reconnect_needed)

        self.frame_callbacks: List[Callable] = []
        self.status_callbacks: List[Callable] = []
        self.error_callbacks: List[Callable] = []

        self.stats = {
            'frames_read': 0,
            'frames_dropped': 0,
            'read_errors': 0,
            'reconnect_count': 0,
            'start_time': 0.0,
            'last_frame_time': 0.0,
        }

    def add_frame_callback(self, cb: Callable):
        self.frame_callbacks.append(cb)

    def add_status_callback(self, cb: Callable):
        self.status_callbacks.append(cb)

    def add_error_callback(self, cb: Callable):
        self.error_callbacks.append(cb)

    def start(self) -> bool:
        with self._lock:
            if self.is_running:
                return False
            self.is_running = True
            self._stop_event.clear()
            self.stats['start_time'] = time.time()

        self.health_monitor.start_monitoring()

        self._read_thread = threading.Thread(
            target=self._read_loop,
            name=f"VideoReader-{self.stream_id}",
            daemon=True
        )
        self._read_thread.start()
        self._update_status(VideoStreamStatus.CONNECTING)
        return True

    def stop(self):
        with self._lock:
            if not self.is_running:
                return
            self.is_running = False
            self._stop_event.set()

        self.health_monitor.stop_monitoring()
        self._close_capture()

        if hasattr(self, '_read_thread') and self._read_thread.is_alive():
            self._read_thread.join(timeout=5.0)

        while not self.frame_queue.empty():
            try:
                self.frame_queue.get_nowait()
            except queue.Empty:
                break

        self._update_status(VideoStreamStatus.IDLE)

    def cleanup(self):
        self.stop()

    def get_status(self) -> VideoStreamStatus:
        return self.status

    def get_health_status(self) -> StreamHealthStatus:
        return self.health_monitor.get_status()

    # ------------------------------------------------------------------ #
    def _read_loop(self):
        logging.info(f"[{self.stream_id}] 读取线程启动: {self.config.stream_url}")
        while not self._stop_event.is_set():
            try:
                if not self._ensure_connection():
                    self._update_status(VideoStreamStatus.ERROR)
                    self._stop_event.wait(self.config.reconnect_delay)
                    continue

                if self._read_frame():
                    self.consecutive_failures = 0
                else:
                    self._handle_read_failure()
            except Exception as e:
                logging.error(f"[{self.stream_id}] 读取循环异常: {e}")
                self._handle_read_failure()

        logging.info(f"[{self.stream_id}] 读取线程结束")

    def _ensure_connection(self) -> bool:
        if self.cap and self.cap.isOpened():
            return True
        return self._connect()

    def _connect(self) -> bool:
        self._close_capture()
        self._update_status(VideoStreamStatus.CONNECTING)
        try:
            logging.info(f"[{self.stream_id}] 连接: {self.config.stream_url} (pull_device={self.config.pull_device})")
            self.cap = self._open_capture()
            if hasattr(cv2, 'CAP_PROP_BUFFERSIZE'):
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            if hasattr(cv2, 'CAP_PROP_OPEN_TIMEOUT_MSEC'):
                self.cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, self.config.connection_timeout * 1000)
            if hasattr(cv2, 'CAP_PROP_READ_TIMEOUT_MSEC'):
                self.cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, self.config.read_timeout * 1000)

            if self.cap.isOpened():
                logging.info(f"[{self.stream_id}] 连接成功")
                self._update_status(VideoStreamStatus.CONNECTED)
                return True
            else:
                logging.warning(f"[{self.stream_id}] 无法打开流")
                self._close_capture()
                return False
        except Exception as e:
            logging.error(f"[{self.stream_id}] 连接异常: {e}")
            self._close_capture()
            return False

    def _read_frame(self) -> bool:
        if not self.cap or not self.cap.isOpened():
            return False
        try:
            with _capture_open_lock:
                ret, frame = self.cap.read()
            if ret and frame is not None and getattr(frame, 'size', 0) > 0 and self.stats['frames_read'] == 0:
                logging.info(f"[{self.stream_id}] capture_stage=read_ok_first shape={getattr(frame, 'shape', None)} dtype={getattr(frame, 'dtype', None)}")
            if not ret or frame is None or frame.size == 0:
                logging.warning(f"[{self.stream_id}] 读到空帧，准备重连")
                return False

            now = time.time()
            is_first = self.stats['frames_read'] == 0
            self.stats['frames_read'] += 1
            self.stats['last_frame_time'] = now
            self.last_successful_read = now

            if is_first:
                h, w = frame.shape[:2]
                cost = int((now - self.stats['start_time']) * 1000)
                logging.info(f"[{self.stream_id}] 首帧 cost={cost}ms size={w}x{h}")

            self.health_monitor.report_frame_received()

            if self.status != VideoStreamStatus.READING:
                self._update_status(VideoStreamStatus.READING)

            self._enqueue_frame(frame)

            for cb in self.frame_callbacks:
                try:
                    cb(self.stream_id, frame)
                except Exception as e:
                    logging.error(f"[{self.stream_id}] 帧回调异常: {e}")

            return True
        except cv2.error as e:
            logging.warning(f"[{self.stream_id}] OpenCV解码异常，准备重连: {e}")
            self._close_capture()
            return False
        except Exception as e:
            logging.error(f"[{self.stream_id}] 读帧异常: {e}")
            self._close_capture()
            return False

    def _enqueue_frame(self, frame):
        try:
            if self.config.drop_frames_on_full and self.frame_queue.full():
                try:
                    self.frame_queue.get_nowait()
                    self.stats['frames_dropped'] += 1
                except queue.Empty:
                    pass
            self.frame_queue.put_nowait(frame)
        except queue.Full:
            self.stats['frames_dropped'] += 1

    def _handle_read_failure(self):
        self.consecutive_failures += 1
        self.stats['read_errors'] += 1
        self.health_monitor.report_error()

        if self.consecutive_failures >= self.max_consecutive_failures:
            logging.warning(f"[{self.stream_id}] 连续失败 {self.consecutive_failures} 次，标记为中断")
            self._update_status(VideoStreamStatus.INTERRUPTED)
            self._close_capture()
            self._stop_event.wait(self.config.reconnect_delay)
        else:
            self._stop_event.wait(0.1)

    def _close_capture(self):
        if self.cap:
            try:
                with _capture_open_lock:
                    self.cap.release()
            except Exception:
                pass
            self.cap = None

    def _update_status(self, new_status: VideoStreamStatus):
        if self.status == new_status:
            return
        old = self.status
        self.status = new_status
        logging.debug(f"[{self.stream_id}] 状态: {old.value} -> {new_status.value}")
        for cb in self.status_callbacks:
            try:
                cb(self.stream_id, new_status)
            except Exception as e:
                logging.error(f"[{self.stream_id}] 状态回调异常: {e}")

    def _on_health_status_change(self, stream_id: str, health_status: StreamHealthStatus):
        if health_status in (StreamHealthStatus.INTERRUPTED, StreamHealthStatus.ERROR):
            self._update_status(VideoStreamStatus.INTERRUPTED)
            self._close_capture()

    def _on_reconnect_needed(self, stream_id: str):
        self.stats['reconnect_count'] += 1
        self._close_capture()
        self._update_status(VideoStreamStatus.RECONNECTING)
        self._stop_event.wait(self.config.reconnect_delay)
