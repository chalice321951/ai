# -*- coding: utf-8 -*-
"""
流健康监控模块
"""
import time
import threading
import logging
from enum import Enum
from dataclasses import dataclass, field
from typing import Callable, List, Optional


class StreamHealthStatus(Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    INTERRUPTED = "interrupted"
    ERROR = "error"


@dataclass
class StreamHealthConfig:
    expected_fps: float = 25.0
    min_fps_threshold: float = 10.0
    frame_timeout: float = 10.0
    stream_timeout: float = 30.0
    auto_reconnect: bool = True
    max_reconnect_attempts: int = 5
    reconnect_delay: float = 5.0


class StreamHealthMonitor:
    """流健康监控器"""

    def __init__(self, stream_id: str, config: StreamHealthConfig):
        self.stream_id = stream_id
        self.config = config
        self.status = StreamHealthStatus.HEALTHY

        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._monitor_thread: Optional[threading.Thread] = None

        self._last_frame_time = 0.0
        self._frame_count = 0
        self._error_count = 0
        self._reconnect_count = 0

        self._status_callbacks: List[Callable] = []
        self._reconnect_callbacks: List[Callable] = []

    def add_status_change_callback(self, cb: Callable):
        self._status_callbacks.append(cb)

    def add_reconnect_callback(self, cb: Callable):
        self._reconnect_callbacks.append(cb)

    def start_monitoring(self):
        self._stop_event.clear()
        self._monitor_thread = threading.Thread(
            target=self._monitor_loop,
            name=f"HealthMonitor-{self.stream_id}",
            daemon=True
        )
        self._monitor_thread.start()

    def stop_monitoring(self):
        self._stop_event.set()
        if self._monitor_thread and self._monitor_thread.is_alive():
            self._monitor_thread.join(timeout=3.0)

    def report_frame_received(self):
        with self._lock:
            self._last_frame_time = time.time()
            self._frame_count += 1

    def report_error(self, msg: str = ""):
        with self._lock:
            self._error_count += 1

    def get_status(self) -> StreamHealthStatus:
        return self.status

    def get_metrics(self) -> dict:
        with self._lock:
            return {
                'frame_count': self._frame_count,
                'error_count': self._error_count,
                'reconnect_count': self._reconnect_count,
                'last_frame_time': self._last_frame_time,
            }

    def _monitor_loop(self):
        while not self._stop_event.is_set():
            self._check_health()
            self._stop_event.wait(1.0)

    def _check_health(self):
        now = time.time()
        with self._lock:
            last = self._last_frame_time

        if last == 0.0:
            return

        elapsed = now - last
        if elapsed > self.config.frame_timeout:
            new_status = StreamHealthStatus.INTERRUPTED
        elif elapsed > self.config.frame_timeout * 0.5:
            new_status = StreamHealthStatus.DEGRADED
        else:
            new_status = StreamHealthStatus.HEALTHY

        if new_status != self.status:
            old = self.status
            self.status = new_status
            logging.info(f"[HealthMonitor] {self.stream_id}: {old.value} -> {new_status.value}")
            for cb in self._status_callbacks:
                try:
                    cb(self.stream_id, new_status)
                except Exception as e:
                    logging.error(f"健康状态回调异常: {e}")

            if new_status == StreamHealthStatus.INTERRUPTED and self.config.auto_reconnect:
                with self._lock:
                    self._reconnect_count += 1
                for cb in self._reconnect_callbacks:
                    try:
                        cb(self.stream_id)
                    except Exception as e:
                        logging.error(f"重连回调异常: {e}")
