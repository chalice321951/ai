# -*- coding: utf-8 -*-
"""
Unified multi-stream inference scheduler.

The scheduler keeps only the latest frame per stream, runs inference in a
shared worker thread, and writes the latest result back to each stream cache.
"""
import logging
import threading
import time
from typing import Any, Dict, Optional

import numpy as np

from .inference_engine import InferenceEngine


class UnifiedInferenceScheduler:
    """Shared inference scheduler for multiple streams."""

    def __init__(self, config):
        self.config = config
        self._engine = InferenceEngine(config)
        self._lock = threading.Lock()
        self._wake_event = threading.Event()
        self._stop_event = threading.Event()
        self._states: Dict[str, Dict[str, Any]] = {}
        self._worker_thread: Optional[threading.Thread] = None

        if self._engine.is_loaded():
            self._worker_thread = threading.Thread(
                target=self._worker_loop,
                daemon=True,
                name="UnifiedInferenceScheduler",
            )
            self._worker_thread.start()
            logging.info("统一推理调度器已启动")
        else:
            logging.warning("统一推理调度器未启动：推理引擎未加载")

    def is_loaded(self) -> bool:
        return self._engine.is_loaded()

    def ensure_stream(self, stream_key: str):
        key = str(stream_key or "").strip()
        if not key:
            return
        with self._lock:
            self._states.setdefault(key, self._new_stream_state())

    def submit_frame(self, stream_key: str, frame: np.ndarray, algo_id: str = None, frame_id: int = 0) -> bool:
        key = str(stream_key or "").strip()
        if not key or frame is None or not self.is_loaded():
            return False

        frame_copy = np.ascontiguousarray(frame, dtype=np.uint8)
        with self._lock:
            state = self._states.setdefault(key, self._new_stream_state())
            state["latest_frame"] = frame_copy
            state["latest_frame_id"] = int(frame_id or 0)
            state["algo_id"] = algo_id
            state["submitted_count"] += 1
            state["last_submit_ts"] = time.time()
        self._wake_event.set()
        return True

    def get_latest_result(self, stream_key: str) -> Optional[Dict[str, Any]]:
        key = str(stream_key or "").strip()
        if not key:
            return None
        with self._lock:
            state = self._states.get(key)
            if not state:
                return None
            result = state.get("latest_result")
            if result is None:
                return None
            return {
                "frame_id": state.get("latest_result_frame_id", 0),
                "results": result,
                "result_ts": state.get("latest_result_ts", 0.0),
            }

    def clear_stream_result(self, stream_key: str):
        key = str(stream_key or "").strip()
        if not key:
            return
        with self._lock:
            state = self._states.get(key)
            if not state:
                return
            state["latest_result"] = None
            state["latest_result_frame_id"] = 0
            state["latest_result_ts"] = 0.0

    def reset_stream_tracking(self, stream_key: str):
        key = str(stream_key or "").strip()
        if not key:
            return
        with self._lock:
            state = self._states.get(key)
            if state:
                state["latest_frame"] = None
                state["latest_frame_id"] = 0
                state["latest_result"] = None
                state["latest_result_frame_id"] = 0
                state["latest_result_ts"] = 0.0
                state["processing"] = False
        self._engine.reset_stream_tracking(key)

    def remove_stream(self, stream_key: str):
        key = str(stream_key or "").strip()
        if not key:
            return
        with self._lock:
            self._states.pop(key, None)
        self._engine.reset_stream_tracking(key)

    def cleanup(self):
        self._stop_event.set()
        self._wake_event.set()
        if self._worker_thread is not None and self._worker_thread.is_alive():
            self._worker_thread.join(timeout=5.0)
        self._engine.cleanup()

    def _new_stream_state(self) -> Dict[str, Any]:
        return {
            "latest_frame": None,
            "latest_frame_id": 0,
            "latest_result": None,
            "latest_result_frame_id": 0,
            "latest_result_ts": 0.0,
            "algo_id": None,
            "processing": False,
            "submitted_count": 0,
            "completed_count": 0,
            "last_submit_ts": 0.0,
        }

    def _pick_next_task(self) -> Optional[Dict[str, Any]]:
        with self._lock:
            selected_key = None
            selected_state = None
            oldest_ts = None
            for key, state in self._states.items():
                if state.get("processing"):
                    continue
                if state.get("latest_frame") is None:
                    continue
                state_ts = float(state.get("last_submit_ts", 0.0) or 0.0)
                if selected_state is None or state_ts < oldest_ts:
                    selected_key = key
                    selected_state = state
                    oldest_ts = state_ts

            if selected_state is None:
                return None

            task = {
                "stream_key": selected_key,
                "frame": selected_state["latest_frame"],
                "frame_id": selected_state["latest_frame_id"],
                "algo_id": selected_state.get("algo_id"),
            }
            selected_state["latest_frame"] = None
            selected_state["processing"] = True
            return task

    def _store_result(self, stream_key: str, frame_id: int, results: Dict[str, Any]):
        with self._lock:
            state = self._states.get(stream_key)
            if not state:
                return
            state["latest_result"] = results or {}
            state["latest_result_frame_id"] = int(frame_id or 0)
            state["latest_result_ts"] = time.time()
            state["processing"] = False
            state["completed_count"] += 1

    def _mark_failed(self, stream_key: str):
        with self._lock:
            state = self._states.get(stream_key)
            if state:
                state["processing"] = False

    def _worker_loop(self):
        while not self._stop_event.is_set():
            task = self._pick_next_task()
            if task is None:
                self._wake_event.wait(0.05)
                self._wake_event.clear()
                continue

            stream_key = task["stream_key"]
            try:
                results = self._engine.infer(
                    frame=task["frame"],
                    algo_id=task.get("algo_id"),
                    stream_key=stream_key,
                )
                self._store_result(stream_key, task.get("frame_id", 0), results)
            except Exception as e:
                logging.error(f"[{stream_key}] 统一推理调度失败: {e}")
                self._mark_failed(stream_key)
