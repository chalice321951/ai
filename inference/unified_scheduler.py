# -*- coding: utf-8 -*-
"""
Unified multi-stream inference scheduler with micro-batching.
"""
import logging
import threading
import time
from typing import Any, Dict, List, Optional

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
        self._batch_size = max(1, int(getattr(config, 'inference_batch_size', 4) or 4))
        self._batch_wait_ms = max(0, int(getattr(config, 'inference_batch_wait_ms', 8) or 8))

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
        now = time.time()
        with self._lock:
            state = self._states.setdefault(key, self._new_stream_state())
            state["latest_frame"] = frame_copy
            state["latest_frame_id"] = int(frame_id or 0)
            state["algo_id"] = algo_id
            state["submitted_count"] += 1
            state["last_submit_ts"] = now
        self._wake_event.set()
        return True

    def get_latest_result(self, stream_key: str) -> Optional[Dict[str, Any]]:
        key = str(stream_key or "").strip()
        if not key:
            return None
        with self._lock:
            state = self._states.get(key)
            if not state or state.get("latest_result") is None:
                return None
            return {
                "frame_id": state.get("latest_result_frame_id", 0),
                "results": state.get("latest_result") or {},
                "result_ts": state.get("latest_result_ts", 0.0),
            }

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

    def _pick_batch(self) -> List[Dict[str, Any]]:
        tasks: List[Dict[str, Any]] = []
        with self._lock:
            candidates = []
            for key, state in self._states.items():
                if state.get("processing"):
                    continue
                if state.get("latest_frame") is None:
                    continue
                candidates.append((float(state.get("last_submit_ts", 0.0) or 0.0), key, state))

            candidates.sort(key=lambda item: item[0])
            for _, key, state in candidates[:self._batch_size]:
                tasks.append({
                    "stream_key": key,
                    "frame": state["latest_frame"],
                    "frame_id": state["latest_frame_id"],
                    "algo_id": state.get("algo_id"),
                })
                state["latest_frame"] = None
                state["processing"] = True
        return tasks

    def _store_results(self, tasks: List[Dict[str, Any]], outputs: Dict[str, Dict[str, Any]]):
        now = time.time()
        with self._lock:
            for task in tasks:
                stream_key = str(task.get("stream_key", "") or "")
                state = self._states.get(stream_key)
                if not state:
                    continue
                state["latest_result"] = outputs.get(stream_key, {}) or {}
                state["latest_result_frame_id"] = int(task.get("frame_id", 0) or 0)
                state["latest_result_ts"] = now
                state["processing"] = False
                state["completed_count"] += 1

    def _mark_failed(self, tasks: List[Dict[str, Any]]):
        with self._lock:
            for task in tasks:
                state = self._states.get(str(task.get("stream_key", "") or ""))
                if state:
                    state["processing"] = False

    def _worker_loop(self):
        while not self._stop_event.is_set():
            tasks = self._pick_batch()
            if not tasks:
                self._wake_event.wait(0.05)
                self._wake_event.clear()
                continue

            if len(tasks) < self._batch_size and self._batch_wait_ms > 0:
                time.sleep(self._batch_wait_ms / 1000.0)
                extra_tasks = self._pick_batch()
                if extra_tasks:
                    tasks.extend(extra_tasks[: max(0, self._batch_size - len(tasks))])

            try:
                outputs = self._engine.infer_batch(tasks)
                self._store_results(tasks, outputs)
            except Exception as e:
                logging.error(f"统一推理批量调度失败: {e}")
                self._mark_failed(tasks)
