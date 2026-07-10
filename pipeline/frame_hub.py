# -*- coding: utf-8 -*-
"""
FrameHub - 线程安全的最新帧广播器。

只保留每个流的最新帧，不堆积。多个 AlgoWorker 可以同时从 FrameHub 获取帧。
参考 ai_process_acl/video/realtime_fusion.py 的 FrameHub 实现。
"""
import threading
import logging
from typing import Dict, Optional, Any

import numpy as np


class FrameHub:
    """
    最新帧广播器。

    设计原则：
    1. 每个流只保留最新帧，旧帧被覆盖
    2. 线程安全：支持多个 AlgoWorker 并发读取
    3. 零拷贝广播：Worker 获取的是帧的引用（需自行拷贝）
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._frames: Dict[str, Dict[str, Any]] = {}

    def set_frame(self, stream_key: str, frame: np.ndarray, frame_id: int = 0) -> None:
        """
        设置流的最新帧（覆盖旧帧）。

        Args:
            stream_key: 流标识
            frame: 帧数据
            frame_id: 帧编号
        """
        if stream_key is None or frame is None:
            return

        # 拷贝帧数据，确保线程安全
        frame_copy = np.ascontiguousarray(frame, dtype=np.uint8)

        # 调试：记录首次提交的流
        with self._lock:
            is_new = stream_key not in self._frames
        if is_new:
            import logging
            logging.info(f"[FrameHub] 新流提交帧: stream_key={stream_key} fid={frame_id}")

        with self._lock:
            self._frames[stream_key] = {
                "frame": frame_copy,
                "frame_id": int(frame_id or 0),
            }

    def get_frame(self, stream_key: str) -> Optional[np.ndarray]:
        """
        获取流的最新帧（返回内部引用，零拷贝）。

        ⚠️ 注意：返回的是内部 numpy 数组的直接引用，不要原地修改。
        如果需要修改帧内容，请先调用 frame.copy()。
        YOLO 的 predict/track 不会原地修改输入帧，所以当前使用是安全的。

        Args:
            stream_key: 流标识

        Returns:
            帧数据引用，如果流不存在返回 None
        """
        with self._lock:
            entry = self._frames.get(stream_key)
            if entry is None:
                return None
            return entry["frame"]

    def get_frame_copy(self, stream_key: str) -> Optional[np.ndarray]:
        """
        获取流的最新帧的拷贝。

        Args:
            stream_key: 流标识

        Returns:
            帧数据的拷贝，如果流不存在返回 None
        """
        with self._lock:
            entry = self._frames.get(stream_key)
            if entry is None:
                return None
            return entry["frame"].copy()

    def get_frame_with_id(self, stream_key: str) -> Optional[tuple]:
        """
        获取流的最新帧及其帧编号（返回内部引用，零拷贝）。

        ⚠️ 注意：返回的 frame 是内部 numpy 数组的直接引用，不要原地修改。

        Args:
            stream_key: 流标识

        Returns:
            (frame, frame_id) 元组，如果流不存在或已被 take 返回 None
        """
        with self._lock:
            entry = self._frames.get(stream_key)
            if entry is None or entry.get("frame") is None:
                return None
            return entry["frame"], entry["frame_id"]

    def take_frame_with_id(self, stream_key: str) -> Optional[tuple]:
        """
        取出流的最新帧并清空 hub 中的引用（避免同一帧被反复推理）。

        与旧版 UnifiedInferenceScheduler._pick_batch 行为一致：
        取出帧后置为 None，等 camera.py 提交新帧才会再有内容。

        Args:
            stream_key: 流标识

        Returns:
            (frame, frame_id) 元组，如果流不存在或没有新帧返回 None
        """
        with self._lock:
            entry = self._frames.get(stream_key)
            if entry is None or entry.get("frame") is None:
                return None
            frame = entry["frame"]
            frame_id = entry["frame_id"]
            entry["frame"] = None  # 清空引用，避免被重复取
            return frame, frame_id

    def remove_stream(self, stream_key: str) -> None:
        """
        移除流的帧数据。

        Args:
            stream_key: 流标识
        """
        with self._lock:
            self._frames.pop(stream_key, None)

    def get_stream_keys(self) -> list:
        """
        获取所有流的标识列表。

        Returns:
            流标识列表
        """
        with self._lock:
            return list(self._frames.keys())

    def clear(self) -> None:
        """清空所有帧数据。"""
        with self._lock:
            self._frames.clear()
