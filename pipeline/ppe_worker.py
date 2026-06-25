# -*- coding: utf-8 -*-
"""
PPE Worker - PPE 检测专用 Worker 线程。

封装 PPEDetector，集成到 MultiModelPipeline。
与 AlgoWorker 类似，但使用 PPEDetector 进行两阶段检测。
"""
import logging
import time
from typing import Optional, Any

import numpy as np

from .frame_hub import FrameHub
from .result_store import ResultStore


class PPEWorker:
    """
    PPE 检测专用 Worker 线程。

    与 AlgoWorker 的区别：
    - 使用 PPEDetector 进行两阶段检测
    - 每个流维护独立的 frame_count
    - 结果存储为 PPEResult
    """

    def __init__(
        self,
        algo_id: str,
        ppe_detector: Any,
        frame_hub: FrameHub,
        result_store: ResultStore,
        config: Any = None,
    ):
        """
        初始化 PPE Worker。

        Args:
            algo_id: 算法 ID（通常为 "ppe"）
            ppe_detector: PPEDetector 实例
            frame_hub: FrameHub 实例
            result_store: ResultStore 实例
            config: 配置对象
        """
        self.algo_id = algo_id
        self.ppe_detector = ppe_detector
        self.frame_hub = frame_hub
        self.result_store = result_store
        self.config = config

        # 每个流的帧计数
        self._stream_frame_counts = {}

        # 推理间隔控制（与 AlgoWorker 对齐）
        self._inference_interval = 1
        if config is not None:
            model_intervals = getattr(config, 'model_intervals', {}) or {}
            self._inference_interval = max(1, int(model_intervals.get('ppe', 1)))

        # 线程控制
        import threading
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._wake_event = threading.Event()

        # 统计信息
        self._total_frames = 0
        self._total_inferences = 0
        self._total_inference_time_ms = 0.0

    def start(self, stream_keys: list = None) -> None:
        """启动 Worker 线程。"""
        import threading
        if self._thread is not None and self._thread.is_alive():
            logging.warning(f"[PPEWorker-{self.algo_id}] Worker 已在运行")
            return

        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._worker_loop,
            daemon=True,
            name=f"PPEWorker-{self.algo_id}",
            kwargs={"stream_keys": stream_keys},
        )
        self._thread.start()
        logging.info(f"[PPEWorker-{self.algo_id}] Worker 已启动")

    def stop(self) -> None:
        """停止 Worker 线程。"""
        self._stop_event.set()
        self._wake_event.set()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=5.0)
        logging.info(f"[PPEWorker-{self.algo_id}] Worker 已停止")

    def wake(self) -> None:
        """唤醒 Worker 线程。"""
        self._wake_event.set()

    def is_alive(self) -> bool:
        """检查 Worker 线程是否存活。"""
        return self._thread is not None and self._thread.is_alive()

    def get_stats(self) -> dict:
        """获取统计信息。"""
        avg_time = self._total_inference_time_ms / max(1, self._total_inferences)
        return {
            "algo_id": self.algo_id,
            "type": "ppe",
            "total_frames": self._total_frames,
            "total_inferences": self._total_inferences,
            "avg_inference_time_ms": round(avg_time, 2),
            "is_alive": self.is_alive(),
        }

    def _worker_loop(self, stream_keys: list = None) -> None:
        """Worker 主循环。"""
        logging.info(f"[PPEWorker-{self.algo_id}] Worker 循环开始")

        while not self._stop_event.is_set():
            if stream_keys:
                target_streams = stream_keys
            else:
                target_streams = self.frame_hub.get_stream_keys()

            if not target_streams:
                self._wake_event.wait(0.05)
                self._wake_event.clear()
                continue

            has_work = False
            for stream_key in target_streams:
                if self._stop_event.is_set():
                    break

                # 获取最新帧（带 frame_id）
                frame_with_id = self.frame_hub.get_frame_with_id(stream_key)
                if frame_with_id is None:
                    continue
                frame, hub_frame_id = frame_with_id

                has_work = True
                self._total_frames += 1

                # 每个流独立的帧计数（用于跳帧控制）
                frame_count = self._stream_frame_counts.get(stream_key, 0) + 1
                self._stream_frame_counts[stream_key] = frame_count

                # 推理间隔控制：每 N 帧推理一次
                if frame_count % self._inference_interval != 0:
                    continue

                # 执行 PPE 检测
                start_time = time.time()
                try:
                    result = self.ppe_detector.detect(
                        frame=frame,
                        stream_key=stream_key,
                        frame_count=frame_count,
                    )
                    inference_time_ms = (time.time() - start_time) * 1000

                    # 存储结果（使用 FrameHub 提供的 frame_id，用于 camera.py 找原始帧）
                    self.result_store.store_result(
                        stream_key=stream_key,
                        algo_id=self.algo_id,
                        results=result,
                        frame_id=hub_frame_id,
                        inference_time_ms=inference_time_ms,
                    )

                    self._total_inferences += 1
                    self._total_inference_time_ms += inference_time_ms

                except Exception as e:
                    logging.error(f"[PPEWorker-{self.algo_id}] 推理失败: {e}")

            if not has_work:
                self._wake_event.wait(0.05)
                self._wake_event.clear()

        logging.info(f"[PPEWorker-{self.algo_id}] Worker 循环结束")

    def cleanup(self) -> None:
        """清理资源。"""
        self.stop()
        if self.ppe_detector is not None:
            self.ppe_detector.cleanup()
