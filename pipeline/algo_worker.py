# -*- coding: utf-8 -*-
"""
AlgoWorker - 独立的算法 Worker 线程。

每个模型运行在独立的 Worker 线程中，互不阻塞。
从 FrameHub 获取最新帧，推理后将结果存入 ResultStore。

关键设计：
- 模型权重共享（省显存）
- 每个流独立的 ByteTrack tracker（隔离追踪状态）
- 使用 model.track() + 每个流独立保存/恢复模型内部 tracker 状态
"""
import threading
import time
import logging
from typing import Optional, Any, Dict

import numpy as np

from .frame_hub import FrameHub
from .result_store import ResultStore


class AlgoWorker:
    """
    独立的算法 Worker 线程。

    设计原则：
    1. 每个模型一个独立 Worker，互不阻塞
    2. 从 FrameHub 获取最新帧（覆盖式，不堆积）
    3. 推理结果存入 ResultStore（带 TTL）
    4. 支持推理间隔配置（跳帧执行）
    5. 每个流独立的 ByteTrack tracker（隔离追踪状态）
    """

    def __init__(
        self,
        algo_id: str,
        model: Any,
        frame_hub: FrameHub,
        result_store: ResultStore,
        config: Any = None,
        tracker_config: str = "bytetrack.yaml",
        cascade_verifier: Any = None,
    ):
        """
        初始化 AlgoWorker。

        Args:
            algo_id: 算法/模型 ID
            model: YOLO 模型实例
            frame_hub: FrameHub 实例
            result_store: ResultStore 实例
            config: 配置对象
            tracker_config: ByteTrack 配置文件路径
            cascade_verifier: 可选的 CascadeVerifier 实例，用于对本模型的检测框
                做二次校验（如用人车模型复核大型机械模型的误检框）
        """
        self.algo_id = algo_id
        self.model = model
        self.frame_hub = frame_hub
        self.result_store = result_store
        self.config = config
        self._cascade_verifier = cascade_verifier

        # 推理配置
        self._conf_threshold = float(getattr(config, 'default_conf_threshold', 0.5) if config else 0.5)
        self._device = 'cpu'
        self._tracker_config = tracker_config
        # 外部 camera.py 已有 SimpleTracker 负责追踪，AlgoWorker 只做检测（predict），
        # 与旧版 UnifiedInferenceScheduler 行为一致，避免双重追踪和 model.track() 潜在问题。
        self._tracking_enabled = False
        self._tracking_persist = bool(getattr(config, 'tracking_persist', True) if config else True)

        # 每个流独立的 tracker 状态缓存
        self._stream_trackers: Dict[str, Any] = {}

        # 推理间隔控制
        self._inference_interval = 1  # 每 N 帧推理一次
        self._stream_frame_counters: Dict[str, int] = {}

        # 缓存首次 start 传入的 stream_keys，健康检查重启时复用
        self._cached_stream_keys: Optional[list] = None

        # 线程控制
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._wake_event = threading.Event()

        # 统计信息
        self._total_frames = 0
        self._total_inferences = 0
        self._total_inference_time_ms = 0.0

    def set_inference_interval(self, interval: int) -> None:
        self._inference_interval = max(1, int(interval))

    def set_device(self, device: str) -> None:
        self._device = device

    def set_conf_threshold(self, conf: float) -> None:
        self._conf_threshold = float(conf)

    def _save_tracker_state(self, stream_key: str) -> None:
        """保存当前模型内部 tracker 状态到流缓存。"""
        try:
            predictor = getattr(self.model, 'predictor', None)
            if predictor is None:
                return
            trackers = getattr(predictor, 'trackers', None)
            if trackers:
                self._stream_trackers[stream_key] = list(trackers)
        except Exception as e:
            logging.debug(f"[AlgoWorker-{self.algo_id}] 保存 tracker 状态失败: {e}")

    def _restore_tracker_state(self, stream_key: str) -> None:
        """从流缓存恢复 tracker 状态到模型。"""
        try:
            cached = self._stream_trackers.get(stream_key)
            if cached is None:
                # 首次访问该流：删除 predictor.trackers 属性，
                # 让 ultralytics 的 on_predict_start 回调重新创建 trackers。
                # 注意：不能设为 [] —— on_predict_start 检查 hasattr 而非 truthiness，
                # 设为空列表会让回调误以为已初始化，跳过创建导致后续 IndexError。
                predictor = getattr(self.model, 'predictor', None)
                if predictor is not None and hasattr(predictor, 'trackers'):
                    try:
                        delattr(predictor, 'trackers')
                    except AttributeError:
                        pass
                return
            predictor = getattr(self.model, 'predictor', None)
            if predictor is not None:
                predictor.trackers = cached
        except Exception as e:
            logging.debug(f"[AlgoWorker-{self.algo_id}] 恢复 tracker 状态失败: {e}")

    def start(self, stream_keys: list = None) -> None:
        if self._thread is not None and self._thread.is_alive():
            logging.warning(f"[AlgoWorker-{self.algo_id}] Worker 已在运行")
            return

        # 缓存第一次指定的 stream_keys，健康检查重启时复用，避免行为漂移
        if stream_keys is not None:
            self._cached_stream_keys = list(stream_keys)
        actual_stream_keys = getattr(self, '_cached_stream_keys', None) or stream_keys

        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._worker_loop,
            daemon=True,
            name=f"AlgoWorker-{self.algo_id}",
            kwargs={"stream_keys": actual_stream_keys},
        )
        self._thread.start()
        logging.info(f"[AlgoWorker-{self.algo_id}] Worker 已启动")

    def stop(self) -> None:
        self._stop_event.set()
        self._wake_event.set()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=5.0)
        logging.info(f"[AlgoWorker-{self.algo_id}] Worker 已停止")

    def wake(self) -> None:
        self._wake_event.set()

    def is_alive(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def get_stats(self) -> dict:
        avg_time = self._total_inference_time_ms / max(1, self._total_inferences)
        return {
            "algo_id": self.algo_id,
            "total_frames": self._total_frames,
            "total_inferences": self._total_inferences,
            "avg_inference_time_ms": round(avg_time, 2),
            "inference_interval": self._inference_interval,
            "is_alive": self.is_alive(),
            "stream_trackers": len(self._stream_trackers),
        }

    def _worker_loop(self, stream_keys: list = None) -> None:
        logging.info(f"[AlgoWorker-{self.algo_id}] Worker 循环开始（批推理模式）")

        # 尝试引入 torch 用于 CUDA sync（关键：与旧版 InferenceEngine.infer_batch 行为一致）
        try:
            import torch
            _torch = torch
        except Exception:
            _torch = None

        while not self._stop_event.is_set():
            if stream_keys:
                target_streams = stream_keys
            else:
                target_streams = self.frame_hub.get_stream_keys()

            if not target_streams:
                self._wake_event.wait(0.05)
                self._wake_event.clear()
                continue

            # 收集所有流的最新帧到一个批次（复刻旧版 _pick_batch 行为）
            batch_streams = []
            batch_frames = []
            batch_frame_ids = []

            for stream_key in target_streams:
                if self._stop_event.is_set():
                    break

                # take_frame_with_id：取出后清空，避免同一帧被重复推理
                frame_with_id = self.frame_hub.take_frame_with_id(stream_key)
                if frame_with_id is None:
                    continue
                frame, hub_frame_id = frame_with_id

                self._total_frames += 1

                # 每个流独立的帧计数器（interval 跳帧仍按流独立）
                counter = self._stream_frame_counters.get(stream_key, 0) + 1
                self._stream_frame_counters[stream_key] = counter

                if counter % self._inference_interval != 0:
                    continue

                batch_streams.append(stream_key)
                batch_frames.append(frame)
                batch_frame_ids.append(hub_frame_id)

            if not batch_frames:
                self._wake_event.wait(0.05)
                self._wake_event.clear()
                continue

            # 执行批推理（一次 model.predict 处理多流的帧）
            start_time = time.time()
            try:
                res_batch = self.model.predict(
                    batch_frames,
                    conf=self._conf_threshold,
                    device=self._device,
                    verbose=False,
                )

                # 关键：显式等待 GPU 完成，避免异步队列积累导致状态污染
                if _torch is not None and str(self._device).lower().startswith('cuda'):
                    try:
                        _torch.cuda.synchronize()
                    except Exception:
                        pass

                inference_time_ms = (time.time() - start_time) * 1000

                if not isinstance(res_batch, (list, tuple)):
                    res_batch = [res_batch]

                # 分发结果到每个流
                for idx, stream_key in enumerate(batch_streams):
                    if idx >= len(res_batch):
                        continue
                    single_result = res_batch[idx]

                    # 级联二次校验：用校验模型复核本模型的检测框，剔除误检
                    if self._cascade_verifier is not None:
                        single_result = self._cascade_verifier.filter_result(
                            batch_frames[idx], single_result
                        )

                    # 统计检测数
                    det_count = 0
                    try:
                        if single_result is not None and hasattr(single_result, 'boxes') and single_result.boxes is not None:
                            det_count = len(single_result.boxes)
                    except Exception:
                        pass

                    # 用 list 包裹保持与 _extract_raw_detections 的兼容格式
                    self.result_store.store_result(
                        stream_key=stream_key,
                        algo_id=self.algo_id,
                        results=[single_result],
                        frame_id=batch_frame_ids[idx],
                        inference_time_ms=inference_time_ms,
                    )

                    self._total_inferences += 1

                    logging.info(
                        f"[AlgoWorker-{self.algo_id}] stream={stream_key} fid={batch_frame_ids[idx]} "
                        f"detections={det_count} infer_time={inference_time_ms:.0f}ms batch={len(batch_frames)}"
                    )

                self._total_inference_time_ms += inference_time_ms

            except Exception as e:
                logging.error(f"[AlgoWorker-{self.algo_id}] 批量推理失败: {e}")

        logging.info(f"[AlgoWorker-{self.algo_id}] Worker 循环结束")

    def _run_inference(self, frame: np.ndarray, stream_key: str = "default") -> Any:
        """
        执行推理。

        关键设计：使用 model.track() + 每个流独立保存/恢复模型内部 tracker 状态，
        避免 ByteTrack 跨流混淆。

        Args:
            frame: 输入帧
            stream_key: 流标识（用于隔离 tracker 状态）

        Returns:
            推理结果（box.id 包含 track_id）
        """
        if self._tracking_enabled:
            # 1. 恢复该流的 tracker 状态到模型
            self._restore_tracker_state(stream_key)

            # 2. 使用 model.track() 进行追踪
            results = self.model.track(
                frame,
                conf=self._conf_threshold,
                device=self._device,
                persist=self._tracking_persist,
                tracker=self._tracker_config,
                verbose=False,
            )

            # 3. 保存当前 tracker 状态回流缓存
            self._save_tracker_state(stream_key)
        else:
            results = self.model.predict(
                frame,
                conf=self._conf_threshold,
                device=self._device,
                verbose=False,
            )

        return results

    def cleanup(self) -> None:
        """清理资源。"""
        self.stop()
        self._stream_trackers.clear()
        self._stream_frame_counters.clear()

    def remove_stream(self, stream_key: str) -> None:
        """移除指定流的所有缓存状态（tracker、计数器）。"""
        self._stream_trackers.pop(stream_key, None)
        self._stream_frame_counters.pop(stream_key, None)
