# -*- coding: utf-8 -*-
"""
MultiModelPipeline - 多模型并行推理管线。

管理多个 AlgoWorker，实现多模型并行推理。
参考 ai_process_acl/video/realtime_multi_algo_pipeline.py 的 RealtimeMultiAlgoPipeline 实现。
"""
import logging
import time
from typing import Dict, List, Optional, Any

import numpy as np

from .frame_hub import FrameHub
from .result_store import ResultStore, AlgorithmResult
from .algo_worker import AlgoWorker
from .ppe_worker import PPEWorker


class MultiModelPipeline:
    """
    多模型并行推理管线。

    设计原则：
    1. 每个模型一个独立的 AlgoWorker 线程
    2. 所有 Worker 共享同一 FrameHub（最新帧广播）
    3. 推理结果存入共享的 ResultStore（带 TTL）
    4. 支持快照获取所有模型的最新结果
    """

    def __init__(self, config: Any = None):
        """
        初始化多模型管线。

        Args:
            config: 配置对象
        """
        self.config = config
        self._frame_hub = FrameHub()

        # 获取 TTL 配置
        ttl_ms = float(getattr(config, 'max_infer_result_age', 0.5) * 1000) if config else 500.0
        self._result_store = ResultStore(default_ttl_ms=ttl_ms)

        self._workers: Dict[str, AlgoWorker] = {}
        self._model_configs: Dict[str, Dict[str, Any]] = {}
        self._started = False

    def add_model(
        self,
        algo_id: str,
        model: Any,
        conf_threshold: float = 0.5,
        device: str = 'cpu',
        inference_interval: int = 1,
        tracker_config: str = 'bytetrack.yaml',
    ) -> bool:
        """
        添加模型到管线。

        Args:
            algo_id: 模型 ID
            model: YOLO 模型实例
            conf_threshold: 置信度阈值
            device: 推理设备
            inference_interval: 推理间隔（每 N 帧推理一次）
            tracker_config: ByteTrack 配置文件路径

        Returns:
            是否添加成功
        """
        if algo_id in self._workers:
            logging.warning(f"[MultiModelPipeline] 模型 {algo_id} 已存在")
            return False

        # 创建 Worker
        worker = AlgoWorker(
            algo_id=algo_id,
            model=model,
            frame_hub=self._frame_hub,
            result_store=self._result_store,
            config=self.config,
            tracker_config=tracker_config,
        )
        worker.set_conf_threshold(conf_threshold)
        worker.set_device(device)
        worker.set_inference_interval(inference_interval)

        self._workers[algo_id] = worker
        self._model_configs[algo_id] = {
            'conf_threshold': conf_threshold,
            'device': device,
            'inference_interval': inference_interval,
            'tracker_config': tracker_config,
        }

        logging.info(f"[MultiModelPipeline] 添加模型 {algo_id}, interval={inference_interval}, device={device}")

        # 如果管线已启动，自动启动新 Worker
        if self._started:
            worker.start()

        return True

    def add_ppe_model(
        self,
        algo_id: str,
        ppe_detector: Any,
        inference_interval: int = 1,
    ) -> bool:
        """
        添加 PPE 检测器到管线。

        Args:
            algo_id: 算法 ID（建议使用 model_id 如 "3099"）
            ppe_detector: PPEDetector 实例
            inference_interval: 推理间隔（每 N 帧推理一次）

        Returns:
            是否添加成功
        """
        if algo_id in self._workers:
            logging.warning(f"[MultiModelPipeline] 模型 {algo_id} 已存在")
            return False

        # 创建 PPE Worker，传入 inference_interval
        worker = PPEWorker(
            algo_id=algo_id,
            ppe_detector=ppe_detector,
            frame_hub=self._frame_hub,
            result_store=self._result_store,
            config=self.config,
            inference_interval=inference_interval,
        )

        self._workers[algo_id] = worker
        self._model_configs[algo_id] = {
            'type': 'ppe',
            'inference_interval': inference_interval,
        }

        logging.info(f"[MultiModelPipeline] 添加 PPE 模型 {algo_id}, interval={inference_interval}")

        # 如果管线已启动，自动启动新 Worker
        if self._started:
            worker.start()

        return True

    def remove_model(self, algo_id: str) -> bool:
        """
        移除模型。

        Args:
            algo_id: 模型 ID

        Returns:
            是否移除成功
        """
        worker = self._workers.pop(algo_id, None)
        if worker is None:
            return False

        worker.stop()
        self._model_configs.pop(algo_id, None)
        logging.info(f"[MultiModelPipeline] 移除模型 {algo_id}")
        return True

    def start(self, stream_keys: list = None) -> None:
        """
        启动所有 Worker。

        Args:
            stream_keys: 要处理的流标识列表，None 表示处理所有流
        """
        if self._started:
            logging.warning("[MultiModelPipeline] 管线已在运行")
            return

        for algo_id, worker in self._workers.items():
            worker.start(stream_keys=stream_keys)

        self._started = True
        logging.info(f"[MultiModelPipeline] 管线已启动，共 {len(self._workers)} 个模型")

    def stop(self) -> None:
        """停止所有 Worker。"""
        for worker in self._workers.values():
            worker.stop()

        self._started = False
        logging.info("[MultiModelPipeline] 管线已停止")

    def health_check(self, auto_restart: bool = True) -> Dict[str, bool]:
        """
        检查所有 Worker 的健康状态。

        Args:
            auto_restart: 是否自动重启异常的 Worker

        Returns:
            Dict[algo_id, is_alive] 各 Worker 的存活状态
        """
        status = {}
        for algo_id, worker in self._workers.items():
            is_alive = worker.is_alive()
            status[algo_id] = is_alive

            if not is_alive and self._started:
                logging.warning(f"[MultiModelPipeline] Worker {algo_id} 已停止")
                if auto_restart:
                    logging.info(f"[MultiModelPipeline] 尝试重启 Worker {algo_id}")
                    try:
                        worker.start()
                        # 等待一小段时间检查是否启动成功
                        import time
                        time.sleep(0.1)
                        if worker.is_alive():
                            logging.info(f"[MultiModelPipeline] Worker {algo_id} 重启成功")
                            status[algo_id] = True
                        else:
                            logging.error(f"[MultiModelPipeline] Worker {algo_id} 重启失败")
                    except Exception as e:
                        logging.error(f"[MultiModelPipeline] Worker {algo_id} 重启异常: {e}")

        return status

    def submit_frame(self, stream_key: str, frame: np.ndarray, frame_id: int = 0) -> None:
        """
        提交帧到管线（广播到所有 Worker）。

        Args:
            stream_key: 流标识
            frame: 帧数据
            frame_id: 帧编号
        """
        # 存入 FrameHub
        self._frame_hub.set_frame(stream_key, frame, frame_id)

        # 唤醒所有 Worker
        for worker in self._workers.values():
            worker.wake()

    def get_results(self, stream_key: str, algo_ids: List[str] = None, ttl_ms: float = None) -> Dict[str, AlgorithmResult]:
        """
        获取指定流的多模型推理结果（快照）。

        Args:
            stream_key: 流标识
            algo_ids: 模型 ID 列表，None 表示所有模型
            ttl_ms: TTL（毫秒），None 使用默认值

        Returns:
            Dict[algo_id, AlgorithmResult]，只包含未过期的结果
        """
        return self._result_store.snapshot_results(
            stream_key=stream_key,
            algo_ids=algo_ids,
            ttl_ms=ttl_ms,
        )

    def get_merged_results(self, stream_key: str, algo_ids: List[str] = None, ttl_ms: float = None) -> Dict[str, Any]:
        """
        获取指定流的多模型推理结果（合并原始结果）。

        Args:
            stream_key: 流标识
            algo_ids: 模型 ID 列表，None 表示所有模型
            ttl_ms: TTL（毫秒），None 使用默认值

        Returns:
            Dict[algo_id, raw_results]，只包含未过期的结果
        """
        snapshot = self.get_results(stream_key, algo_ids, ttl_ms)
        return {algo_id: result.results for algo_id, result in snapshot.items()}

    def get_model_ids(self) -> List[str]:
        """
        获取所有模型 ID。

        Returns:
            模型 ID 列表
        """
        return list(self._workers.keys())

    def get_worker_stats(self) -> Dict[str, dict]:
        """
        获取所有 Worker 的统计信息。

        Returns:
            Dict[algo_id, stats]
        """
        return {algo_id: worker.get_stats() for algo_id, worker in self._workers.items()}

    def get_pipeline_stats(self) -> dict:
        """
        获取管线统计信息。

        Returns:
            包含模型数量、Worker 状态等
        """
        worker_stats = self.get_worker_stats()
        result_stats = self._result_store.get_stats()

        return {
            "started": self._started,
            "total_models": len(self._workers),
            "model_ids": self.get_model_ids(),
            "worker_stats": worker_stats,
            "result_store_stats": result_stats,
        }

    def is_model_alive(self, algo_id: str) -> bool:
        """
        检查指定模型的 Worker 是否存活。

        Args:
            algo_id: 模型 ID

        Returns:
            是否存活
        """
        worker = self._workers.get(algo_id)
        return worker is not None and worker.is_alive()

    def is_all_alive(self) -> bool:
        """
        检查所有 Worker 是否存活。

        Returns:
            是否全部存活
        """
        if not self._workers:
            return False
        return all(worker.is_alive() for worker in self._workers.values())

    def remove_stream(self, stream_key: str) -> None:
        """
        移除流的所有数据，包括各 Worker 内部的缓存。

        Args:
            stream_key: 流标识
        """
        self._frame_hub.remove_stream(stream_key)
        self._result_store.remove_stream(stream_key)
        # 同步清理各 Worker 的 stream 级缓存（tracker、计数器、属性缓存）
        for worker in self._workers.values():
            if hasattr(worker, 'remove_stream'):
                try:
                    worker.remove_stream(stream_key)
                except Exception as e:
                    logging.debug(f"[MultiModelPipeline] Worker remove_stream 异常: {e}")

    def cleanup(self) -> None:
        """清理所有资源。"""
        self.stop()
        self._frame_hub.clear()
        self._result_store.clear()
        self._workers.clear()
        self._model_configs.clear()
        logging.info("[MultiModelPipeline] 管线已清理")
