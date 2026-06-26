# -*- coding: utf-8 -*-
"""
Parallel Inference Scheduler - 并行多模型推理调度器。

集成 MultiModelPipeline，实现多模型并行推理。
每个模型运行在独立的 AlgoWorker 线程中，互不阻塞。
参考 ai_process_acl/video/realtime_multi_algo_pipeline.py 的架构设计。
"""
import logging
import threading
import time
from typing import Any, Dict, List, Optional

import numpy as np

from .inference_engine import InferenceEngine
from pipeline import MultiModelPipeline, AlgorithmResult


class ParallelInferenceScheduler:
    """
    并行多模型推理调度器。

    设计原则：
    1. 每个模型一个独立的 AlgoWorker 线程
    2. 所有 Worker 共享同一 FrameHub（最新帧广播）
    3. 推理结果存入共享的 ResultStore（带 TTL）
    4. 支持快照获取所有模型的最新结果
    """

    def __init__(self, config):
        """
        初始化并行推理调度器。

        Args:
            config: 配置对象
        """
        self.config = config
        self._lock = threading.Lock()

        # 配置校验
        if hasattr(config, 'validate'):
            warnings = config.validate()
            for w in warnings:
                logging.warning(f"[ParallelScheduler] 配置警告: {w}")

        # 创建推理引擎（用于加载模型）
        self._engine = InferenceEngine(config)

        # 创建多模型管线
        self._pipeline = MultiModelPipeline(config)

        # 从配置加载模型到管线
        self._load_models_to_pipeline()

        # 启动管线
        if self._pipeline.get_model_ids():
            self._pipeline.start()
            logging.info(f"[ParallelScheduler] 并行调度器已启动，模型: {self._pipeline.get_model_ids()}")
        else:
            logging.warning("[ParallelScheduler] 并行调度器未启动：无可用模型")

        # 启动健康检查线程
        self._health_check_interval = 30  # 秒
        self._health_check_stop = threading.Event()
        self._health_check_thread = threading.Thread(
            target=self._health_check_loop,
            daemon=True,
            name="HealthCheck",
        )
        self._health_check_thread.start()

    def _load_models_to_pipeline(self) -> None:
        """从 InferenceEngine 加载模型到管线。"""
        model_ids = self._engine.get_model_ids()
        model_configs = self._engine.get_model_runtime_configs()

        # 获取 tracking 配置
        tracker_config = str(getattr(self.config, 'tracking_tracker', 'bytetrack.yaml') or 'bytetrack.yaml')

        # 获取模型推理间隔配置
        model_intervals = getattr(self.config, 'model_intervals', {}) or {}

        for algo_id in model_ids:
            # 获取模型实例
            model = self._engine._models.get(algo_id)
            if model is None:
                logging.warning(f"[ParallelScheduler] 模型 {algo_id} 不存在，跳过")
                continue

            # 获取模型配置
            model_cfg = model_configs.get(algo_id, {})
            conf_threshold = model_cfg.get('conf_threshold', 0.5)
            device = model_cfg.get('device', 'cpu')

            # 获取推理间隔
            inference_interval = int(model_intervals.get(algo_id, 1))

            # 添加模型到管线
            self._pipeline.add_model(
                algo_id=algo_id,
                model=model,
                conf_threshold=conf_threshold,
                device=device,
                inference_interval=inference_interval,
                tracker_config=tracker_config,
            )

            logging.info(f"[ParallelScheduler] 添加模型 {algo_id}, interval={inference_interval}, device={device}")

        # 如果 PPE 启用，加载 PPE 检测器
        ppe_enabled = bool(getattr(self.config, 'ppe_enabled', False))
        if ppe_enabled:
            self._load_ppe_to_pipeline()

    def _load_ppe_to_pipeline(self) -> None:
        """加载 PPE 检测器到管线。"""
        from inference.ppe import PPEDetector

        ppe_config = getattr(self.config, 'ppe_config', {})
        if not ppe_config:
            logging.warning("[ParallelScheduler] PPE 配置为空，跳过")
            return

        # 获取 PPE 人体检测模型路径
        ppe_detection = ppe_config.get('detection', {})
        ppe_model_id = ppe_detection.get('model_id', '3099')

        # 从 InferenceEngine 获取模型路径
        model_configs = self._engine.get_model_runtime_configs()
        ppe_model_cfg = model_configs.get(ppe_model_id, {})
        ppe_model_path = ppe_model_cfg.get('path', '')

        if not ppe_model_path:
            logging.warning(f"[ParallelScheduler] PPE 模型 {ppe_model_id} 路径为空，跳过")
            return

        # 获取 tracking 配置
        tracker_config = str(getattr(self.config, 'tracking_tracker', 'bytetrack.yaml') or 'bytetrack.yaml')

        # 复用 InferenceEngine 中已加载的模型实例（避免重复加载，省显存）
        shared_model = self._engine._models.get(ppe_model_id)
        if shared_model is not None:
            # 从管线中移除已注册的 AlgoWorker（避免同一模型双重推理）
            if ppe_model_id in self._pipeline.get_model_ids():
                self._pipeline.remove_model(ppe_model_id)
                logging.info(f"[ParallelScheduler] 从管线移除 AlgoWorker[{ppe_model_id}]，由 PPEDetector 接管")

        # 读取该 model_id 对应的推理间隔（如 model_intervals["3099"]=5）
        model_intervals = getattr(self.config, 'model_intervals', {}) or {}
        ppe_interval = int(model_intervals.get(ppe_model_id, 1))

        # 创建 PPE 检测器（复用共享模型，algo_id 用 model_id）
        ppe_detector = PPEDetector(
            config=ppe_config,
            model_path=ppe_model_path,
            device=ppe_model_cfg.get('device', 'cpu'),
            shared_model=shared_model,
            algo_id=ppe_model_id,
        )

        # 将 tracker_config 传入 PPE 配置
        ppe_detector._tracker_config = tracker_config

        # 添加到管线：algo_id 用 model_id（"3099"），不再用 "ppe"
        self._pipeline.add_ppe_model(
            algo_id=ppe_model_id,
            ppe_detector=ppe_detector,
            inference_interval=ppe_interval,
        )

        logging.info(
            f"[ParallelScheduler] 添加 PPE 检测器, algo_id={ppe_model_id}, "
            f"model={ppe_model_path}, interval={ppe_interval}, "
            f"shared={shared_model is not None}"
        )

    def _health_check_loop(self) -> None:
        """健康检查循环，定期检查 Worker 状态并自动重启异常 Worker。"""
        while not self._health_check_stop.is_set():
            try:
                status = self._pipeline.health_check(auto_restart=True)
                dead_workers = [aid for aid, alive in status.items() if not alive]
                if dead_workers:
                    logging.warning(f"[ParallelScheduler] 异常 Worker: {dead_workers}")
            except Exception as e:
                logging.error(f"[ParallelScheduler] 健康检查异常: {e}")

            # 等待下次检查
            self._health_check_stop.wait(self._health_check_interval)

    def health_check(self) -> Dict[str, bool]:
        """手动触发健康检查。"""
        return self._pipeline.health_check(auto_restart=True)

    def is_loaded(self) -> bool:
        """检查是否有可用模型。"""
        return len(self._pipeline.get_model_ids()) > 0

    def submit_frame(self, stream_key: str, frame: np.ndarray, algo_id: str = None, frame_id: int = 0) -> bool:
        """
        提交帧进行推理。

        Args:
            stream_key: 流标识
            frame: 输入帧
            algo_id: 指定的算法 ID（模型 ID），None 表示提交到所有模型
            frame_id: 帧编号

        Returns:
            bool: 是否提交成功
        """
        if not self.is_loaded():
            return False

        # 提交到管线（广播到所有 Worker）
        self._pipeline.submit_frame(stream_key, frame, frame_id)
        return True

    def submit_frame_multi_model(self, stream_key: str, frame: np.ndarray, model_ids: List[str] = None, frame_id: int = 0) -> bool:
        """
        提交帧到多个模型进行推理。

        Args:
            stream_key: 流标识
            frame: 输入帧
            model_ids: 模型 ID 列表，None 表示所有模型
            frame_id: 帧编号

        Returns:
            bool: 是否提交成功
        """
        if not self.is_loaded():
            return False

        # 提交到管线（广播到所有 Worker）
        self._pipeline.submit_frame(stream_key, frame, frame_id)
        return True

    def get_latest_result(self, stream_key: str) -> Optional[Dict[str, Any]]:
        """
        获取指定流的最新推理结果（所有模型合并）。

        Args:
            stream_key: 流标识

        Returns:
            Dict 包含:
                - frame_id: 帧编号
                - results: Dict[model_id, results] 合并结果
                - result_ts: 结果时间戳
            如果没有结果返回 None
        """
        if not self.is_loaded():
            return None

        # 从管线获取结果
        snapshot = self._pipeline.get_results(stream_key)
        if not snapshot:
            return None

        # 合并结果，同时保留每个 algo 自己的 frame_id（修复 M2：避免 PPE 慢推理时画面错位）
        merged_results = {}
        per_algo_frame_ids = {}
        latest_frame_id = 0
        latest_result_ts = 0.0

        for algo_id, result in snapshot.items():
            merged_results[algo_id] = result.results
            per_algo_frame_ids[algo_id] = result.frame_id
            latest_frame_id = max(latest_frame_id, result.frame_id)
            latest_result_ts = max(latest_result_ts, result.timestamp)

        return {
            "frame_id": latest_frame_id,
            "results": merged_results,
            "result_ts": latest_result_ts,
            "per_algo_frame_ids": per_algo_frame_ids,
        }

    def get_latest_result_multi_model(self, stream_key: str, model_ids: List[str] = None) -> Optional[Dict[str, Any]]:
        """
        获取指定流的多模型推理结果。

        Args:
            stream_key: 流标识
            model_ids: 模型 ID 列表，None 表示所有模型

        Returns:
            Dict 包含:
                - frame_id: 帧编号
                - results: Dict[model_id, results] 合并结果
                - result_ts: 结果时间戳
            如果没有结果返回 None
        """
        if not self.is_loaded():
            return None

        # 从管线获取结果
        snapshot = self._pipeline.get_results(stream_key, algo_ids=model_ids)
        if not snapshot:
            return None

        # 合并结果
        merged_results = {}
        latest_frame_id = 0
        latest_result_ts = 0.0

        for algo_id, result in snapshot.items():
            merged_results[algo_id] = result.results
            latest_frame_id = max(latest_frame_id, result.frame_id)
            latest_result_ts = max(latest_result_ts, result.timestamp)

        return {
            "frame_id": latest_frame_id,
            "results": merged_results,
            "result_ts": latest_result_ts,
        }

    def get_model_ids(self) -> List[str]:
        """获取所有可用模型的 ID 列表。"""
        return self._pipeline.get_model_ids()

    def get_model_runtime_configs(self) -> Dict[str, Dict[str, Any]]:
        """获取所有模型的运行时配置。"""
        return self._engine.get_model_runtime_configs()

    def get_engine(self) -> InferenceEngine:
        """获取推理引擎实例。"""
        return self._engine

    def get_pipeline(self) -> MultiModelPipeline:
        """获取多模型管线实例。"""
        return self._pipeline

    def get_pipeline_stats(self) -> dict:
        """获取管线统计信息。"""
        return self._pipeline.get_pipeline_stats()

    def get_worker_stats(self) -> Dict[str, dict]:
        """获取所有 Worker 的统计信息。"""
        return self._pipeline.get_worker_stats()

    def is_model_alive(self, algo_id: str) -> bool:
        """检查指定模型的 Worker 是否存活。"""
        return self._pipeline.is_model_alive(algo_id)

    def is_all_alive(self) -> bool:
        """检查所有 Worker 是否存活。"""
        return self._pipeline.is_all_alive()

    def ensure_stream(self, stream_key: str):
        """确保流已注册（空操作，管线自动处理）。"""
        pass

    def reset_stream_tracking(self, stream_key: str, model_ids: List[str] = None):
        """
        重置流的追踪状态。

        Args:
            stream_key: 流标识
            model_ids: 模型 ID 列表，None 表示重置所有模型的状态
        """
        self._pipeline.remove_stream(stream_key)

    def remove_stream(self, stream_key: str, model_ids: List[str] = None):
        """
        移除流的状态。

        Args:
            stream_key: 流标识
            model_ids: 模型 ID 列表，None 表示移除所有模型的状态
        """
        self._pipeline.remove_stream(stream_key)

    def cleanup(self):
        """清理所有资源。"""
        # 停止健康检查线程
        self._health_check_stop.set()
        if self._health_check_thread.is_alive():
            self._health_check_thread.join(timeout=2)

        self._pipeline.cleanup()
        self._engine.cleanup()
        logging.info("[ParallelScheduler] 并行调度器已清理")
