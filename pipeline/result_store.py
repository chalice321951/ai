# -*- coding: utf-8 -*-
"""
ResultStore - 带 TTL 的多模型结果缓存。

每个算法的结果独立存储，带 TTL 过期机制。
支持快照获取所有算法的最新结果。
参考 ai_process_acl/video/realtime_fusion.py 的 ResultStore 实现。
"""
import threading
import time
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field


@dataclass
class AlgorithmResult:
    """单个算法的推理结果"""
    algo_id: str
    results: Any  # YOLO 推理结果
    frame_id: int = 0
    timestamp: float = 0.0
    inference_time_ms: float = 0.0


class ResultStore:
    """
    多模型结果缓存。

    设计原则：
    1. 每个算法独立存储，互不干扰
    2. 结果带 TTL，过期后自动失效
    3. 支持快照获取：一次性获取所有算法的最新结果
    4. 线程安全：支持多个 AlgoWorker 并发写入
    """

    def __init__(self, default_ttl_ms: float = 500.0):
        """
        初始化 ResultStore。

        Args:
            default_ttl_ms: 默认 TTL（毫秒），超过此时间的结果视为过期
        """
        self._lock = threading.Lock()
        self._results: Dict[str, Dict[str, AlgorithmResult]] = {}  # stream_key -> {algo_id -> result}
        self._default_ttl_ms = float(default_ttl_ms)

    def store_result(
        self,
        stream_key: str,
        algo_id: str,
        results: Any,
        frame_id: int = 0,
        inference_time_ms: float = 0.0
    ) -> None:
        """
        存储算法的推理结果。

        Args:
            stream_key: 流标识
            algo_id: 算法/模型 ID
            results: 推理结果
            frame_id: 帧编号
            inference_time_ms: 推理耗时（毫秒）
        """
        if not stream_key or not algo_id:
            return

        result = AlgorithmResult(
            algo_id=algo_id,
            results=results,
            frame_id=frame_id,
            timestamp=time.time(),
            inference_time_ms=inference_time_ms,
        )

        with self._lock:
            if stream_key not in self._results:
                self._results[stream_key] = {}
            self._results[stream_key][algo_id] = result

    def get_result(self, stream_key: str, algo_id: str, ttl_ms: float = None) -> Optional[AlgorithmResult]:
        """
        获取指定算法的最新结果。

        Args:
            stream_key: 流标识
            algo_id: 算法/模型 ID
            ttl_ms: TTL（毫秒），None 使用默认值

        Returns:
            算法结果，如果不存在或已过期返回 None
        """
        ttl = ttl_ms if ttl_ms is not None else self._default_ttl_ms

        with self._lock:
            stream_results = self._results.get(stream_key)
            if not stream_results:
                return None

            result = stream_results.get(algo_id)
            if result is None:
                return None

            # 检查是否过期
            age_ms = (time.time() - result.timestamp) * 1000
            if age_ms > ttl:
                return None

            return result

    def snapshot_results(self, stream_key: str, algo_ids: List[str] = None, ttl_ms: float = None) -> Dict[str, AlgorithmResult]:
        """
        快照获取指定流的所有（或指定）算法的最新结果。

        Args:
            stream_key: 流标识
            algo_ids: 算法 ID 列表，None 表示所有算法
            ttl_ms: TTL（毫秒），None 使用默认值

        Returns:
            Dict[algo_id, AlgorithmResult]，只包含未过期的结果
        """
        ttl = ttl_ms if ttl_ms is not None else self._default_ttl_ms
        snapshot = {}

        with self._lock:
            stream_results = self._results.get(stream_key)
            if not stream_results:
                return snapshot

            target_algos = algo_ids if algo_ids else list(stream_results.keys())
            now = time.time()

            for algo_id in target_algos:
                result = stream_results.get(algo_id)
                if result is None:
                    continue

                # 检查是否过期
                age_ms = (now - result.timestamp) * 1000
                if age_ms <= ttl:
                    snapshot[algo_id] = result

        return snapshot

    def get_all_results(self, stream_key: str) -> Dict[str, AlgorithmResult]:
        """
        获取指定流的所有结果（不过滤过期）。

        Args:
            stream_key: 流标识

        Returns:
            Dict[algo_id, AlgorithmResult]
        """
        with self._lock:
            return dict(self._results.get(stream_key, {}))

    def remove_stream(self, stream_key: str) -> None:
        """
        移除流的所有结果。

        Args:
            stream_key: 流标识
        """
        with self._lock:
            self._results.pop(stream_key, None)

    def remove_algo(self, stream_key: str, algo_id: str) -> None:
        """
        移除指定流的指定算法结果。

        Args:
            stream_key: 流标识
            algo_id: 算法/模型 ID
        """
        with self._lock:
            stream_results = self._results.get(stream_key)
            if stream_results:
                stream_results.pop(algo_id, None)

    def clear(self) -> None:
        """清空所有结果。"""
        with self._lock:
            self._results.clear()

    def get_stats(self) -> Dict[str, Any]:
        """
        获取统计信息。

        Returns:
            包含流数量、算法数量等统计信息
        """
        with self._lock:
            total_streams = len(self._results)
            total_results = sum(len(v) for v in self._results.values())
            return {
                "total_streams": total_streams,
                "total_results": total_results,
                "default_ttl_ms": self._default_ttl_ms,
            }
