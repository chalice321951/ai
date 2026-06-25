# -*- coding: utf-8 -*-
"""
多模型并行推理管线模块。

参考 ai_process_acl/video/realtime_multi_algo_pipeline.py 的架构设计。

核心组件：
- FrameHub: 最新帧广播器（只保留最新帧，不堆积）
- ResultStore: 带 TTL 的结果缓存
- AlgoWorker: 独立的算法 Worker 线程（每个模型一个）
- MultiModelPipeline: 多模型管线协调器

使用示例：
    from pipeline import MultiModelPipeline

    # 创建管线
    pipeline = MultiModelPipeline(config)

    # 添加模型
    pipeline.add_model("3001", model1, conf_threshold=0.8, device="cuda:0")
    pipeline.add_model("3099", model2, conf_threshold=0.5, device="cuda:0")

    # 启动管线
    pipeline.start()

    # 提交帧
    pipeline.submit_frame("stream_1", frame, frame_id=1)

    # 获取结果
    results = pipeline.get_results("stream_1")
"""

from .frame_hub import FrameHub
from .result_store import ResultStore, AlgorithmResult
from .algo_worker import AlgoWorker
from .ppe_worker import PPEWorker
from .multi_model_pipeline import MultiModelPipeline

__all__ = [
    'FrameHub',
    'ResultStore',
    'AlgorithmResult',
    'AlgoWorker',
    'PPEWorker',
    'MultiModelPipeline',
]
