# -*- coding: utf-8 -*-
"""
PPE（Personal Protective Equipment）检测模块。

实现安全帽和反光衣的两阶段检测：
1. 第一阶段：人体检测 + ByteTrack 跟踪
2. 第二阶段：属性分类（安全帽/反光衣）

核心组件：
- PPEDetector: 两阶段检测器
- PPEAttrModel: 属性分类模型（MobileNet V3 Small）
- PersonPPEResult: 单个人体的 PPE 检测结果
- PPEResult: 一帧的 PPE 检测结果

使用示例：
    from inference.ppe import PPEDetector

    # 创建检测器
    detector = PPEDetector(config, model_path, device="cuda:0")

    # 执行检测
    result = detector.detect(frame, frame_count=1)

    # 获取违规 overlay
    overlays = detector.get_violation_overlays(result)
"""

from .ppe_result_types import PersonPPEResult, PPEResult
from .ppe_attr_model import PPEAttrModel, load_ppe_attr_model, classify_attributes, prob_to_state
from .ppe_detector import PPEDetector

__all__ = [
    'PersonPPEResult',
    'PPEResult',
    'PPEAttrModel',
    'load_ppe_attr_model',
    'classify_attributes',
    'prob_to_state',
    'PPEDetector',
]
