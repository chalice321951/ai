# -*- coding: utf-8 -*-
"""
PPE 结果数据类型定义。

定义 PPE 检测的结果数据结构，包括：
- PersonPPEResult: 单个人体的 PPE 检测结果
- PPEResult: 一帧的 PPE 检测结果
"""
from dataclasses import dataclass, field
from typing import List, Tuple, Optional


@dataclass
class PersonPPEResult:
    """
    单个人体的 PPE 检测结果。

    Attributes:
        track_id: 人体跟踪 ID（来自 ByteTrack）
        det_box: 原始检测框 (x1, y1, x2, y2)
        crop_box: 扩展后的裁剪框（用于属性分类）
        person_conf: 人体检测置信度
        helmet_prob: 安全帽概率 [0, 1]
        helmet_state: 安全帽状态 ("yes" / "no" / "unknown")
        vest_prob: 反光衣概率 [0, 1]
        vest_state: 反光衣状态 ("yes" / "no" / "unknown")
    """
    track_id: int
    det_box: Tuple[int, int, int, int]  # (x1, y1, x2, y2)
    crop_box: Tuple[int, int, int, int]  # (x1, y1, x2, y2)
    person_conf: float
    helmet_prob: float
    helmet_state: str  # "yes" / "no" / "unknown"
    vest_prob: float
    vest_state: str  # "yes" / "no" / "unknown"

    def is_compliant(self) -> bool:
        """检查是否合规（安全帽和反光衣都佩戴）"""
        return self.helmet_state == "yes" and self.vest_state == "yes"

    def is_unknown(self) -> bool:
        """检查是否有未知状态（不确定是否合规）"""
        return self.helmet_state == "unknown" or self.vest_state == "unknown"

    def is_helmet_violation(self) -> bool:
        """检查是否未佩戴安全帽"""
        return self.helmet_state == "no"

    def is_vest_violation(self) -> bool:
        """检查是否未穿反光衣"""
        return self.vest_state == "no"

    def is_multi_violation(self) -> bool:
        """检查是否多重违规（同时未戴安全帽和反光衣）"""
        return self.is_helmet_violation() and self.is_vest_violation()

    def get_violation_type(self) -> Optional[str]:
        """
        获取违规类型。

        Returns:
            "ppe_multi" / "ppe_helmet" / "ppe_vest" / None（合规）
        """
        if self.is_multi_violation():
            return "ppe_multi"
        elif self.is_helmet_violation():
            return "ppe_helmet"
        elif self.is_vest_violation():
            return "ppe_vest"
        return None

    def to_overlay(self, algo_id: str = "ppe", color: Tuple[int, int, int] = (0, 0, 255)) -> dict:
        """
        转换为 overlay 字典格式（用于告警和渲染）。

        Args:
            algo_id: 算法 ID
            color: 颜色（BGR）

        Returns:
            overlay 字典
        """
        violation_type = self.get_violation_type()
        if violation_type is None:
            return {}

        # 构造标注文本：违规类型 + 置信度 + ID + H/V 状态
        text = (
            f"{violation_type} {self.person_conf:.2f} ID:{self.track_id} "
            f"H:{self.helmet_state}({self.helmet_prob:.2f}) "
            f"V:{self.vest_state}({self.vest_prob:.2f})"
        )

        return {
            "xyxy": self.det_box,
            "text": text,
            "confidence": self.person_conf,
            "class_name": violation_type,
            "track_id": self.track_id,
            "algo_id": algo_id,
            "color": color,
            "ppe_helmet_state": self.helmet_state,
            "ppe_helmet_prob": self.helmet_prob,
            "ppe_vest_state": self.vest_state,
            "ppe_vest_prob": self.vest_prob,
        }


@dataclass
class PPEResult:
    """
    一帧的 PPE 检测结果。

    Attributes:
        persons: 所有检测到的人体的 PPE 结果列表
        inference_time_ms: 推理耗时（毫秒）
        frame_id: 帧编号
    """
    persons: List[PersonPPEResult] = field(default_factory=list)
    inference_time_ms: float = 0.0
    frame_id: int = 0

    @property
    def total_count(self) -> int:
        """总人数"""
        return len(self.persons)

    @property
    def compliant_count(self) -> int:
        """合规人数"""
        return sum(1 for p in self.persons if p.is_compliant())

    @property
    def violation_count(self) -> int:
        """违规人数（仅统计明确违规，不包括 unknown）"""
        return sum(1 for p in self.persons if not p.is_compliant() and not p.is_unknown())

    @property
    def unknown_count(self) -> int:
        """未知状态人数"""
        return sum(1 for p in self.persons if p.is_unknown())

    @property
    def helmet_violation_count(self) -> int:
        """未戴安全帽人数"""
        return sum(1 for p in self.persons if p.is_helmet_violation())

    @property
    def vest_violation_count(self) -> int:
        """未穿反光衣人数"""
        return sum(1 for p in self.persons if p.is_vest_violation())

    @property
    def multi_violation_count(self) -> int:
        """多重违规人数"""
        return sum(1 for p in self.persons if p.is_multi_violation())

    def get_violation_overlays(self, algo_id: str = "ppe", color: Tuple[int, int, int] = (0, 0, 255)) -> List[dict]:
        """
        获取所有违规人体的 overlay 列表。

        Args:
            algo_id: 算法 ID
            color: 颜色（BGR）

        Returns:
            overlay 字典列表
        """
        overlays = []
        for person in self.persons:
            overlay = person.to_overlay(algo_id=algo_id, color=color)
            if overlay:
                overlays.append(overlay)
        return overlays

    def get_statistics(self) -> dict:
        """
        获取统计信息。

        Returns:
            统计信息字典
        """
        return {
            "total": self.total_count,
            "compliant": self.compliant_count,
            "violation": self.violation_count,
            "helmet_violation": self.helmet_violation_count,
            "vest_violation": self.vest_violation_count,
            "multi_violation": self.multi_violation_count,
            "inference_time_ms": round(self.inference_time_ms, 2),
        }
