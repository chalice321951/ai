# -*- coding: utf-8 -*-
"""
PPE 渲染器。

负责将 PPE 检测结果渲染到帧上，包括：
- 违规人体的检测框（红色=违规，绿色=合规）
- 安全帽/反光衣状态标注
- 统计信息

参考 ai_process_acl/ppe/ppe_renderer.py 的实现。
"""
import logging
from typing import List, Tuple, Optional

import numpy as np

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False


class PPERenderer:
    """
    PPE 渲染器。

    负责将 PPE 检测结果渲染到帧上。
    """

    def __init__(self, config: dict = None):
        """
        初始化 PPE 渲染器。

        Args:
            config: 渲染配置字典
        """
        self._config = config or {}
        self._compliant_color = tuple(self._config.get('compliant_color', [0, 255, 0]))  # 绿色
        self._violation_color = tuple(self._config.get('violation_color', [0, 0, 255]))  # 红色
        self._unknown_color = tuple(self._config.get('unknown_color', [128, 128, 128]))  # 灰色
        self._font_scale = self._config.get('font_scale', 0.6)
        self._line_thickness = self._config.get('line_thickness', 2)

    def render(
        self,
        frame: np.ndarray,
        ppe_result,
        show_statistics: bool = True,
    ) -> np.ndarray:
        """
        渲染 PPE 检测结果到帧上。

        Args:
            frame: 输入帧
            ppe_result: PPE 检测结果 (PPEResult)
            show_statistics: 是否显示统计信息

        Returns:
            渲染后的帧
        """
        if not CV2_AVAILABLE:
            return frame

        output = frame.copy()

        # 渲染每个人体
        for person in ppe_result.persons:
            self._draw_person(output, person)

        # 渲染统计信息
        if show_statistics:
            self._draw_statistics(output, ppe_result)

        return output

    def _draw_person(self, frame: np.ndarray, person) -> None:
        """
        绘制单个人体的 PPE 信息。

        Args:
            frame: 帧
            person: PersonPPEResult
        """
        x1, y1, x2, y2 = person.det_box

        # 根据状态选择颜色
        if person.is_compliant():
            color = self._compliant_color
        elif person.is_multi_violation():
            color = self._violation_color
        elif person.is_helmet_violation() or person.is_vest_violation():
            color = self._violation_color
        else:
            color = self._unknown_color

        # 绘制检测框
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, self._line_thickness)

        # 绘制标签
        label_lines = [
            f"ID:{person.track_id} {person.person_conf:.2f}",
            f"H:{person.helmet_state}({person.helmet_prob:.2f})",
            f"V:{person.vest_state}({person.vest_prob:.2f})",
        ]

        # 计算标签位置
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = self._font_scale
        thickness = 1

        # 计算最大文本宽度
        max_width = 0
        for line in label_lines:
            (text_width, text_height), _ = cv2.getTextSize(line, font, font_scale, thickness)
            max_width = max(max_width, text_width)

        # 绘制标签背景
        label_height = int(text_height * 1.5)
        total_height = label_height * len(label_lines)
        cv2.rectangle(
            frame,
            (x1, y1 - total_height - 5),
            (x1 + max_width + 10, y1),
            color,
            -1,  # 填充
        )

        # 绘制标签文本
        for i, line in enumerate(label_lines):
            text_y = y1 - total_height + (i + 1) * label_height
            cv2.putText(
                frame,
                line,
                (x1 + 5, text_y),
                font,
                font_scale,
                (255, 255, 255),  # 白色文本
                thickness,
            )

    def _draw_statistics(self, frame: np.ndarray, ppe_result) -> None:
        """
        绘制统计信息。

        Args:
            frame: 帧
            ppe_result: PPE 检测结果
        """
        stats = ppe_result.get_statistics()

        # 统计信息文本
        lines = [
            f"Total: {stats['total']}",
            f"Compliant: {stats['compliant']}",
            f"Violation: {stats['violation']}",
            f"Helmet: {stats['helmet_violation']}",
            f"Vest: {stats['vest_violation']}",
        ]

        # 计算位置（右上角）
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        thickness = 1

        # 计算最大文本宽度
        max_width = 0
        for line in lines:
            (text_width, text_height), _ = cv2.getTextSize(line, font, font_scale, thickness)
            max_width = max(max_width, text_width)

        # 绘制背景
        x = frame.shape[1] - max_width - 20
        y = 10
        line_height = int(text_height * 1.5)
        total_height = line_height * len(lines)
        cv2.rectangle(
            frame,
            (x, y),
            (x + max_width + 10, y + total_height + 10),
            (0, 0, 0),  # 黑色背景
            -1,
        )

        # 绘制文本
        for i, line in enumerate(lines):
            text_y = y + (i + 1) * line_height
            cv2.putText(
                frame,
                line,
                (x + 5, text_y),
                font,
                font_scale,
                (255, 255, 255),  # 白色文本
                thickness,
            )


    def render_alarm_level(
        self,
        frame: np.ndarray,
        person,
        alarm_level: str,
    ) -> None:
        """
        在检测框上标注报警等级。

        Args:
            frame: 帧
            person: PersonPPEResult
            alarm_level: 报警等级 ("1" / "2" / "3")
        """
        if not CV2_AVAILABLE or alarm_level is None:
            return

        x1, y1, x2, y2 = person.det_box

        # 在检测框左上角绘制等级
        label = f"L{alarm_level}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        thickness = 2

        # 绘制背景
        (text_width, text_height), _ = cv2.getTextSize(label, font, font_scale, thickness)
        cv2.rectangle(
            frame,
            (x1, y1 - text_height - 10),
            (x1 + text_width + 10, y1),
            (0, 0, 255),  # 红色背景
            -1,
        )

        # 绘制文本
        cv2.putText(
            frame,
            label,
            (x1 + 5, y1 - 5),
            font,
            font_scale,
            (255, 255, 255),  # 白色文本
            thickness,
        )
