# -*- coding: utf-8 -*-
"""
CascadeVerifier - 级联二次校验器。

背景：某些模型（如 3001 大型机械）会把画面中的人/车误分类成目标物体，
造成误报。用另一个专门检测人/车的模型（如 3002）对同一块检测框区域
复核：如果 3002 在这块区域里检出了人/车，说明 3001 的这个框是误检，
直接剔除；3002 检不出，才认为这确实是目标物体，保留该框继续走后续
追踪/告警流程。
"""
import logging
from typing import Any

import numpy as np


class CascadeVerifier:
    """对上游模型的检测框做二次校验，剔除命中校验类别的误检框。"""

    def __init__(
        self,
        verifier_model: Any,
        verifier_class_names: set,
        box_expand_ratio: float = 0.08,
        conf_threshold: float = 0.25,
        device: str = 'cpu',
    ):
        """
        Args:
            verifier_model: 校验模型实例（YOLO），建议使用共享实例，避免重复加载
            verifier_class_names: 命中即判定为误检的类别名集合（如人/车类别）
            box_expand_ratio: 裁剪检测框时的外扩比例，避免边缘裁切过紧导致漏检
            conf_threshold: 校验模型推理时使用的置信度阈值
            device: 校验模型的推理设备
        """
        self._model = verifier_model
        self._class_names = {str(c).strip().lower() for c in (verifier_class_names or set())}
        self._box_expand_ratio = max(0.0, float(box_expand_ratio))
        self._conf_threshold = float(conf_threshold)
        self._device = device

    def _expand_box(self, x1: float, y1: float, x2: float, y2: float, frame_shape: tuple) -> tuple:
        h, w = frame_shape[:2]
        box_w = x2 - x1
        box_h = y2 - y1
        expand_w = int(box_w * self._box_expand_ratio)
        expand_h = int(box_h * self._box_expand_ratio)
        return (
            max(0, int(x1) - expand_w),
            max(0, int(y1) - expand_h),
            min(w, int(x2) + expand_w),
            min(h, int(y2) + expand_h),
        )

    def _hit_verifier_class(self, verify_result: Any) -> bool:
        vboxes = getattr(verify_result, 'boxes', None)
        if vboxes is None or len(vboxes) == 0:
            return False
        names = getattr(verify_result, 'names', {})
        try:
            clss = vboxes.cls.cpu().numpy() if hasattr(vboxes.cls, 'cpu') else np.asarray(vboxes.cls)
        except Exception:
            return False
        for cls_id in clss:
            label = names.get(int(cls_id), '') if isinstance(names, dict) else ''
            if str(label).strip().lower() in self._class_names:
                return True
        return False

    def filter_result(self, frame: np.ndarray, result: Any) -> Any:
        """
        对单帧检测结果做级联校验过滤。

        Args:
            frame: 原始帧（用于裁剪检测框区域）
            result: 单帧 YOLO Results 对象

        Returns:
            过滤后的 Results 对象；校验模型不可用、无检测框，或本次校验
            过程出现异常时，原样返回未经过滤的结果（不影响主流程）。
        """
        if self._model is None or result is None:
            return result
        boxes = getattr(result, 'boxes', None)
        if boxes is None or len(boxes) == 0:
            return result

        try:
            xyxy = boxes.xyxy.cpu().numpy() if hasattr(boxes.xyxy, 'cpu') else np.asarray(boxes.xyxy)
        except Exception as e:
            logging.debug(f"[CascadeVerifier] 读取检测框失败: {e}")
            return result

        crops = []
        valid_indices = []
        for i in range(len(xyxy)):
            x1, y1, x2, y2 = self._expand_box(*xyxy[i], frame_shape=frame.shape)
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                continue
            crops.append(crop)
            valid_indices.append(i)

        if not crops:
            return result

        try:
            verify_results = self._model.predict(
                crops,
                conf=self._conf_threshold,
                device=self._device,
                verbose=False,
            )
        except Exception as e:
            logging.debug(f"[CascadeVerifier] 校验推理失败: {e}")
            return result

        if not isinstance(verify_results, (list, tuple)):
            verify_results = [verify_results]

        reject_indices = {
            idx for idx, vres in zip(valid_indices, verify_results)
            if self._hit_verifier_class(vres)
        }
        if not reject_indices:
            return result

        keep_mask = np.array([i not in reject_indices for i in range(len(xyxy))])
        if keep_mask.all():
            return result
        try:
            filtered = result[keep_mask]
            logging.debug(
                f"[CascadeVerifier] 剔除 {len(reject_indices)}/{len(xyxy)} 个疑似人/车误检框"
            )
            return filtered
        except Exception as e:
            logging.debug(f"[CascadeVerifier] 过滤检测框失败: {e}")
            return result
