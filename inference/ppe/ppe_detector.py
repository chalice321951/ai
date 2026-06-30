# -*- coding: utf-8 -*-
"""
PPE 两阶段检测器。

第一阶段：人体检测 + ByteTrack 跟踪
第二阶段：属性分类（安全帽/反光衣）

参考 ai_process_acl/ppe/ppe_detector.py 的实现。
"""
import logging
import time
from typing import List, Optional, Any, Tuple, Dict

import numpy as np

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

from .ppe_result_types import PersonPPEResult, PPEResult
from .ppe_attr_model import (
    load_ppe_attr_model,
    classify_attributes,
    prob_to_state,
)


class PPEDetector:
    """
    PPE 两阶段检测器。

    设计原则：
    1. 第一阶段：使用 YOLO 检测人体 + ByteTrack 跟踪
    2. 第二阶段：使用 MobileNet V3 进行属性分类
    3. 属性分类间隔执行，复用缓存结果（基于 track_id）
    4. 每个 PPEDetector 实例独立，不与其他模型共享 tracker
    """

    def __init__(self, config: dict, model_path: str = '', device: str = 'cpu',
                 shared_model: Any = None, algo_id: str = None):
        """
        初始化 PPE 检测器。

        Args:
            config: PPE 配置字典
            model_path: 人体检测模型路径（当 shared_model 为 None 时使用）
            device: 推理设备
            shared_model: 已加载的共享模型实例（避免重复加载，省显存）
            algo_id: 算法 ID（建议使用 model_id 如 "3099"）
        """
        self._config = config
        self._device = device
        # algo_id：统一用 model_id（如 "3099"），不再用 "ppe" 字符串
        self._algo_id = algo_id or config.get('detection', {}).get('model_id', '3099')
        # 生命周期标志（cleanup 后置 False，防止 detect 静默失效）
        self._alive = True

        # 检测配置（对齐参考项目的默认值）
        self._detection_config = config.get('detection', {})
        self._person_class_names = self._detection_config.get('person_class_names', ['person'])
        self._box_expand_ratio = self._detection_config.get('box_expand_ratio', 0.05)
        self._top_extra_ratio = self._detection_config.get('top_extra_ratio', 0.05)

        # 置信度阈值：由 parallel_scheduler 从 config.get_conf_threshold(algo_id) 读取后传入
        # 不再从 ppe.detection.person_conf_threshold 读取（已移除该配置项）
        self._person_conf_threshold = float(config.get('person_conf_threshold', 0.25))

        # 属性分类配置
        self._attribute_config = config.get('attribute', {})
        self._attr_model_path = self._attribute_config.get('model_path', '')
        self._image_size = self._attribute_config.get('image_size', 160)
        self._inference_interval = self._attribute_config.get('inference_interval', 3)
        self._helmet_pos_threshold = self._attribute_config.get('helmet_pos_threshold', 0.6)
        self._helmet_neg_threshold = self._attribute_config.get('helmet_neg_threshold', 0.3)
        self._vest_pos_threshold = self._attribute_config.get('vest_pos_threshold', 0.6)
        self._vest_neg_threshold = self._attribute_config.get('vest_neg_threshold', 0.3)

        # 渲染配置
        self._rendering_config = config.get('rendering', {})

        # 告警配置
        self._alarm_config = config.get('alarm', {})

        # 加载人体检测模型（优先使用共享模型，避免重复加载省显存）
        if shared_model is not None:
            self._person_model = shared_model
            logging.info(f"[PPEDetector] 使用共享的人体检测模型实例（节省显存）")
        else:
            self._person_model = self._load_person_model(model_path)

        # 加载属性分类模型
        self._attr_model = load_ppe_attr_model(self._attr_model_path, device)

        # 缓存：{stream_key: {track_id: (helmet_prob, vest_prob, frame_count)}}
        self._attr_cache: Dict[str, dict] = {}
        self._frame_counter = 0

        # 每个流独立的 ByteTrack tracker
        self._stream_trackers: Dict[str, Any] = {}

        # ByteTrack 配置 - 从全局配置读取，或使用默认值
        self._tracker_config = config.get('tracker_config', 'bytetrack.yaml')

        logging.info(f"[PPEDetector] 初始化完成, device={device}, interval={self._inference_interval}")

    def _load_person_model(self, model_path: str) -> Any:
        """
        加载人体检测模型。

        Args:
            model_path: 模型路径

        Returns:
            YOLO 模型实例
        """
        try:
            from ultralytics import YOLO
            model = YOLO(model_path)
            logging.info(f"[PPEDetector] 人体检测模型加载成功: {model_path}")
            return model
        except Exception as e:
            logging.error(f"[PPEDetector] 人体检测模型加载失败: {e}")
            return None

    def _save_tracker_state(self, stream_key: str) -> None:
        """保存当前模型内部 tracker 状态到流缓存。"""
        try:
            predictor = getattr(self._person_model, 'predictor', None)
            if predictor is None:
                return
            trackers = getattr(predictor, 'trackers', None)
            if trackers:
                self._stream_trackers[stream_key] = list(trackers)
        except Exception as e:
            logging.debug(f"[PPEDetector] 保存 tracker 状态失败: {e}")

    def _restore_tracker_state(self, stream_key: str) -> None:
        """从流缓存恢复 tracker 状态到模型。"""
        try:
            cached = self._stream_trackers.get(stream_key)
            if cached is None:
                # 首次访问该流：删除 predictor.trackers 属性，
                # 让 ultralytics 的 on_predict_start 回调重新创建 trackers。
                # 注意：不能设为 [] —— on_predict_start 检查 hasattr 而非 truthiness。
                predictor = getattr(self._person_model, 'predictor', None)
                if predictor is not None and hasattr(predictor, 'trackers'):
                    try:
                        delattr(predictor, 'trackers')
                    except AttributeError:
                        pass
                return
            predictor = getattr(self._person_model, 'predictor', None)
            if predictor is not None:
                predictor.trackers = cached
        except Exception as e:
            logging.debug(f"[PPEDetector] 恢复 tracker 状态失败: {e}")

    def detect(self, frame: np.ndarray, stream_key: str = "default", frame_count: int = 0) -> PPEResult:
        """
        执行 PPE 检测。

        关键改动：使用 model.track() + 每个流独立保存/恢复模型内部 tracker 状态，
        避免 ByteTrack 跨流混淆。

        Args:
            frame: 输入帧 (H, W, 3) BGR
            stream_key: 流标识（用于 tracker 和缓存隔离）
            frame_count: 帧计数

        Returns:
            PPEResult 检测结果
        """
        if not self._alive:
            raise RuntimeError("PPEDetector 已 cleanup，不能再使用")

        start_time = time.time()

        # 捕获到局部变量，防止 cleanup 中途把 self._person_model / self._attr_model 置为 None
        # 导致 PPEWorker 线程在使用过程中抛 AttributeError。
        person_model = self._person_model
        attr_model = self._attr_model
        if person_model is None:
            return PPEResult(inference_time_ms=0, frame_id=frame_count)

        # 第一阶段：人体检测 + ByteTrack（每个流独立 tracker 状态）
        # 1. 恢复该流的 tracker 状态
        self._restore_tracker_state(stream_key)

        # 2. 使用 model.track() 进行追踪
        person_results = person_model.track(
            frame,
            conf=self._person_conf_threshold,
            persist=True,
            tracker=self._tracker_config,
            classes=[0],  # COCO person class
            verbose=False,
        )

        # 3. 保存 tracker 状态回流缓存
        self._save_tracker_state(stream_key)

        persons = []
        if person_results and len(person_results) > 0:
            result = person_results[0]
            if result.boxes is not None:
                # 优先从 _track_ids 读取（_apply_tracker_to_result 写入的）
                track_ids_tensor = getattr(result.boxes, '_track_ids', None)
                for i, box in enumerate(result.boxes):
                    # 提取跟踪 ID：优先用 _track_ids，回退到 box.id
                    if track_ids_tensor is not None and i < len(track_ids_tensor):
                        track_id = int(track_ids_tensor[i].item())
                    elif box.id is not None:
                        track_id = int(box.id[0])
                    else:
                        track_id = -1

                    # 提取检测框
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    conf = float(box.conf[0])

                    # 扩展裁剪框
                    crop_box = self._expand_box(x1, y1, x2, y2, frame.shape)

                    # 第二阶段：属性分类
                    helmet_prob, vest_prob = self._classify_with_cache(
                        frame, crop_box, track_id, stream_key, frame_count
                    )

                    helmet_state = prob_to_state(
                        helmet_prob, self._helmet_pos_threshold, self._helmet_neg_threshold
                    )
                    vest_state = prob_to_state(
                        vest_prob, self._vest_pos_threshold, self._vest_neg_threshold
                    )

                    person_result = PersonPPEResult(
                        track_id=track_id,
                        det_box=(x1, y1, x2, y2),
                        crop_box=crop_box,
                        person_conf=conf,
                        helmet_prob=helmet_prob,
                        helmet_state=helmet_state,
                        vest_prob=vest_prob,
                        vest_state=vest_state,
                    )
                    persons.append(person_result)

        # 清理过期缓存
        self._cleanup_cache(stream_key, frame_count)

        inference_time_ms = (time.time() - start_time) * 1000

        return PPEResult(
            persons=persons,
            inference_time_ms=inference_time_ms,
            frame_id=frame_count,
        )

    def _expand_box(
        self, x1: int, y1: int, x2: int, y2: int, frame_shape: tuple
    ) -> Tuple[int, int, int, int]:
        """
        扩展检测框以包含头部和上身（对齐参考项目 expand_box 逻辑）。

        参考项目有独立的 top_extra_ratio，专门多扩展头顶区域，
        因为安全帽在头顶，如果扩展不够会裁剪掉。

        Args:
            x1, y1, x2, y2: 原始检测框
            frame_shape: 帧形状 (H, W, C)

        Returns:
            扩展后的检测框 (x1, y1, x2, y2)
        """
        h, w = frame_shape[:2]
        box_h = y2 - y1
        box_w = x2 - x1

        expand_h = int(box_h * self._box_expand_ratio)
        expand_w = int(box_w * self._box_expand_ratio)
        top_extra = int(box_h * self._top_extra_ratio)

        new_x1 = max(0, x1 - expand_w)
        new_y1 = max(0, y1 - expand_h - top_extra)  # 顶部多扩展 top_extra
        new_x2 = min(w, x2 + expand_w)
        new_y2 = min(h, y2 + expand_h)

        return (new_x1, new_y1, new_x2, new_y2)

    def _classify_with_cache(
        self,
        frame: np.ndarray,
        crop_box: Tuple[int, int, int, int],
        track_id: int,
        stream_key: str,
        frame_count: int,
    ) -> Tuple[float, float]:
        """
        带缓存的属性分类。

        如果 track_id 在缓存中且未过期，返回缓存结果。
        否则执行属性分类并缓存结果。

        Args:
            frame: 输入帧
            crop_box: 裁剪框
            track_id: 跟踪 ID
            stream_key: 流标识
            frame_count: 帧计数

        Returns:
            (helmet_prob, vest_prob)
        """
        # track_id < 0 表示未被跟踪（ByteTrack 还没确认），跳过缓存：
        # 多个未跟踪的人会共用 track_id=-1，缓存会互相串扰。
        if track_id is None or (isinstance(track_id, int) and track_id < 0):
            # 不缓存，每次都做属性分类
            x1, y1, x2, y2 = crop_box
            crop = frame[y1:y2, x1:x2]
            attr_model = self._attr_model  # 局部变量防 cleanup 竞争
            if attr_model is not None and crop.size > 0:
                return classify_attributes(
                    attr_model, crop, self._device, self._image_size
                )
            return 0.5, 0.5

        # 初始化流缓存
        if stream_key not in self._attr_cache:
            self._attr_cache[stream_key] = {}

        stream_cache = self._attr_cache[stream_key]

        # 检查缓存
        if track_id in stream_cache:
            cached = stream_cache[track_id]
            cached_frame_count = cached[2]
            if frame_count - cached_frame_count < self._inference_interval:
                return cached[0], cached[1]

        # 执行属性分类
        x1, y1, x2, y2 = crop_box
        crop = frame[y1:y2, x1:x2]

        # 捕获到局部变量，防止 cleanup 中途置空导致 AttributeError
        attr_model = self._attr_model
        if attr_model is not None and crop.size > 0:
            helmet_prob, vest_prob = classify_attributes(
                attr_model, crop, self._device, self._image_size
            )
        else:
            helmet_prob, vest_prob = 0.5, 0.5

        # 更新缓存
        stream_cache[track_id] = (helmet_prob, vest_prob, frame_count)

        return helmet_prob, vest_prob

    def _cleanup_cache(self, stream_key: str, frame_count: int, max_age: int = 100) -> None:
        """
        清理过期缓存。

        Args:
            stream_key: 流标识
            frame_count: 当前帧计数
            max_age: 最大缓存年龄（帧数）
        """
        if stream_key not in self._attr_cache:
            return

        stream_cache = self._attr_cache.get(stream_key)
        if stream_cache is None:
            return

        # 用 list() 拷贝快照后再迭代，避免迭代过程中 remove_stream 等并发操作
        # 修改 dict 大小导致 "dictionary changed size during iteration" 错误。
        expired_keys = [
            k for k, v in list(stream_cache.items())
            if frame_count - v[2] > max_age
        ]
        for k in expired_keys:
            stream_cache.pop(k, None)

    def get_violation_overlays(
        self,
        ppe_result: PPEResult,
        algo_id: str = None,
    ) -> List[dict]:
        """
        获取违规人体的 overlay 列表。

        Args:
            ppe_result: PPE 检测结果
            algo_id: 算法 ID（默认使用 self._algo_id，即 model_id）

        Returns:
            overlay 字典列表
        """
        violation_color = tuple(self._rendering_config.get('violation_color', [0, 0, 255]))
        return ppe_result.get_violation_overlays(
            algo_id=algo_id or self._algo_id,
            color=violation_color,
        )

    def cleanup(self) -> None:
        """清理资源。共享模型不置空，由 InferenceEngine 统一管理。"""
        self._alive = False
        self._attr_cache.clear()
        self._stream_trackers.clear()
        # 只清理 PPE 专用的属性分类模型，共享的 person_model 不置空
        self._attr_model = None
        logging.info("[PPEDetector] 已清理（缓存、tracker、属性模型）")

    def remove_stream(self, stream_key: str) -> None:
        """移除指定流的所有缓存状态。"""
        self._attr_cache.pop(stream_key, None)
        self._stream_trackers.pop(stream_key, None)
