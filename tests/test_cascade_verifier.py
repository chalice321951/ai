# -*- coding: utf-8 -*-
"""
CascadeVerifier 单元测试。

模拟 ultralytics Results 的最小接口（boxes.xyxy/conf/cls + names + __getitem__），
不依赖真实的 ultralytics / cv2 安装。
"""
import sys
import os
import logging

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

from pipeline.cascade_verifier import CascadeVerifier


class FakeBoxes:
    def __init__(self, xyxy, conf, cls):
        self.xyxy = np.asarray(xyxy, dtype=float)
        self.conf = np.asarray(conf, dtype=float)
        self.cls = np.asarray(cls, dtype=float)

    def __len__(self):
        return len(self.xyxy)


class FakeResult:
    """模拟 ultralytics Results：支持 boolean mask 索引。"""

    def __init__(self, xyxy, conf, cls, names):
        self.boxes = FakeBoxes(xyxy, conf, cls) if len(xyxy) > 0 else None
        self.names = names

    def __getitem__(self, keep_mask):
        keep_mask = np.asarray(keep_mask, dtype=bool)
        idx = np.where(keep_mask)[0]
        new_xyxy = self.boxes.xyxy[idx]
        new_conf = self.boxes.conf[idx]
        new_cls = self.boxes.cls[idx]
        return FakeResult(new_xyxy, new_conf, new_cls, self.names)


class FakeVerifierModel:
    """模拟 3002 人车模型：按裁剪区域索引返回预设检测结果。"""

    def __init__(self, hit_indices: set):
        # hit_indices: 第几个 crop（按调用顺序）应命中人/车类别
        self._hit_indices = hit_indices
        self.call_count = 0

    def predict(self, crops, conf=0.25, device='cpu', verbose=False):
        results = []
        for i, _crop in enumerate(crops):
            if self.call_count + i in self._hit_indices:
                results.append(FakeResult(
                    xyxy=[[0, 0, 10, 10]], conf=[0.9], cls=[0.0],
                    names={0: 'person'},
                ))
            else:
                results.append(FakeResult(xyxy=[], conf=[], cls=[], names={0: 'person'}))
        self.call_count += len(crops)
        return results


def test_filters_out_misclassified_person():
    """3001 检出 2 个框：第 0 个实际是人（应被剔除），第 1 个是真实机械（应保留）。"""
    logger.info("测试: 剔除被误分类为机械的人")

    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    result = FakeResult(
        xyxy=[[10, 10, 50, 50], [100, 100, 300, 300]],
        conf=[0.8, 0.9],
        cls=[0.0, 0.0],
        names={0: 'excavator'},
    )

    verifier_model = FakeVerifierModel(hit_indices={0})  # 第 0 个裁剪区域命中"人"
    verifier = CascadeVerifier(
        verifier_model=verifier_model,
        verifier_class_names={'person', 'car'},
        box_expand_ratio=0.08,
    )

    filtered = verifier.filter_result(frame, result)

    assert len(filtered.boxes) == 1, f"应剩余 1 个检测框，实际 {len(filtered.boxes)}"
    assert list(filtered.boxes.xyxy[0]) == [100, 100, 300, 300], "剩余的框应是真实机械框"
    logger.info("✓ 通过")
    return True


def test_keeps_all_when_no_hit():
    """3002 在所有裁剪区域都没检出人/车时，应保留全部原始检测框。"""
    logger.info("测试: 无命中时保留全部检测框")

    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    result = FakeResult(
        xyxy=[[10, 10, 50, 50], [100, 100, 300, 300]],
        conf=[0.8, 0.9],
        cls=[0.0, 0.0],
        names={0: 'excavator'},
    )

    verifier_model = FakeVerifierModel(hit_indices=set())
    verifier = CascadeVerifier(
        verifier_model=verifier_model,
        verifier_class_names={'person', 'car'},
        box_expand_ratio=0.08,
    )

    filtered = verifier.filter_result(frame, result)

    assert filtered is result, "无命中时应原样返回原始结果对象"
    assert len(filtered.boxes) == 2, f"应保留 2 个检测框，实际 {len(filtered.boxes)}"
    logger.info("✓ 通过")
    return True


def test_no_verifier_model_passthrough():
    """校验模型未加载（None）时应直接透传，不影响主流程。"""
    logger.info("测试: 校验模型为 None 时透传")

    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    result = FakeResult(
        xyxy=[[10, 10, 50, 50]], conf=[0.8], cls=[0.0], names={0: 'excavator'},
    )
    verifier = CascadeVerifier(
        verifier_model=None,
        verifier_class_names={'person'},
    )
    filtered = verifier.filter_result(frame, result)
    assert filtered is result, "校验模型为 None 时应原样返回结果"
    logger.info("✓ 通过")
    return True


def test_empty_detections_passthrough():
    """本身没有检测框时应直接透传。"""
    logger.info("测试: 无检测框时透传")

    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    result = FakeResult(xyxy=[], conf=[], cls=[], names={0: 'excavator'})
    verifier_model = FakeVerifierModel(hit_indices=set())
    verifier = CascadeVerifier(
        verifier_model=verifier_model,
        verifier_class_names={'person'},
    )
    filtered = verifier.filter_result(frame, result)
    assert filtered is result, "无检测框时应原样返回结果"
    logger.info("✓ 通过")
    return True


def main():
    tests = [
        test_filters_out_misclassified_person,
        test_keeps_all_when_no_hit,
        test_no_verifier_model_passthrough,
        test_empty_detections_passthrough,
    ]

    passed = 0
    failed = 0
    for test in tests:
        try:
            if test():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            logger.error(f"测试失败: {e}")
            failed += 1

    logger.info(f"测试结果: {passed} 通过, {failed} 失败")
    return failed == 0


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
