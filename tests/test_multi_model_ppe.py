# -*- coding: utf-8 -*-
"""
多模型并行推理与 PPE 检测的端到端测试脚本。

测试内容：
1. 配置加载
2. 并行调度器初始化
3. FrameHub 和 ResultStore
4. PPE 检测器（模拟）
5. 告警系统组合键去重
"""
import sys
import os
import time
import logging
import numpy as np

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def test_config_loading():
    """测试配置加载"""
    logger.info("=" * 50)
    logger.info("测试 1: 配置加载")
    logger.info("=" * 50)

    from config.algorithm_config import CameraConfig

    config = CameraConfig()

    # 验证基本配置
    assert hasattr(config, 'tracking_enabled'), "缺少 tracking_enabled"
    assert hasattr(config, 'model_class_filters'), "缺少 model_class_filters"
    assert hasattr(config, 'model_intervals'), "缺少 model_intervals"
    assert hasattr(config, 'use_spatial_level'), "缺少 use_spatial_level"
    assert hasattr(config, 'fixed_alarm_level'), "缺少 fixed_alarm_level"
    assert hasattr(config, 'ppe_enabled'), "缺少 ppe_enabled"
    assert hasattr(config, 'ppe_config'), "缺少 ppe_config"

    logger.info(f"  tracking_enabled: {config.tracking_enabled}")
    logger.info(f"  model_class_filters: {config.model_class_filters}")
    logger.info(f"  model_intervals: {config.model_intervals}")
    logger.info(f"  use_spatial_level: {config.use_spatial_level}")
    logger.info(f"  fixed_alarm_level: {config.fixed_alarm_level}")
    logger.info(f"  ppe_enabled: {config.ppe_enabled}")

    logger.info("✓ 配置加载测试通过")
    return True


def test_frame_hub():
    """测试 FrameHub"""
    logger.info("=" * 50)
    logger.info("测试 2: FrameHub")
    logger.info("=" * 50)

    from pipeline.frame_hub import FrameHub

    hub = FrameHub()

    # 创建测试帧
    frame1 = np.zeros((100, 100, 3), dtype=np.uint8)
    frame2 = np.ones((100, 100, 3), dtype=np.uint8) * 255

    # 设置帧
    hub.set_frame("stream1", frame1, frame_id=1)
    hub.set_frame("stream2", frame2, frame_id=2)

    # 获取帧
    got_frame1 = hub.get_frame("stream1")
    got_frame2 = hub.get_frame("stream2")

    assert got_frame1 is not None, "stream1 帧不存在"
    assert got_frame2 is not None, "stream2 帧不存在"
    assert np.array_equal(got_frame1, frame1), "stream1 帧数据不匹配"
    assert np.array_equal(got_frame2, frame2), "stream2 帧数据不匹配"

    # 测试覆盖
    frame3 = np.ones((100, 100, 3), dtype=np.uint8) * 128
    hub.set_frame("stream1", frame3, frame_id=3)
    got_frame3 = hub.get_frame("stream1")
    assert np.array_equal(got_frame3, frame3), "stream1 帧覆盖失败"

    # 测试流列表
    keys = hub.get_stream_keys()
    assert "stream1" in keys, "stream1 不在流列表中"
    assert "stream2" in keys, "stream2 不在流列表中"

    logger.info("✓ FrameHub 测试通过")
    return True


def test_result_store():
    """测试 ResultStore"""
    logger.info("=" * 50)
    logger.info("测试 3: ResultStore")
    logger.info("=" * 50)

    from pipeline.result_store import ResultStore

    store = ResultStore(default_ttl_ms=500)

    # 存储结果
    store.store_result("stream1", "model1", {"boxes": [1, 2, 3]}, frame_id=1, inference_time_ms=10)
    store.store_result("stream1", "model2", {"boxes": [4, 5, 6]}, frame_id=1, inference_time_ms=15)

    # 获取结果
    result1 = store.get_result("stream1", "model1")
    result2 = store.get_result("stream1", "model2")

    assert result1 is not None, "model1 结果不存在"
    assert result2 is not None, "model2 结果不存在"
    assert result1.results == {"boxes": [1, 2, 3]}, "model1 结果数据不匹配"
    assert result2.results == {"boxes": [4, 5, 6]}, "model2 结果数据不匹配"

    # 快照获取
    snapshot = store.snapshot_results("stream1")
    assert len(snapshot) == 2, f"快照应包含 2 个结果，实际 {len(snapshot)}"

    # 测试 TTL 过期
    time.sleep(0.6)  # 等待超过 TTL
    expired_result = store.get_result("stream1", "model1")
    assert expired_result is None, "结果应该已过期"

    logger.info("✓ ResultStore 测试通过")
    return True


def test_alert_dedup():
    """测试告警组合键去重"""
    logger.info("=" * 50)
    logger.info("测试 4: 告警组合键去重")
    logger.info("=" * 50)

    # 检查 cv2 是否可用
    try:
        import cv2
    except ImportError:
        logger.warning("  cv2 未安装，跳过此测试")
        return True

    from alert.alert_system import AlertSystem, create_count_threshold_rule, AlertLevel

    # 创建告警系统
    class MockConfig:
        def get_alarm_config(self):
            return {}

    config = MockConfig()
    system = AlertSystem(config)

    # 添加规则
    rule = create_count_threshold_rule(
        rule_id="test_rule",
        threshold=1,
        cooldown=0,
        level=AlertLevel.MEDIUM,
    )
    system.add_rule(rule)

    # 模拟告警处理器
    class MockHandler:
        def __init__(self):
            self.alerts = []

        def handle_alert(self, event, frame, target_info, frame_ts=None, raw_frame=None):
            self.alerts.append(event)

    handler = MockHandler()
    system.alert_handler = handler

    # 测试组合键去重
    frame = np.zeros((100, 100, 3), dtype=np.uint8)

    # 第一次告警：模型A track_id=1
    target_info_1 = {
        'tracking_enabled': True,
        'track_ids': [1],
        'algo_id': '3001',
        'class_name': 'excavator',
    }
    system.process_frame_alerts(frame, {'test_rule': 1.0}, target_info=target_info_1)
    assert len(handler.alerts) == 1, f"应触发 1 次告警，实际 {len(handler.alerts)}"

    # 第二次告警：模型B track_id=1（不同 algo_id，应触发）
    target_info_2 = {
        'tracking_enabled': True,
        'track_ids': [1],
        'algo_id': 'ppe',
        'class_name': 'ppe_helmet',
    }
    system.process_frame_alerts(frame, {'test_rule': 1.0}, target_info=target_info_2)
    assert len(handler.alerts) == 2, f"应触发 2 次告警，实际 {len(handler.alerts)}"

    # 第三次告警：模型A track_id=1（相同组合键，不应触发）
    system.process_frame_alerts(frame, {'test_rule': 1.0}, target_info=target_info_1)
    assert len(handler.alerts) == 2, f"应仍为 2 次告警，实际 {len(handler.alerts)}"

    # 第四次告警：模型A track_id=2（不同 track_id，应触发）
    target_info_3 = {
        'tracking_enabled': True,
        'track_ids': [2],
        'algo_id': '3001',
        'class_name': 'excavator',
    }
    system.process_frame_alerts(frame, {'test_rule': 1.0}, target_info=target_info_3)
    assert len(handler.alerts) == 3, f"应触发 3 次告警，实际 {len(handler.alerts)}"

    logger.info("✓ 告警组合键去重测试通过")
    return True


def test_ppe_result_types():
    """测试 PPE 结果类型"""
    logger.info("=" * 50)
    logger.info("测试 5: PPE 结果类型")
    logger.info("=" * 50)

    from inference.ppe import PersonPPEResult, PPEResult

    # 创建测试数据
    person1 = PersonPPEResult(
        track_id=1,
        det_box=(100, 100, 200, 300),
        crop_box=(90, 90, 210, 310),
        person_conf=0.95,
        helmet_prob=0.1,
        helmet_state="no",
        vest_prob=0.8,
        vest_state="yes",
    )

    person2 = PersonPPEResult(
        track_id=2,
        det_box=(300, 100, 400, 300),
        crop_box=(290, 90, 410, 310),
        person_conf=0.90,
        helmet_prob=0.9,
        helmet_state="yes",
        vest_prob=0.9,
        vest_state="yes",
    )

    # 验证违规类型
    assert person1.is_helmet_violation() == True, "person1 应为 helmet 违规"
    assert person1.is_vest_violation() == False, "person1 不应为 vest 违规"
    assert person1.is_multi_violation() == False, "person1 不应为多重违规"
    assert person1.get_violation_type() == "ppe_helmet", f"违规类型应为 ppe_helmet，实际 {person1.get_violation_type()}"

    assert person2.is_compliant() == True, "person2 应为合规"
    assert person2.get_violation_type() is None, "合规人员违规类型应为 None"

    # 创建 PPEResult
    result = PPEResult(persons=[person1, person2], inference_time_ms=15.5, frame_id=1)

    # 验证统计
    assert result.total_count == 2, f"总人数应为 2，实际 {result.total_count}"
    assert result.compliant_count == 1, f"合规人数应为 1，实际 {result.compliant_count}"
    assert result.violation_count == 1, f"违规人数应为 1，实际 {result.violation_count}"
    assert result.helmet_violation_count == 1, f"helmet 违规人数应为 1，实际 {result.helmet_violation_count}"

    # 验证 overlay 转换
    overlays = result.get_violation_overlays(algo_id="ppe")
    assert len(overlays) == 1, f"应有 1 个 overlay，实际 {len(overlays)}"
    assert overlays[0]['class_name'] == 'ppe_helmet', f"overlay 类名应为 ppe_helmet，实际 {overlays[0]['class_name']}"
    assert overlays[0]['algo_id'] == 'ppe', f"overlay algo_id 应为 ppe，实际 {overlays[0]['algo_id']}"

    logger.info("✓ PPE 结果类型测试通过")
    return True


def test_ppe_renderer():
    """测试 PPE 渲染器"""
    logger.info("=" * 50)
    logger.info("测试 6: PPE 渲染器")
    logger.info("=" * 50)

    # 检查 cv2 是否可用
    try:
        import cv2
        cv2_available = True
    except ImportError:
        cv2_available = False
        logger.warning("  cv2 未安装，跳过渲染测试")

    from inference.ppe import PersonPPEResult, PPEResult
    from renderer.ppe_renderer import PPERenderer

    # 创建测试数据
    person = PersonPPEResult(
        track_id=1,
        det_box=(100, 100, 200, 300),
        crop_box=(90, 90, 210, 310),
        person_conf=0.95,
        helmet_prob=0.1,
        helmet_state="no",
        vest_prob=0.8,
        vest_state="yes",
    )

    result = PPEResult(persons=[person], inference_time_ms=15.5, frame_id=1)

    # 创建渲染器
    renderer = PPERenderer({
        'compliant_color': [0, 255, 0],
        'violation_color': [0, 0, 255],
        'unknown_color': [128, 128, 128],
    })

    if cv2_available:
        # 渲染
        frame = np.zeros((400, 400, 3), dtype=np.uint8)
        output = renderer.render(frame, result, show_statistics=True)

        assert output.shape == frame.shape, "输出帧形状应与输入相同"
        assert not np.array_equal(output, frame), "输出帧应与输入不同（已渲染）"

    logger.info("✓ PPE 渲染器测试通过")
    return True


def main():
    """运行所有测试"""
    logger.info("=" * 60)
    logger.info("多模型并行推理与 PPE 检测 - 端到端测试")
    logger.info("=" * 60)

    tests = [
        test_config_loading,
        test_frame_hub,
        test_result_store,
        test_alert_dedup,
        test_ppe_result_types,
        test_ppe_renderer,
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

    logger.info("=" * 60)
    logger.info(f"测试结果: {passed} 通过, {failed} 失败")
    logger.info("=" * 60)

    return failed == 0


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
