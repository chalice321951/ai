# -*- coding: utf-8 -*-

import unittest
from types import SimpleNamespace

from nan.camera_server import PlatformApiClient


class _DummyAlertLevel:
    value = "medium"


class _DummyAlertEvent:
    def __init__(self, metadata):
        self.metadata = metadata
        self.alert_level = _DummyAlertLevel()
        self.timestamp = 1.0
        self.event_id = "evt-1"
        self.rule_id = "alarm_any_detection"


class TestPlatformAlarmLevel(unittest.TestCase):
    def test_payload_prefers_dynamic_alarm_level(self):
        config = SimpleNamespace(platform_device_type=1, platform_vendor_id=2, platform_task_type=3)
        client = PlatformApiClient(config=config)
        event = _DummyAlertEvent(metadata={"target_info": {"alarm_level": "3", "class_name": "person"}})
        payload = client._build_alarm_payload(
            stream_cfg={"name": "cam-1", "alarmAccuratePosition": "0,0"},
            alert_event=event,
            image_url="a.jpg",
            video_url="b.mp4",
        )
        self.assertEqual(payload["alarmLevel"], 3)


if __name__ == "__main__":
    unittest.main()
