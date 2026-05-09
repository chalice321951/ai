import sys
from types import SimpleNamespace

import numpy as np

sys.modules.setdefault("cv2", SimpleNamespace())

from alert.alert_system import AlertSystem, create_count_threshold_rule


class _DummyHandler:
    def __init__(self):
        self.calls = []

    def handle_alert(self, alert_event, frame, target_info, frame_ts=None):
        self.calls.append({
            'event_id': alert_event.event_id,
            'rule_id': alert_event.rule_id,
            'target_info': target_info,
            'frame_ts': frame_ts,
        })


def test_same_track_id_alerts_only_once():
    alert_system = AlertSystem()
    alert_system.alert_handler = _DummyHandler()
    alert_system.add_rule(create_count_threshold_rule(
        rule_id="alarm_any_detection",
        threshold=1,
        cooldown=0,
    ))

    frame = np.zeros((8, 8, 3), dtype=np.uint8)
    detection_dict = {"alarm_any_detection": 1.0}
    same_target = {
        'tracking_enabled': True,
        'track_ids': [5],
    }

    alert_system.process_frame_alerts(frame, detection_dict, target_info=same_target)
    alert_system.process_frame_alerts(frame, detection_dict, target_info=same_target)
    alert_system.process_frame_alerts(frame, detection_dict, target_info=same_target)

    assert len(alert_system.alert_handler.calls) == 1


def test_new_track_id_can_alert_again():
    alert_system = AlertSystem()
    alert_system.alert_handler = _DummyHandler()
    alert_system.add_rule(create_count_threshold_rule(
        rule_id="alarm_any_detection",
        threshold=1,
        cooldown=0,
    ))

    frame = np.zeros((8, 8, 3), dtype=np.uint8)
    detection_dict = {"alarm_any_detection": 1.0}

    alert_system.process_frame_alerts(frame, detection_dict, target_info={
        'tracking_enabled': True,
        'track_ids': [5],
    })
    alert_system.process_frame_alerts(frame, detection_dict, target_info={
        'tracking_enabled': True,
        'track_ids': [6],
    })

    assert len(alert_system.alert_handler.calls) == 2
