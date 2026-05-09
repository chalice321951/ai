# -*- coding: utf-8 -*-

import json
import os
import tempfile
import unittest

from utils.alarm_level import classify_point_alarm_level_uv


class TestAlarmLevel(unittest.TestCase):
    @staticmethod
    def _seg(x, color=None):
        item = {"u1": x, "v1": 20, "u2": x, "v2": 80}
        if color is not None:
            item["color_bgr"] = list(color)
        return item

    def test_single_line_uses_color_level(self):
        self.assertEqual(
            classify_point_alarm_level_uv((10, 50), {"outside_border_-40m_1": [self._seg(40, (0, 0, 255))]}),
            1,
        )
        self.assertEqual(
            classify_point_alarm_level_uv((10, 50), {"outside_border_-20m_1": [self._seg(60, (0, 255, 255))]}),
            2,
        )
        self.assertEqual(
            classify_point_alarm_level_uv((10, 50), {"outside_border_0m_1": [self._seg(80, (38, 167, 255))]}),
            3,
        )

    def test_multi_line_uses_band_level(self):
        projected = {
            "outside_border_-40m_1": [self._seg(40, (0, 0, 255))],
            "outside_border_-20m_1": [self._seg(60, (0, 255, 255))],
            "outside_border_0m_1": [self._seg(80, (38, 167, 255))],
        }
        self.assertEqual(classify_point_alarm_level_uv((50, 50), projected), 2)
        self.assertEqual(classify_point_alarm_level_uv((70, 50), projected), 3)

    def test_outside_orange_returns_none(self):
        projected = {
            "outside_border_-40m_1": [self._seg(40, (0, 0, 255))],
            "outside_border_-20m_1": [self._seg(60, (0, 255, 255))],
            "outside_border_0m_1": [self._seg(80, (38, 167, 255))],
        }
        self.assertIsNone(classify_point_alarm_level_uv((95, 50), projected))

    def test_can_use_colors_from_border_json(self):
        projected = {
            "outside_border_-40m_1": [self._seg(40)],
            "outside_border_-20m_1": [self._seg(60)],
            "outside_border_0m_1": [self._seg(80)],
        }
        data = [
            {"curve_id": "outside_border_-40m_1", "color_bgr": [0, 0, 255], "points": ["0,0,0", "1,0,0"]},
            {"curve_id": "outside_border_-20m_1", "color_bgr": [0, 255, 255], "points": ["0,0,0", "1,0,0"]},
            {"curve_id": "outside_border_0m_1", "color_bgr": [38, 167, 255], "points": ["0,0,0", "1,0,0"]},
        ]
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)
            path = f.name
        try:
            self.assertEqual(classify_point_alarm_level_uv((50, 50), projected, border_json_path=path), 2)
            self.assertIsNone(classify_point_alarm_level_uv((95, 50), projected, border_json_path=path))
        finally:
            try:
                os.unlink(path)
            except OSError:
                pass


if __name__ == "__main__":
    unittest.main()
