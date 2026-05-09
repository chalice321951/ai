# -*- coding: utf-8 -*-

import json
import os
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from utils.alarm_level import classify_point_alarm_level_uv

if "cv2" not in sys.modules:
    class _FakeCv2:
        @staticmethod
        def clipLine(rect, p0, p1):
            width = int(rect[2])
            height = int(rect[3])
            x0, y0 = int(p0[0]), int(p0[1])
            x1, y1 = int(p1[0]), int(p1[1])
            x0 = max(0, min(width, x0))
            x1 = max(0, min(width, x1))
            y0 = max(0, min(height, y0))
            y1 = max(0, min(height, y1))
            return True, (x0, y0), (x1, y1)

        @staticmethod
        def line(*args, **kwargs):
            return None

    sys.modules["cv2"] = _FakeCv2()

if "pyproj" not in sys.modules:
    class _FakeTransformer:
        @classmethod
        def from_crs(cls, *args, **kwargs):
            return cls()

        def transform(self, *args):
            return args

    class _FakeCRS:
        @staticmethod
        def from_epsg(code):
            return code

    sys.modules["pyproj"] = type("FakePyProj", (), {"Transformer": _FakeTransformer, "CRS": _FakeCRS})()

from utils.zone_projector import CameraProjector


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

    def test_zone_projector_returns_clipped_visible_segments(self):
        projector = CameraProjector(
            sensor_crop_w_mm=100.0,
            sensor_crop_h_mm=100.0,
            expect_img_w=100,
            expect_img_h=100,
        )
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        border_data = [
            {"curve_id": "outside_border_-40m_1", "color_bgr": [0, 0, 255], "points": ["0,0,0", "1,0,0"]},
        ]
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8") as f:
            json.dump(border_data, f, ensure_ascii=False)
            path = f.name
        try:
            with patch.object(projector, "_focal_physical_mm", return_value=1.0), \
                 patch.object(projector, "_rotation_world_to_camera", return_value=np.eye(3, dtype=np.float32)), \
                 patch.object(projector._to_ecef, "transform", return_value=(
                     np.asarray([-1000.0, 1000.0]),
                     np.asarray([50.0, 50.0]),
                     np.asarray([1.0, 1.0]),
                 )), \
                 patch.object(projector, "_wgs84_to_ecef", return_value=np.zeros((3,), dtype=np.float32)), \
                 patch.object(projector, "_ecef_to_enu_matrix", return_value=np.eye(3, dtype=np.float32)):
                projected, _ = projector.project_points_from_json(
                    path,
                    {
                        "zoom_factor": 1.0,
                        "gimbal_yaw": 0.0,
                        "gimbal_pitch": 0.0,
                        "gimbal_roll": 0.0,
                        "longitude": 0.0,
                        "latitude": 0.0,
                        "height": 1.0,
                    },
                    img,
                    ground_alt=0.0,
                    draw=False,
                    use_cache=False,
                )
            seg = projected["outside_border_-40m_1"][0]
            self.assertGreaterEqual(seg["u1"], 0)
            self.assertLessEqual(seg["u1"], 100)
            self.assertGreaterEqual(seg["u2"], 0)
            self.assertLessEqual(seg["u2"], 100)
        finally:
            try:
                os.unlink(path)
            except OSError:
                pass


if __name__ == "__main__":
    unittest.main()
