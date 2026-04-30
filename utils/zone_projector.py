# -*- coding: utf-8 -*-
"""
地理保护区界线投影模块。

功能：
基于摄像头 WGS84 位置、云台姿态、变焦倍数和保护区界线 JSON，
将地图上的真实界线投影绘制到相机视频帧中。

关键约束：
1. 保留“相邻两点都在画外，但线段穿过画面，则仍然绘制”的能力；
2. 只连接 JSON 中原本相邻的点，不跨越相机后方点或异常投影点重连；
3. 支持边界 JSON 预解析与文件修改感知缓存。
"""

from __future__ import annotations

import json
import math
import os
import threading
from typing import Dict, List, Optional, Tuple, Any

import cv2
import numpy as np
from pyproj import CRS, Transformer

_FULLFRAME_DIAG_MM = 43.266615  # 35mm 全画幅对角线 mm


def _rodrigues(axis: np.ndarray, angle_rad: float) -> np.ndarray:
    axis = np.asarray(axis, dtype=float)
    n = np.linalg.norm(axis)
    if n == 0:
        return np.eye(3)
    axis = axis / n
    x, y, z = axis
    c = math.cos(angle_rad)
    s = math.sin(angle_rad)
    C = 1 - c
    return np.array([
        [c + x * x * C, x * y * C - z * s, x * z * C + y * s],
        [y * x * C + z * s, c + y * y * C, y * z * C - x * s],
        [z * x * C - y * s, z * y * C + x * s, c + z * z * C],
    ], dtype=float)


class CameraProjector:
    def __init__(self,
                 sensor_full_w_mm: float = 7.18,
                 sensor_full_h_mm: float = 5.32,
                 sensor_crop_w_mm: float = 7.18,
                 sensor_crop_h_mm: float = 4.04,
                 expect_img_w: int = 1920,
                 expect_img_h: int = 1080,
                 assume_focal_is_35mm_equiv: Optional[bool] = None):
        self.sensor_full_w = float(sensor_full_w_mm)
        self.sensor_full_h = float(sensor_full_h_mm)
        self.sensor_crop_w = float(sensor_crop_w_mm)
        self.sensor_crop_h = float(sensor_crop_h_mm)
        self.img_w = int(expect_img_w)
        self.img_h = int(expect_img_h)
        self.assume_focal_is_35mm_equiv = assume_focal_is_35mm_equiv
        self.sensor_full_diag = math.hypot(self.sensor_full_w, self.sensor_full_h)

        self._to_ecef = Transformer.from_crs(CRS.from_epsg(4326), CRS.from_epsg(4978), always_xy=True)
        self._from_ecef = Transformer.from_crs(CRS.from_epsg(4978), CRS.from_epsg(4326), always_xy=True)

        self._curves_cache: Dict[str, Dict[str, Any]] = {}
        self._cache_lock = threading.Lock()

    # ------------------------- JSON 缓存 -------------------------
    @staticmethod
    def _preparse_curves(curves: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """把 points 字符串预解析为 numpy 数组，减少每帧重复 split/float 的开销。"""
        for curve in curves:
            if not isinstance(curve, dict):
                continue
            pts = curve.get("points")
            if not isinstance(pts, list):
                continue
            parsed = []
            for p in pts:
                if not isinstance(p, str):
                    continue
                try:
                    lon, lat, alt = map(float, p.split(","))
                    parsed.append((lon, lat, alt))
                except Exception:
                    continue
            if parsed:
                curve["_coords_np"] = np.asarray(parsed, dtype=np.float64)
        return curves

    def _load_curves_cached(self, border_json_path: str) -> List[Dict[str, Any]]:
        path = os.path.abspath(str(border_json_path))
        st = os.stat(path)
        key = path
        sig = (int(st.st_mtime_ns), int(st.st_size))
        with self._cache_lock:
            item = self._curves_cache.get(key)
            if item and item.get("sig") == sig:
                return item["curves"]
        with open(path, "r", encoding="utf-8") as f:
            curves = json.load(f)
        if not isinstance(curves, list):
            raise ValueError(f"边界 JSON 顶层应为 list: {path}")
        curves = self._preparse_curves(curves)
        with self._cache_lock:
            self._curves_cache[key] = {"sig": sig, "curves": curves}
        return curves

    def _load_curves_nocache(self, border_json_path: str) -> List[Dict[str, Any]]:
        with open(border_json_path, "r", encoding="utf-8") as f:
            curves = json.load(f)
        if not isinstance(curves, list):
            raise ValueError(f"边界 JSON 顶层应为 list: {border_json_path}")
        return self._preparse_curves(curves)

    # ------------------------- 坐标变换 -------------------------
    def _wgs84_to_ecef(self, lon: float, lat: float, alt: float) -> np.ndarray:
        x, y, z = self._to_ecef.transform(lon, lat, alt)
        return np.array([x, y, z], dtype=float)

    def _ecef_to_enu_matrix(self, ref_lon: float, ref_lat: float) -> np.ndarray:
        lam = math.radians(ref_lon)
        phi = math.radians(ref_lat)
        return np.array([
            [-math.sin(lam), math.cos(lam), 0.0],
            [-math.sin(phi) * math.cos(lam), -math.sin(phi) * math.sin(lam), math.cos(phi)],
            [math.cos(phi) * math.cos(lam), math.cos(phi) * math.sin(lam), math.sin(phi)],
        ], dtype=float)

    def _wgs84_to_enu(self, lon: float, lat: float, alt: float,
                      ref_lon: float, ref_lat: float, ref_alt: float) -> np.ndarray:
        p = self._wgs84_to_ecef(lon, lat, alt)
        pr = self._wgs84_to_ecef(ref_lon, ref_lat, ref_alt)
        return self._ecef_to_enu_matrix(ref_lon, ref_lat) @ (p - pr)

    # ------------------------- 姿态与焦距 -------------------------
    def _rotation_world_to_camera(self, yaw_deg: float, pitch_deg: float, roll_deg: float,
                                  debug: bool = False) -> np.ndarray:
        basis = np.eye(3)
        yaw_rad = -math.radians(yaw_deg)
        cz = np.array([
            [math.cos(yaw_rad), -math.sin(yaw_rad), 0.0],
            [math.sin(yaw_rad), math.cos(yaw_rad), 0.0],
            [0.0, 0.0, 1.0],
        ], dtype=float)
        basis = cz @ basis

        pitch_rad = math.radians(pitch_deg)
        axis_pitch = basis[:, 0]
        basis = _rodrigues(axis_pitch, pitch_rad) @ basis

        roll_rad = math.radians(roll_deg)
        axis_roll = basis[:, 1]
        basis = _rodrigues(axis_roll, roll_rad) @ basis

        c2w = basis
        w2b = c2w.T
        # 相机坐标系：X 向右，Y 向下，Z 向前。
        m = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 0.0, -1.0],
            [0.0, 1.0, 0.0],
        ], dtype=float)
        return m @ w2b

    def _focal_physical_mm(self, focal_in_srt: float,
                           force_35mm_equiv: Optional[bool] = None,
                           debug: bool = False) -> float:
        if force_35mm_equiv is None:
            use35 = self.assume_focal_is_35mm_equiv
        else:
            use35 = force_35mm_equiv
        if use35 is None:
            use35 = float(focal_in_srt) > 13.0
        if use35:
            return float(focal_in_srt) * (self.sensor_full_diag / _FULLFRAME_DIAG_MM)
        return float(focal_in_srt)

    # ------------------------- 投影绘制 -------------------------
    @staticmethod
    def _is_reasonable_uv(u: float, v: float, max_abs: float) -> bool:
        return math.isfinite(u) and math.isfinite(v) and abs(u) <= max_abs and abs(v) <= max_abs

    @staticmethod
    def _safe_int_point(u: float, v: float, max_abs: int) -> Tuple[int, int]:
        ui = int(round(max(-max_abs, min(max_abs, float(u)))))
        vi = int(round(max(-max_abs, min(max_abs, float(v)))))
        return ui, vi

    def project_points_from_json(self,
                                 border_json_path: str,
                                 info: Dict[str, Any],
                                 img: np.ndarray,
                                 ground_alt: float,
                                 force_35mm_equiv: Optional[bool] = None,
                                 line_color_bgr: Optional[Tuple[int, int, int]] = None,
                                 line_thickness: Optional[int] = None,
                                 use_cache: bool = True,
                                 draw: bool = True) -> Tuple[Dict[str, List[Dict[str, int]]], np.ndarray]:
        """
        将保护区界线投影到图像上。

        只连接 JSON 中原本相邻的轨迹点：
        - 前方点↔前方点：正常投影，允许两点都在画外但线段穿过画面；
        - 后方点↔前方点：把后方端点裁剪到近平面再投影；
        - 后方点↔后方点：跳过；
        - 投影异常：跳过该相邻段，不跨段重连。
        """
        if img is None:
            raise ValueError("输入的 img 为空或无效")

        EPS_DEPTH = 1e-1
        MAX_PIXEL_ABS = 2 ** 27
        SAFE_INT_ABS = 2 ** 30

        img_h, img_w = img.shape[:2]
        self.img_w = int(img_w)
        self.img_h = int(img_h)

        curves = self._load_curves_cached(border_json_path) if use_cache else self._load_curves_nocache(border_json_path)

        # 与前面调试后可用版本保持一致：zoom_factor * 6，并明确按物理焦距处理。
        srt_f = float(info["zoom_factor"]) * 6.0
        f_phys_mm = self._focal_physical_mm(srt_f, False if force_35mm_equiv is None else force_35mm_equiv)
        fx = (f_phys_mm / self.sensor_crop_w) * img_w
        fy = (f_phys_mm / self.sensor_crop_h) * img_h
        cx = img_w / 2.0
        cy = img_h / 2.0

        yaw = float(info["gimbal_yaw"])
        pitch = float(info["gimbal_pitch"])
        roll = float(info["gimbal_roll"])
        ref_lon = float(info["longitude"])
        ref_lat = float(info["latitude"])
        ref_alt = float(info["height"]) - float(ground_alt)

        r_wc = self._rotation_world_to_camera(yaw, pitch, roll)
        pr_ecef = self._wgs84_to_ecef(ref_lon, ref_lat, ref_alt)
        r_enu = self._ecef_to_enu_matrix(ref_lon, ref_lat)
        r_combined = r_wc @ r_enu

        if line_thickness is None:
            thickness = 8
        else:
            thickness = max(1, int(line_thickness))
        default_color = tuple(int(x) for x in (line_color_bgr or (0, 200, 255)))

        results: Dict[str, List[Dict[str, int]]] = {}

        for curve in curves:
            if not isinstance(curve, dict):
                continue
            pts = curve.get("points")
            if not isinstance(pts, list):
                continue
            coords = curve.get("_coords_np")
            if coords is None or not isinstance(coords, np.ndarray) or coords.shape[0] < 2:
                continue

            cid = str(curve.get("curve_id", "curve"))
            color = default_color
            c = curve.get("color_bgr")
            if isinstance(c, (list, tuple)) and len(c) == 3:
                try:
                    color = tuple(int(x) for x in c)
                except Exception:
                    color = default_color

            lons = coords[:, 0]
            lats = coords[:, 1]
            alts = coords[:, 2]
            xs, ys, zs = self._to_ecef.transform(lons, lats, alts)
            p_ecef = np.column_stack((xs, ys, zs))
            cam_all = (r_combined @ (p_ecef - pr_ecef).T).T

            curve_segments: List[Dict[str, int]] = []
            for i in range(coords.shape[0] - 1):
                pa = cam_all[i].astype(np.float64, copy=True)
                pb = cam_all[i + 1].astype(np.float64, copy=True)
                za = float(pa[2])
                zb = float(pb[2])

                # 两端都在近平面后：整段不可投影，不跨过去重连。
                if za < EPS_DEPTH and zb < EPS_DEPTH:
                    continue

                # 一前一后：裁到近平面。
                if za < EPS_DEPTH <= zb:
                    denom = zb - za
                    if abs(denom) < 1e-12:
                        continue
                    t = (EPS_DEPTH - za) / denom
                    pa = pa + t * (pb - pa)
                elif zb < EPS_DEPTH <= za:
                    denom = za - zb
                    if abs(denom) < 1e-12:
                        continue
                    t = (EPS_DEPTH - zb) / denom
                    pb = pb + t * (pa - pb)

                if pa[2] < EPS_DEPTH or pb[2] < EPS_DEPTH:
                    continue

                u1 = fx * (pa[0] / pa[2]) + cx
                v1 = fy * (pa[1] / pa[2]) + cy
                u2 = fx * (pb[0] / pb[2]) + cx
                v2 = fy * (pb[1] / pb[2]) + cy

                if not self._is_reasonable_uv(u1, v1, MAX_PIXEL_ABS):
                    continue
                if not self._is_reasonable_uv(u2, v2, MAX_PIXEL_ABS):
                    continue

                p0 = self._safe_int_point(u1, v1, SAFE_INT_ABS)
                p1 = self._safe_int_point(u2, v2, SAFE_INT_ABS)

                # clipLine 负责判断：即使两端都在画外，只要线段穿过画面，就保留。
                ok, q0, q1 = cv2.clipLine((0, 0, int(img_w), int(img_h)), p0, p1)
                if not ok:
                    continue

                q0 = (int(q0[0]), int(q0[1]))
                q1 = (int(q1[0]), int(q1[1]))
                if draw:
                    cv2.line(img, q0, q1, color, thickness)
                curve_segments.append({"u1": q0[0], "v1": q0[1], "u2": q1[0], "v2": q1[1]})

            results[cid] = curve_segments

        return results, img

    # ------------------------- 像素反算地面坐标 -------------------------
    def pixel_to_wgs84(self,
                       u: float,
                       v: float,
                       info: Dict[str, Any],
                       ground_alt: float = 0.0) -> Tuple[float, float]:
        srt_f = float(info["zoom_factor"]) * 6.0
        f_phys_mm = self._focal_physical_mm(srt_f, False)
        fx = (f_phys_mm / self.sensor_crop_w) * self.img_w
        fy = (f_phys_mm / self.sensor_crop_h) * self.img_h
        cx = self.img_w / 2.0
        cy = self.img_h / 2.0

        x = (float(u) - cx) / fx
        y = (float(v) - cy) / fy
        ray_cam = np.array([x, y, 1.0], dtype=float)
        ray_cam /= np.linalg.norm(ray_cam)

        yaw = float(info["gimbal_yaw"])
        pitch = float(info["gimbal_pitch"])
        roll = float(info["gimbal_roll"])
        r_wc = self._rotation_world_to_camera(yaw, pitch, roll)
        ray_enu = r_wc.T @ ray_cam
        ray_enu /= np.linalg.norm(ray_enu)

        ref_lon = float(info["longitude"])
        ref_lat = float(info["latitude"])
        ref_alt = float(info["height"])
        cam_enu = np.array([0.0, 0.0, ref_alt], dtype=float)

        dz = float(ray_enu[2])
        if abs(dz) < 1e-8:
            raise ValueError("射线几乎平行于地面，无法求交点")
        t = (float(ground_alt) - cam_enu[2]) / dz
        if t < 0:
            raise ValueError("像素射线与地面无交点（方向向上）")
        p_enu = cam_enu + t * ray_enu

        r = self._ecef_to_enu_matrix(ref_lon, ref_lat)
        p_ecef = self._wgs84_to_ecef(ref_lon, ref_lat, ref_alt) + (r.T @ p_enu)
        lon, lat, _alt = self._from_ecef.transform(float(p_ecef[0]), float(p_ecef[1]), float(p_ecef[2]))
        return float(lon), float(lat)
