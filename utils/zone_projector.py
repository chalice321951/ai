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
    def _is_finite_uv(u: float, v: float) -> bool:
        """判断投影像素是否为有限数。"""
        return math.isfinite(u) and math.isfinite(v)

    @staticmethod
    def _clip_segment_to_rect_float(u1: float, v1: float,
                                    u2: float, v2: float,
                                    img_w: int, img_h: int,
                                    margin_px: float = 0.0,
                                    eps: float = 1e-12) -> Optional[Tuple[float, float, float, float, float, float]]:
        """使用浮点 Liang-Barsky 算法裁剪线段到图像矩形内。

        为什么不用 cv2.clipLine 直接裁剪？
        - cv2.clipLine 需要整型坐标；当投影点靠近近平面或在相机侧后方时，u/v 可能非常大，
          转成 int 后容易触发 OpenCV 内部整数范围/裁剪精度问题；
        - 如果先把超大点按“图像中心 -> 点”的方向压缩，虽然能避免超限，但会改变“当前线段
          端点A -> 端点B”的真实直线方向，导致局部本应入屏的线段被误判为不入屏。

        本函数直接在原始浮点投影坐标上做线段-矩形求交：
        - 保留真实二维投影线段方向；
        - 输出点必定位于图像范围附近，之后再转 int 绘制，不会出现超大整数；
        - 仍然支持“两端都在画外，但连线穿过画面”的情况；
        - 同时返回 t0/t1，表示裁剪后端点在原二维线段上的参数位置。
        """
        if img_w <= 0 or img_h <= 0:
            return None
        if not (math.isfinite(u1) and math.isfinite(v1) and math.isfinite(u2) and math.isfinite(v2)):
            return None

        # margin_px 用于实现“距图片边缘一定范围内不画线”：
        # 线段先裁剪到 [margin, w-1-margin] × [margin, h-1-margin] 的内部矩形。
        margin = max(0.0, float(margin_px))
        max_margin = max(0.0, min(float(img_w), float(img_h)) * 0.5 - 1.0)
        if margin > max_margin:
            margin = max_margin

        xmin = margin
        ymin = margin
        xmax = float(img_w - 1) - margin
        ymax = float(img_h - 1) - margin

        dx = float(u2) - float(u1)
        dy = float(v2) - float(v1)

        # 退化为一个点。
        if abs(dx) <= eps and abs(dy) <= eps:
            if xmin <= u1 <= xmax and ymin <= v1 <= ymax:
                return float(u1), float(v1), float(u1), float(v1), 0.0, 0.0
            return None

        t0 = 0.0
        t1 = 1.0

        # 约束依次为：x>=xmin, x<=xmax, y>=ymin, y<=ymax
        checks = (
            (-dx, float(u1) - xmin),
            ( dx, xmax - float(u1)),
            (-dy, float(v1) - ymin),
            ( dy, ymax - float(v1)),
        )

        for p, q in checks:
            if abs(p) <= eps:
                # 线段与该边界平行；如果起点已经在该半平面外，则整段不可见。
                if q < 0.0:
                    return None
                continue

            r = q / p
            if p < 0.0:
                if r > t1:
                    return None
                if r > t0:
                    t0 = r
            else:
                if r < t0:
                    return None
                if r < t1:
                    t1 = r

        cu1 = float(u1) + t0 * dx
        cv1 = float(v1) + t0 * dy
        cu2 = float(u1) + t1 * dx
        cv2 = float(v1) + t1 * dy

        return cu1, cv1, cu2, cv2, t0, t1

    @staticmethod
    def _safe_int_point(u: float, v: float,
                        img_w: Optional[int] = None,
                        img_h: Optional[int] = None) -> Tuple[int, int]:
        """把浮点像素坐标转为 OpenCV 绘图用整数坐标。

        这里的输入通常已经经过 _clip_segment_to_rect_float 裁剪，理论上处于图像范围内。
        仍然保留一次 clamp，避免浮点舍入后出现 -1 或 img_w/img_h 这类边界值。
        """
        ui = int(round(float(u)))
        vi = int(round(float(v)))
        if img_w is not None:
            ui = max(0, min(int(img_w) - 1, ui))
        if img_h is not None:
            vi = max(0, min(int(img_h) - 1, vi))
        return ui, vi


    @staticmethod
    def _round_half_up(x: float) -> int:
        """传统四舍五入，避免 Python round(2.5)=2 的银行家舍入。"""
        return int(math.floor(float(x) + 0.5))

    @staticmethod
    def _projective_param_from_uv(pa: np.ndarray,
                                  pb: np.ndarray,
                                  u: float,
                                  v: float,
                                  fx: float,
                                  fy: float,
                                  cx: float,
                                  cy: float,
                                  fallback_t: float = 0.0,
                                  eps: float = 1e-12) -> float:
        """由投影后的像素坐标反求其在三维线段 pa->pb 上的大致参数 lambda。

        为什么不用 pixel_to_wgs84 来估算距离？
        - 当前绘制对象本来就是 JSON 中相邻轨迹点形成的三维线段；
        - 对裁剪后的屏幕边界点，直接在这条三维线段上反求参数，比先由像素反算 WGS84 再求距离更直接；
        - 即使线段的一端来自近平面裁剪，本函数仍能得到该可见线段端点对应的相机坐标。

        参数含义：
        - lambda=0 表示 pa；lambda=1 表示 pb。
        """
        try:
            pa = np.asarray(pa, dtype=np.float64)
            pb = np.asarray(pb, dtype=np.float64)
            d = pb - pa
            x = (float(u) - float(cx)) / float(fx)
            y = (float(v) - float(cy)) / float(fy)

            candidates = []
            denom_x = float(d[0]) - x * float(d[2])
            if abs(denom_x) > eps:
                candidates.append((x * float(pa[2]) - float(pa[0])) / denom_x)

            denom_y = float(d[1]) - y * float(d[2])
            if abs(denom_y) > eps:
                candidates.append((y * float(pa[2]) - float(pa[1])) / denom_y)

            if not candidates:
                lam = float(fallback_t)
            else:
                lam = float(sum(candidates) / len(candidates))

            if not math.isfinite(lam):
                lam = float(fallback_t)
            return max(0.0, min(1.0, lam))
        except Exception:
            return max(0.0, min(1.0, float(fallback_t)))

    @staticmethod
    def _distance_to_dynamic_thickness(distance: float,
                                       near_distance: float,
                                       far_distance: float,
                                       base_thickness: int,
                                       near_factor: float = 1.5,
                                       far_factor: float = 0.5) -> int:
        """把距离映射为线宽：近处粗，远处细。

        线宽范围约为：
        - 最近：near_factor * base_thickness
        - 最远：far_factor * base_thickness
        """
        base = max(1, int(base_thickness))
        near_th = max(1, CameraProjector._round_half_up(float(base) * float(near_factor)))
        far_th = max(1, CameraProjector._round_half_up(float(base) * float(far_factor)))

        if not (math.isfinite(distance) and math.isfinite(near_distance) and math.isfinite(far_distance)):
            return base
        if far_distance <= near_distance + 1e-9:
            return base

        ratio = (float(distance) - float(near_distance)) / (float(far_distance) - float(near_distance))
        ratio = max(0.0, min(1.0, ratio))

        th = near_th + ratio * (far_th - near_th)
        return max(1, CameraProjector._round_half_up(th))

    @staticmethod
    def _distance_to_vertical_drop(distance: float,
                                   near_distance: float,
                                   drop_far_distance: float,
                                   max_drop_px: float = 20.0) -> float:
        """把距离映射为画面中的下移像素量。

        甲方需求：
        - 近端基本不下移；
        - 距离越远，下移越明显；
        - 90% 分位数及更远处达到最大下移量，默认 20px。

        注意：这里的“下移”是视觉效果处理，不是严格几何投影。
        OpenCV 图像坐标中 y 向下为正，因此返回值会加到 y 坐标上。
        """
        if not (math.isfinite(distance) and math.isfinite(near_distance) and math.isfinite(drop_far_distance)):
            return 0.0
        if drop_far_distance <= near_distance + 1e-9:
            return 0.0

        ratio = (float(distance) - float(near_distance)) / (float(drop_far_distance) - float(near_distance))
        ratio = max(0.0, min(1.0, ratio))
        return float(max_drop_px) * ratio

    @staticmethod
    def _dash_visible_at(s: float,
                         dash_len_px: float,
                         gap_len_px: float) -> bool:
        """判断虚线在当前沿线距离 s 处是否应绘制。"""
        dash_len_px = max(1.0, float(dash_len_px))
        gap_len_px = max(1.0, float(gap_len_px))
        period = dash_len_px + gap_len_px
        return (float(s) % period) < dash_len_px

    @staticmethod
    def _curve_level_from_id(curve_id: str) -> str:
        """根据 curve_id 前缀判断线条层级：inner / middle / outer / default。"""
        cid = str(curve_id or "")
        if cid.startswith("outside_border_-40m") or cid.startswith("inside_border_-40m"):
            return "inner"
        if cid.startswith("outside_border_-20m") or cid.startswith("inside_border_-20m"):
            return "middle"
        if cid.startswith("outside_border_0m") or cid.startswith("inside_border_0m"):
            return "outer"
        return "default"

    @staticmethod
    def _safe_positive_int(value: Any, default_value: int) -> int:
        """把用户/JSON传入的线宽转换为 >=1 的整数像素。"""
        try:
            v = int(round(float(value)))
            if v >= 1:
                return v
        except Exception:
            pass
        return max(1, int(default_value))

    @staticmethod
    def _safe_float_or_none(value: Any) -> Optional[float]:
        """把可选距离参数转换为 float；None 或非法值返回 None，表示使用自动分位数。"""
        if value is None:
            return None
        try:
            v = float(value)
            if math.isfinite(v):
                return v
        except Exception:
            pass
        return None

    @staticmethod
    def _clamp01(value: Any, default_value: float = 1.0) -> float:
        """亮度比例限制在 [0, 1]。"""
        try:
            v = float(value)
            if math.isfinite(v):
                return max(0.0, min(1.0, v))
        except Exception:
            pass
        return max(0.0, min(1.0, float(default_value)))

    @staticmethod
    def _select_group_value(curve_id: str,
                            inner_value: Any,
                            middle_value: Any,
                            outer_value: Any,
                            default_value: Any) -> Any:
        """根据 curve_id 层级选择 test_zone_projector.py 中配置的分组参数。"""
        level = CameraProjector._curve_level_from_id(curve_id)
        if level == "inner":
            return inner_value
        if level == "middle":
            return middle_value
        if level == "outer":
            return outer_value
        return default_value

    @staticmethod
    def _apply_brightness_to_color(color_bgr: Tuple[int, int, int], brightness: float) -> Tuple[int, int, int]:
        """在 JSON 的 color_bgr 基础上叠加亮度比例。brightness=1 保持原色，brightness=0 变为黑色。"""
        b = CameraProjector._clamp01(brightness, 1.0)
        arr = np.asarray(color_bgr, dtype=np.float64) * b
        arr = np.clip(arr, 0.0, 255.0)
        return int(round(float(arr[0]))), int(round(float(arr[1]))), int(round(float(arr[2])))

    @staticmethod
    def _edge_margin_px(img_w: int, img_h: int, edge_margin_ratio: float) -> int:
        """根据图片短边比例计算边缘不绘制区域宽度，单位：像素。"""
        try:
            ratio = float(edge_margin_ratio)
        except Exception:
            ratio = 0.0
        ratio = max(0.0, min(0.49, ratio))
        return int(math.floor(float(min(int(img_w), int(img_h))) * ratio))

    @staticmethod
    def _restore_edge_margin(img: np.ndarray, original_img: np.ndarray, margin_px: int) -> None:
        """绘制完成后恢复图片边缘区域，确保抗锯齿、线宽、下移等效果不会溢出到边缘禁画区。"""
        if img is None or original_img is None or img.size == 0 or original_img.size == 0:
            return
        m = int(margin_px)
        if m <= 0:
            return
        h, w = img.shape[:2]
        m = min(m, h // 2, w // 2)
        if m <= 0:
            return
        img[:m, :] = original_img[:m, :]
        img[h - m:, :] = original_img[h - m:, :]
        img[:, :m] = original_img[:, :m]
        img[:, w - m:] = original_img[:, w - m:]

    @staticmethod
    def _draw_line_dynamic_thickness(img: np.ndarray,
                                     p0: Tuple[int, int],
                                     p1: Tuple[int, int],
                                     color: Tuple[int, int, int],
                                     base_thickness: int,
                                     dist0: float,
                                     dist1: float,
                                     thickness_near_distance: float,
                                     thickness_far_distance: float,
                                     drop_near_distance: float,
                                     drop_far_distance: float,
                                     line_type: str = "solid",
                                     fixed_dash_len_px: float = 8.0,
                                     fixed_gap_len_px: float = 6.0,
                                     max_drop_px: float = 20.0,
                                     drop_max_step_px: float = 55.0,
                                     thickness_max_step_px: float = 55.0,
                                     near_factor: float = 1.5,
                                     far_factor: float = 0.5) -> None:
        """把一条线段拆成若干小段，按距离渐变绘制。

        支持：
        1. 近粗远细：使用 thickness_* 参数独立控制；
        2. 实线/虚线：由 curve["line_type"] 控制，solid 实线，dash 虚线；虚线实段/间隔由调用方传入；
        3. 远处下移：使用 drop_* 参数独立控制，和近粗远细不共享距离参数。
        """
        x0, y0 = int(p0[0]), int(p0[1])
        x1, y1 = int(p1[0]), int(p1[1])
        dx = float(x1 - x0)
        dy = float(y1 - y0)
        length = math.hypot(dx, dy)

        line_type = str(line_type or "solid").strip().lower()
        if line_type not in ("solid", "dash"):
            line_type = "solid"

        base_thickness = max(1, int(base_thickness))
        fixed_dash_len_px = max(1.0, float(fixed_dash_len_px))
        fixed_gap_len_px = max(1.0, float(fixed_gap_len_px))
        drop_max_step_px = max(2.0, float(drop_max_step_px))
        thickness_max_step_px = max(2.0, float(thickness_max_step_px))
        solid_step_px = max(2.0, min(drop_max_step_px, thickness_max_step_px))

        if length <= 1e-6:
            mid_dist = 0.5 * (float(dist0) + float(dist1))
            drop = CameraProjector._distance_to_vertical_drop(mid_dist, drop_near_distance, drop_far_distance, max_drop_px)
            th = CameraProjector._distance_to_dynamic_thickness(
                mid_dist, thickness_near_distance, thickness_far_distance,
                base_thickness, near_factor=near_factor, far_factor=far_factor
            )
            if line_type == "solid":
                cv2.circle(img, (x0, int(round(y0 + drop))), max(1, th // 2), color, thickness=-1, lineType=cv2.LINE_AA)
            return

        if line_type == "dash":
            # 虚线采用固定屏幕像素长度和固定屏幕像素间隔，不随线宽变化。
            period = fixed_dash_len_px + fixed_gap_len_px
            s = 0.0
            max_dash_count = int(math.ceil(length / max(1.0, period))) + 2
            max_dash_count = min(max_dash_count, 10000)

            for _ in range(max_dash_count):
                dash_start = s
                dash_end = min(s + fixed_dash_len_px, length)
                if dash_start >= length:
                    break

                t_a = dash_start / length
                t_b = dash_end / length
                t_m = 0.5 * (t_a + t_b)

                dist_a = float(dist0) + (float(dist1) - float(dist0)) * t_a
                dist_b = float(dist0) + (float(dist1) - float(dist0)) * t_b
                mid_dist = float(dist0) + (float(dist1) - float(dist0)) * t_m

                drop_a = CameraProjector._distance_to_vertical_drop(dist_a, drop_near_distance, drop_far_distance, max_drop_px)
                drop_b = CameraProjector._distance_to_vertical_drop(dist_b, drop_near_distance, drop_far_distance, max_drop_px)

                xa = int(round(x0 + dx * t_a))
                ya = int(round(y0 + dy * t_a + drop_a))
                xb = int(round(x0 + dx * t_b))
                yb = int(round(y0 + dy * t_b + drop_b))

                th = CameraProjector._distance_to_dynamic_thickness(
                    mid_dist, thickness_near_distance, thickness_far_distance,
                    base_thickness, near_factor=near_factor, far_factor=far_factor
                )
                if abs(xa - xb) <= 1 and abs(ya - yb) <= 1:
                    cv2.circle(img, (xa, ya), max(1, th // 2), color, thickness=-1, lineType=cv2.LINE_AA)
                else:
                    cv2.line(img, (xa, ya), (xb, yb), color, th, lineType=cv2.LINE_AA)

                s += period
            return

        # 实线：按较小的步长拆分，保证“近粗远细”和“远处下移”都足够平滑。
        steps = max(1, int(math.ceil(length / solid_step_px)))
        steps = min(240, steps)

        for k in range(steps):
            t_a = k / steps
            t_b = (k + 1) / steps
            t_m = 0.5 * (t_a + t_b)

            dist_a = float(dist0) + (float(dist1) - float(dist0)) * t_a
            dist_b = float(dist0) + (float(dist1) - float(dist0)) * t_b
            mid_dist = float(dist0) + (float(dist1) - float(dist0)) * t_m

            drop_a = CameraProjector._distance_to_vertical_drop(dist_a, drop_near_distance, drop_far_distance, max_drop_px)
            drop_b = CameraProjector._distance_to_vertical_drop(dist_b, drop_near_distance, drop_far_distance, max_drop_px)

            xa = int(round(x0 + dx * t_a))
            ya = int(round(y0 + dy * t_a + drop_a))
            xb = int(round(x0 + dx * t_b))
            yb = int(round(y0 + dy * t_b + drop_b))

            th = CameraProjector._distance_to_dynamic_thickness(
                mid_dist, thickness_near_distance, thickness_far_distance,
                base_thickness, near_factor=near_factor, far_factor=far_factor
            )
            cv2.line(img, (xa, ya), (xb, yb), color, th, lineType=cv2.LINE_AA)

    def project_points_from_json(self,
                                 border_json_path: str,
                                 info: Dict[str, Any],
                                 img: np.ndarray,
                                 ground_alt: float,
                                 force_35mm_equiv: Optional[bool] = None,
                                 line_color_bgr: Optional[Tuple[int, int, int]] = None,
                                 line_thickness: Optional[int] = None,
                                 use_cache: bool = True,
                                 draw: bool = True,
                                 fixed_dash_len_px: float = 8.0,
                                 fixed_gap_len_px: float = 6.0,
                                 json_thickness_enable: bool = False,
                                 line_thickness_inner: Optional[int] = None,
                                 line_thickness_middle: Optional[int] = None,
                                 line_thickness_outer: Optional[int] = None,
                                 line_brightness_inner: float = 1.0,
                                 line_brightness_middle: float = 1.0,
                                 line_brightness_outer: float = 1.0,
                                 drop_max_drop_px: float = 20.0,
                                 drop_max_step_px: float = 55.0,
                                 drop_near_distance: Optional[float] = None,
                                 drop_far_distance: Optional[float] = None,
                                 thickness_near_factor: float = 1.5,
                                 thickness_far_factor: float = 0.5,
                                 thickness_max_step_px: float = 55.0,
                                 thickness_near_distance: Optional[float] = None,
                                 thickness_far_distance: Optional[float] = None,
                                 edge_margin_ratio: float = 0.03) -> Tuple[Dict[str, List[Dict[str, int]]], np.ndarray]:
        """
        将保护区界线投影到图像上。

        只连接 JSON 中原本相邻的轨迹点：
        - 前方点↔前方点：正常投影，允许两点都在画外但线段穿过画面；
        - 后方点↔前方点：把后方端点裁剪到近平面再投影；
        - 后方点↔后方点：跳过；
        - 投影异常：跳过该相邻段，不跨段重连。

        本版本的关键修正：
        - 不再用 MAX_PIXEL_ABS 过滤超大投影点；
        - 不再把超大点按图像中心方向压缩后交给 cv2.clipLine；
        - 改为在原始浮点投影坐标上做线段-图像矩形裁剪，避免数值超限，同时不改变线段方向；
        - 支持“近粗远细”的动态线宽绘制，且近粗远细参数与下移参数相互独立；
        - 支持 curve["line_type"] 控制线型："solid" 为实线，"dash" 为虚线；
        - 支持 curve["line_thickness"] 或 test_zone_projector.py 分组配置控制每条曲线线宽；
        - 支持 test_zone_projector.py 分组亮度比例控制每条曲线颜色亮度；
        - 支持随距离拉远让画线在画面中线性下移；
        - 支持按图片短边比例设置边缘禁画区。
        """
        if img is None:
            raise ValueError("输入的 img 为空或无效")

        EPS_DEPTH = 1e-1
        DEPTH_TOL = 1e-9  # 近裁剪面浮点容差，避免 0.099999999999994 这类误差造成漏画

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
            default_thickness = 4
        else:
            default_thickness = max(1, int(line_thickness))

        # 如果调用方没有传分组线宽，则退回到 line_thickness/default_thickness，保证兼容旧调用方式。
        line_thickness_inner = self._safe_positive_int(line_thickness_inner, default_thickness) if line_thickness_inner is not None else default_thickness
        line_thickness_middle = self._safe_positive_int(line_thickness_middle, default_thickness) if line_thickness_middle is not None else default_thickness
        line_thickness_outer = self._safe_positive_int(line_thickness_outer, default_thickness) if line_thickness_outer is not None else default_thickness

        line_brightness_inner = self._clamp01(line_brightness_inner, 1.0)
        line_brightness_middle = self._clamp01(line_brightness_middle, 1.0)
        line_brightness_outer = self._clamp01(line_brightness_outer, 1.0)

        default_color = tuple(int(x) for x in (line_color_bgr or (0, 200, 255)))

        edge_margin_px = self._edge_margin_px(img_w, img_h, edge_margin_ratio)
        original_img_for_edge_restore = img.copy() if draw and edge_margin_px > 0 else None

        results: Dict[str, List[Dict[str, int]]] = {}
        all_draw_segments: List[Dict[str, Any]] = []

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

            # 甲方新增字段：line_type 与 curve_id / color_bgr / points 同层级。
            # solid：实线；dash：虚线。缺失或异常时默认按实线处理，兼容旧 JSON。
            line_type = str(curve.get("line_type", "solid") or "solid").strip().lower()
            if line_type not in ("solid", "dash"):
                line_type = "solid"

            # 曲线线宽：
            # - json_thickness_enable=True 且 JSON 中存在 curve["line_thickness"] 时，优先使用 JSON 的线宽；
            # - 否则按 curve_id 所属层级使用 test_zone_projector.py 中的 INNER/MIDDLE/OUTER 线宽。
            group_thickness = self._select_group_value(
                cid, line_thickness_inner, line_thickness_middle, line_thickness_outer, default_thickness
            )
            curve_thickness = self._safe_positive_int(group_thickness, default_thickness)
            if bool(json_thickness_enable) and "line_thickness" in curve:
                curve_thickness = self._safe_positive_int(curve.get("line_thickness"), curve_thickness)

            # 曲线亮度：在 JSON 的 color_bgr 基础上叠加分组亮度比例。
            # brightness=1.0 保持原色；brightness=0.5 表示颜色整体减半；brightness=0.0 近似黑色。
            curve_brightness = self._select_group_value(
                cid, line_brightness_inner, line_brightness_middle, line_brightness_outer, 1.0
            )
            curve_brightness = self._clamp01(curve_brightness, 1.0)
            color = self._apply_brightness_to_color(color, curve_brightness)

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

                # 一前一后：在三维相机坐标中先裁到近平面。
                # 这样可以避免把相机背后的点直接投影到二维图像中。
                if za < EPS_DEPTH <= zb:
                    denom = zb - za
                    if abs(denom) < 1e-12:
                        continue
                    t = (EPS_DEPTH - za) / denom
                    pa = pa + t * (pb - pa)
                    # 浮点计算后 pa[2] 可能变成 0.099999999999994 这种略小于 EPS_DEPTH 的数，
                    # 如果后面严格判断 pa[2] < EPS_DEPTH，就会把本应绘制的前后交界线段误删。
                    pa[2] = EPS_DEPTH
                elif zb < EPS_DEPTH <= za:
                    denom = za - zb
                    if abs(denom) < 1e-12:
                        continue
                    t = (EPS_DEPTH - zb) / denom
                    pb = pb + t * (pa - pb)
                    pb[2] = EPS_DEPTH

                # 只剔除真正仍在近平面后的端点；允许极小浮点误差。
                if pa[2] < EPS_DEPTH - DEPTH_TOL or pb[2] < EPS_DEPTH - DEPTH_TOL:
                    continue
                if pa[2] < EPS_DEPTH:
                    pa[2] = EPS_DEPTH
                if pb[2] < EPS_DEPTH:
                    pb[2] = EPS_DEPTH

                u1 = fx * (pa[0] / pa[2]) + cx
                v1 = fy * (pa[1] / pa[2]) + cy
                u2 = fx * (pb[0] / pb[2]) + cx
                v2 = fy * (pb[1] / pb[2]) + cy

                # 只丢弃 NaN/Inf。有限但极大的 u/v 不能直接丢弃，
                # 否则会漏画很多“从画外延伸进画面”的线段。
                if not self._is_finite_uv(u1, v1):
                    continue
                if not self._is_finite_uv(u2, v2):
                    continue

                # 关键：使用原始浮点投影线段做裁剪，而不是先压缩端点。
                clipped = self._clip_segment_to_rect_float(u1, v1, u2, v2, img_w, img_h, margin_px=edge_margin_px)
                if clipped is None:
                    continue

                cu1, cv1, cu2, cv2_, t0_2d, t1_2d = clipped
                p0 = self._safe_int_point(cu1, cv1, img_w, img_h)
                p1 = self._safe_int_point(cu2, cv2_, img_w, img_h)

                # 为“近粗远细”线宽渐变估算可见线段两端到相机的三维距离。
                # 这里不是简单用原始端点 pa/pb 的距离，而是对裁剪后的屏幕端点反求其在三维线段上的位置。
                lam0 = self._projective_param_from_uv(pa, pb, cu1, cv1, fx, fy, cx, cy, fallback_t=t0_2d)
                lam1 = self._projective_param_from_uv(pa, pb, cu2, cv2_, fx, fy, cx, cy, fallback_t=t1_2d)
                cam0 = pa + lam0 * (pb - pa)
                cam1 = pa + lam1 * (pb - pa)
                dist0 = float(np.linalg.norm(cam0))
                dist1 = float(np.linalg.norm(cam1))

                seg_item = {
                    "u1": p0[0], "v1": p0[1],
                    "u2": p1[0], "v2": p1[1],
                    "dist1": dist0, "dist2": dist1,
                    "color_bgr": color,
                    "line_type": line_type,
                    "line_thickness": curve_thickness,
                    "line_brightness": curve_brightness,
                }
                curve_segments.append(seg_item)
                all_draw_segments.append(seg_item)

            results[cid] = curve_segments

        # 统一计算本帧所有可见线段的距离范围，再绘制。
        # 这里分两套距离参数：
        # - thickness_*：只控制“近粗远细”；
        # - drop_*：只控制“随距离增加向下偏移”。
        if draw and all_draw_segments:
            dist_values = []
            for seg in all_draw_segments:
                d0 = float(seg.get("dist1", float("nan")))
                d1 = float(seg.get("dist2", float("nan")))
                if math.isfinite(d0) and d0 >= 0.0:
                    dist_values.append(d0)
                if math.isfinite(d1) and d1 >= 0.0:
                    dist_values.append(d1)

            if len(dist_values) >= 2:
                arr = np.asarray(dist_values, dtype=np.float64)
                auto_near_distance = float(np.percentile(arr, 5))
                auto_thickness_far_distance = float(np.percentile(arr, 95))
                auto_drop_far_distance = float(np.percentile(arr, 90))
                if (not math.isfinite(auto_near_distance)
                        or not math.isfinite(auto_thickness_far_distance)
                        or auto_thickness_far_distance <= auto_near_distance):
                    auto_near_distance = float(np.min(arr))
                    auto_thickness_far_distance = float(np.max(arr))
                if (not math.isfinite(auto_drop_far_distance)
                        or auto_drop_far_distance <= auto_near_distance):
                    auto_drop_far_distance = auto_thickness_far_distance
            else:
                auto_near_distance = 0.0
                auto_thickness_far_distance = 1.0
                auto_drop_far_distance = 1.0

            user_thickness_near = self._safe_float_or_none(thickness_near_distance)
            user_thickness_far = self._safe_float_or_none(thickness_far_distance)
            user_drop_near = self._safe_float_or_none(drop_near_distance)
            user_drop_far = self._safe_float_or_none(drop_far_distance)

            actual_thickness_near_distance = auto_near_distance if user_thickness_near is None else user_thickness_near
            actual_thickness_far_distance = auto_thickness_far_distance if user_thickness_far is None else user_thickness_far
            if actual_thickness_far_distance <= actual_thickness_near_distance + 1e-9:
                actual_thickness_far_distance = actual_thickness_near_distance + 1.0

            actual_drop_near_distance = auto_near_distance if user_drop_near is None else user_drop_near
            actual_drop_far_distance = auto_drop_far_distance if user_drop_far is None else user_drop_far
            if actual_drop_far_distance <= actual_drop_near_distance + 1e-9:
                actual_drop_far_distance = actual_drop_near_distance + 1.0

            for seg in all_draw_segments:
                p0 = (int(seg["u1"]), int(seg["v1"]))
                p1 = (int(seg["u2"]), int(seg["v2"]))
                color_seg = tuple(int(x) for x in seg.get("color_bgr", default_color))
                line_type_seg = str(seg.get("line_type", "solid") or "solid").strip().lower()
                base_thickness_seg = self._safe_positive_int(seg.get("line_thickness", default_thickness), default_thickness)
                self._draw_line_dynamic_thickness(
                    img, p0, p1, color_seg, base_thickness_seg,
                    float(seg.get("dist1", 0.0)),
                    float(seg.get("dist2", 0.0)),
                    actual_thickness_near_distance,
                    actual_thickness_far_distance,
                    actual_drop_near_distance,
                    actual_drop_far_distance,
                    line_type=line_type_seg,
                    fixed_dash_len_px=fixed_dash_len_px,
                    fixed_gap_len_px=fixed_gap_len_px,
                    max_drop_px=drop_max_drop_px,
                    drop_max_step_px=drop_max_step_px,
                    thickness_max_step_px=thickness_max_step_px,
                    near_factor=thickness_near_factor,
                    far_factor=thickness_far_factor,
                )

            # 双保险：即使线宽、抗锯齿或远处下移让线条溢出到边缘区域，也强制恢复原图边缘。
            if edge_margin_px > 0 and original_img_for_edge_restore is not None:
                self._restore_edge_margin(img, original_img_for_edge_restore, edge_margin_px)

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
