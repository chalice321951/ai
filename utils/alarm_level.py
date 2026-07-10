# -*- coding: utf-8 -*-

from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, List, Optional, Tuple


_BOUNDARY_CURVE_RE = re.compile(
    r"^(?P<side>inside|outside)_border_(?P<offset>-?\d+(?:\.\d+)?)m_(?P<segment>\d+)$",
    re.IGNORECASE,
)
_BOUNDARY_RULES_CACHE: Dict[str, Dict[str, Any]] = {}
_COLOR_NAME_BY_BGR = {
    (0, 0, 255): "red",
    (0, 255, 255): "yellow",
    (38, 167, 255): "orange",
    (0, 255, 0): "green",
}
_LEVEL_BY_COLOR = {
    "red": 1,
    "yellow": 2,
    "orange": 3,
}
_STREAM_STRATEGY_BY_NAME = {
    "龙王庙": "between_orange_x_scanline_level1",
    "岗下江南郡": "orange_enclosed_level1",
    "国动塔": "orange_enclosed_level1_strict",
    "中央香榭": "orange_enclosed_level1_strict",
    "罗家集": "luojiaji_mixed",
}


def parse_boundary_curve_id(curve_id: Any) -> Optional[Dict[str, Any]]:
    if not isinstance(curve_id, str):
        return None
    match = _BOUNDARY_CURVE_RE.match(curve_id.strip())
    if not match:
        return None
    try:
        return {
            "side": str(match.group("side")).lower(),
            "offset_m": float(match.group("offset")),
            "segment": int(match.group("segment")),
        }
    except Exception:
        return None


def _normalize_bgr(color: Any) -> Optional[Tuple[int, int, int]]:
    if not isinstance(color, (list, tuple)) or len(color) != 3:
        return None
    try:
        return int(color[0]), int(color[1]), int(color[2])
    except Exception:
        return None


def _color_name_from_bgr(color: Any) -> Optional[str]:
    return _COLOR_NAME_BY_BGR.get(_normalize_bgr(color))


def _extract_curve_color_name(points: Any, fallback_color: Any = None) -> Optional[str]:
    color_name = _color_name_from_bgr(fallback_color)
    if color_name:
        return color_name
    if not isinstance(points, list):
        return None
    for point in points:
        if not isinstance(point, dict):
            continue
        color_name = _color_name_from_bgr(point.get("color_bgr"))
        if color_name:
            return color_name
    return None


def _extract_curve_uv_points(points: Any) -> List[Tuple[int, int]]:
    out: List[Tuple[int, int]] = []
    if not isinstance(points, list):
        return out
    for point in points:
        if not isinstance(point, dict):
            continue
        if "u" in point and "v" in point:
            try:
                out.append((int(point.get("u")), int(point.get("v"))))
            except Exception:
                continue
            continue
        if all(key in point for key in ("u1", "v1", "u2", "v2")):
            try:
                out.append((int(point.get("u1")), int(point.get("v1"))))
                out.append((int(point.get("u2")), int(point.get("v2"))))
            except Exception:
                continue
    return _dedupe_consecutive_points(out)


def _dedupe_consecutive_points(points: List[Tuple[Any, Any]]) -> List[Tuple[Any, Any]]:
    out: List[Tuple[Any, Any]] = []
    for point in points or []:
        if not out or out[-1] != point:
            out.append(point)
    return out


def _distance_sq(a: Tuple[Any, Any], b: Tuple[Any, Any]) -> float:
    dx = float(a[0]) - float(b[0])
    dy = float(a[1]) - float(b[1])
    return dx * dx + dy * dy


def _merge_polylines_by_endpoints(lines: List[List[Tuple[Any, Any]]]) -> List[Tuple[Any, Any]]:
    pending = [_dedupe_consecutive_points(list(line)) for line in (lines or []) if len(line) >= 2]
    if not pending:
        return []

    merged = pending.pop(0)
    while pending:
        best_idx = -1
        best_candidate: List[Tuple[Any, Any]] = []
        best_distance = None
        for idx, line in enumerate(pending):
            for candidate in (line, list(reversed(line))):
                dist = _distance_sq(merged[-1], candidate[0])
                if best_distance is None or dist < best_distance:
                    best_idx = idx
                    best_candidate = candidate
                    best_distance = dist
        if best_idx < 0:
            break
        pending.pop(best_idx)
        if merged[-1] == best_candidate[0]:
            merged.extend(best_candidate[1:])
        else:
            merged.extend(best_candidate)
        merged = _dedupe_consecutive_points(merged)
    return merged


def _parse_geo_curve_points(points: Any) -> List[Tuple[float, float]]:
    out: List[Tuple[float, float]] = []
    if not isinstance(points, list):
        return out
    for point in points:
        if not isinstance(point, str):
            continue
        try:
            lon, lat, _alt = map(float, point.split(","))
        except Exception:
            continue
        out.append((lon, lat))
    return out


def _build_geo_polyline_from_segments(segments: Dict[int, List[Tuple[float, float]]]) -> List[Tuple[float, float]]:
    ordered = [segments.get(seg_id) or [] for seg_id in sorted(segments.keys())]
    return _merge_polylines_by_endpoints(ordered)


def curve_colors_from_border_file(border_json_path: str) -> Dict[str, str]:
    path = str(border_json_path or "").strip()
    if not path:
        return {}
    try:
        mtime = float(os.path.getmtime(path))
    except Exception:
        mtime = None
    cached = _BOUNDARY_RULES_CACHE.get(path)
    if cached is not None and cached.get("_cache_mtime") == mtime:
        rules = cached.get("rules") or {}
        colors = rules.get("curve_colors") if isinstance(rules, dict) else {}
        return colors if isinstance(colors, dict) else {}

    curve_colors: Dict[str, str] = {}
    try:
        with open(path, "r", encoding="utf-8") as fh:
            items = json.load(fh)
    except Exception:
        _BOUNDARY_RULES_CACHE[path] = {"_cache_mtime": mtime, "rules": {"curve_colors": {}}}
        return {}

    grouped: Dict[Tuple[str, float], Dict[int, List[Tuple[float, float]]]] = {}
    key_colors: Dict[Tuple[str, float], str] = {}
    for item in items:
        if not isinstance(item, dict):
            continue
        curve_id = item.get("curve_id")
        parsed = parse_boundary_curve_id(curve_id)
        color_name = _extract_curve_color_name([], fallback_color=item.get("color_bgr"))
        if isinstance(curve_id, str) and color_name:
            curve_colors[curve_id] = color_name
        if parsed is None:
            continue
        pts = _parse_geo_curve_points(item.get("points"))
        if len(pts) < 2:
            continue
        key = (str(parsed["side"]), float(parsed["offset_m"]))
        grouped.setdefault(key, {})[int(parsed["segment"])] = pts
        if color_name:
            key_colors[key] = color_name

    rules = {"curve_colors": curve_colors, "grouped": grouped, "key_colors": key_colors}
    _BOUNDARY_RULES_CACHE[path] = {"_cache_mtime": mtime, "rules": rules}
    return curve_colors


def visible_boundary_colors_from_projected(
    projected_curves: Optional[Dict[str, List[Dict[str, Any]]]] = None,
    border_json_path: str = "",
) -> List[str]:
    curve_colors = curve_colors_from_border_file(border_json_path)
    names: List[str] = []
    for curve_id, pts in (projected_curves or {}).items():
        if not isinstance(curve_id, str) or not isinstance(pts, list) or not pts:
            continue
        color_name = _extract_curve_color_name(pts)
        if color_name is None:
            color_name = curve_colors.get(curve_id)
        if color_name and color_name not in names:
            names.append(color_name)
    return names


def _polyline_scanline_intersections_x(y: float, points: List[Tuple[int, int]]) -> List[float]:
    xs: List[float] = []
    pts = _dedupe_consecutive_points(list(points or []))
    if len(pts) < 2:
        return xs
    for idx in range(len(pts) - 1):
        x1, y1 = pts[idx]
        x2, y2 = pts[idx + 1]
        y1f = float(y1)
        y2f = float(y2)
        if abs(y2f - y1f) < 1e-6:
            continue
        ymin = min(y1f, y2f)
        ymax = max(y1f, y2f)
        if not (ymin <= float(y) < ymax):
            continue
        ratio = (float(y) - y1f) / (y2f - y1f)
        xs.append(float(x1) + ratio * (float(x2) - float(x1)))
    xs.sort()
    out: List[float] = []
    for x in xs:
        if not out or abs(out[-1] - x) > 1e-3:
            out.append(x)
    return out


def _polyline_vertical_intersections_y(x: float, points: List[Tuple[int, int]]) -> List[float]:
    ys: List[float] = []
    pts = _dedupe_consecutive_points(list(points or []))
    if len(pts) < 2:
        return ys
    for idx in range(len(pts) - 1):
        x1, y1 = pts[idx]
        x2, y2 = pts[idx + 1]
        x1f = float(x1)
        x2f = float(x2)
        if abs(x2f - x1f) < 1e-6:
            continue
        xmin = min(x1f, x2f)
        xmax = max(x1f, x2f)
        if not (xmin <= float(x) < xmax):
            continue
        ratio = (float(x) - x1f) / (x2f - x1f)
        ys.append(float(y1) + ratio * (float(y2) - float(y1)))
    ys.sort()
    out: List[float] = []
    for y in ys:
        if not out or abs(out[-1] - y) > 1e-3:
            out.append(y)
    return out


def _extend_polyline_for_scanline(points: List[Tuple[int, int]], scan_y: float) -> List[Tuple[float, float]]:
    pts = [(float(px), float(py)) for px, py in _dedupe_consecutive_points(list(points or []))]
    if len(pts) < 2:
        return pts
    ys = [py for _, py in pts]
    visible_span_y = max(ys) - min(ys)
    if visible_span_y < 6.0:
        return pts
    max_extend_y = max(80.0, min(260.0, visible_span_y * 6.0))
    extended = list(pts)

    def _build_endpoint_extension(anchor: Tuple[float, float], neighbor: Tuple[float, float]) -> Optional[Tuple[float, float]]:
        ax, ay = anchor
        nx, ny = neighbor
        outward_dy = ay - ny
        if abs(outward_dy) < 1e-6:
            return None
        target_dy = float(scan_y) - ay
        if target_dy * outward_dy <= 0:
            return None
        extend_y = min(max_extend_y, abs(target_dy) + 4.0)
        ratio = extend_y / abs(outward_dy)
        return (
            ax + (ax - nx) * ratio,
            ay + outward_dy / abs(outward_dy) * extend_y,
        )

    first_extension = _build_endpoint_extension(pts[0], pts[1])
    if first_extension is not None:
        extended.insert(0, first_extension)
    last_extension = _build_endpoint_extension(pts[-1], pts[-2])
    if last_extension is not None:
        extended.append(last_extension)
    return extended


def _collect_boundaries_at_y(
    scan_y: float,
    projected_curves: Dict[str, List[Dict[str, Any]]],
    curve_colors: Dict[str, str],
    extend_endpoints: bool = True,
) -> List[Tuple[float, str]]:
    boundaries: List[Tuple[float, str]] = []
    for curve_id, pts in (projected_curves or {}).items():
        if not isinstance(curve_id, str) or not isinstance(pts, list):
            continue
        uv = _extract_curve_uv_points(pts)
        if len(uv) < 2:
            continue
        parsed = parse_boundary_curve_id(curve_id)
        if parsed and parsed.get("side") == "inside":
            continue
        color_name = _extract_curve_color_name(pts)
        if color_name is None:
            color_name = curve_colors.get(curve_id)
        if color_name not in _LEVEL_BY_COLOR:
            continue
        scanline_uv = _extend_polyline_for_scanline(uv, scan_y) if extend_endpoints else uv
        for x_cross in _polyline_scanline_intersections_x(scan_y, scanline_uv):
            boundaries.append((float(x_cross), str(color_name)))
    boundaries.sort(key=lambda item: item[0])
    return boundaries


def _collect_color_positions_at_x(
    scan_x: float,
    projected_curves: Dict[str, List[Dict[str, Any]]],
    curve_colors: Dict[str, str],
    target_color: str,
) -> List[float]:
    positions: List[float] = []
    for curve_id, pts in (projected_curves or {}).items():
        if not isinstance(curve_id, str) or not isinstance(pts, list):
            continue
        uv = _extract_curve_uv_points(pts)
        if len(uv) < 2:
            continue
        parsed = parse_boundary_curve_id(curve_id)
        if parsed and parsed.get("side") == "inside":
            continue
        color_name = _extract_curve_color_name(pts)
        if color_name is None:
            color_name = curve_colors.get(curve_id)
        if color_name != target_color:
            continue
        positions.extend(_polyline_vertical_intersections_y(scan_x, uv))
    positions.sort()
    out: List[float] = []
    for pos in positions:
        if not out or abs(out[-1] - pos) > 1e-3:
            out.append(pos)
    return out


def _point_in_polygon(point: Tuple[float, float], polygon: List[Tuple[float, float]]) -> bool:
    if len(polygon) < 3:
        return False
    x = float(point[0])
    y = float(point[1])
    inside = False
    pts = list(polygon)
    if pts[0] != pts[-1]:
        pts.append(pts[0])
    for idx in range(len(pts) - 1):
        x1, y1 = float(pts[idx][0]), float(pts[idx][1])
        x2, y2 = float(pts[idx + 1][0]), float(pts[idx + 1][1])
        if (y1 > y) == (y2 > y):
            continue
        denom = (y2 - y1)
        if abs(denom) < 1e-9:
            continue
        x_cross = x1 + (y - y1) * (x2 - x1) / denom
        if x_cross >= x:
            inside = not inside
    return inside


def _polyline_area(points: List[Tuple[float, float]]) -> float:
    if len(points) < 3:
        return 0.0
    pts = list(points)
    if pts[0] != pts[-1]:
        pts.append(pts[0])
    area = 0.0
    for idx in range(len(pts) - 1):
        x1, y1 = pts[idx]
        x2, y2 = pts[idx + 1]
        area += x1 * y2 - x2 * y1
    return abs(area) * 0.5


def _polyline_bbox(points: List[Tuple[float, float]]) -> Tuple[float, float, float, float]:
    xs = [float(px) for px, _ in points]
    ys = [float(py) for _, py in points]
    return min(xs), min(ys), max(xs), max(ys)


def _collect_outside_orange_polygons(
    projected_curves: Dict[str, List[Dict[str, Any]]],
    curve_colors: Dict[str, str],
) -> List[List[Tuple[float, float]]]:
    polygons: List[List[Tuple[float, float]]] = []
    for curve_id, pts in (projected_curves or {}).items():
        if not isinstance(curve_id, str) or not isinstance(pts, list):
            continue
        parsed = parse_boundary_curve_id(curve_id)
        if not parsed or parsed.get("side") != "outside":
            continue
        color_name = _extract_curve_color_name(pts)
        if color_name is None:
            color_name = curve_colors.get(curve_id)
        if color_name != "orange":
            continue
        uv = _extract_curve_uv_points(pts)
        if len(uv) < 3:
            continue
        poly = [(float(px), float(py)) for px, py in uv]
        if poly[0] != poly[-1]:
            poly.append(poly[0])
        if _polyline_area(poly) < 100.0:
            continue
        polygons.append(poly)
    polygons.sort(key=_polyline_area, reverse=True)
    return polygons


def _resolve_stream_strategy(stream_name: str) -> str:
    text = str(stream_name or "").strip()
    if not text:
        return "default"
    return str(_STREAM_STRATEGY_BY_NAME.get(text) or "default")


def _classify_orange_enclosed_level1_details(
    scan_x: float,
    boundaries: List[Tuple[float, str]],
    projected_curves: Dict[str, List[Dict[str, Any]]],
    curve_colors: Dict[str, str],
    scan_y: float,
) -> Dict[str, Any]:
    details: Dict[str, Any] = {
        "alarm_level": None,
        "reason": "no_boundaries",
        "boundaries": [(float(x), str(color)) for x, color in boundaries],
        "outside_orange": False,
    }
    polygons = _collect_outside_orange_polygons(projected_curves, curve_colors)
    if polygons:
        # Prefer a true pixel polygon from the projected orange boundary. This
        # matches the visible overlay better than scanline min/max heuristics.
        for poly in polygons:
            if _point_in_polygon((scan_x, scan_y), poly):
                details["alarm_level"] = 1
                details["reason"] = "orange_pixel_polygon"
                details["orange_polygon_bbox"] = _polyline_bbox(poly)
                return details
        details["reason"] = "outside_orange_pixel_polygon"
        details["outside_orange"] = True
        details["orange_polygon_bbox"] = _polyline_bbox(polygons[0])
        return details

    if not boundaries:
        details["reason"] = "no_orange_boundaries"
        return details
    orange_positions = [bx for bx, color_name in boundaries if color_name == "orange"]
    if len(orange_positions) >= 2:
        intervals: List[Tuple[float, float]] = []
        sorted_positions = sorted(float(x) for x in orange_positions)
        for idx in range(0, len(sorted_positions) - 1, 2):
            left_orange = sorted_positions[idx]
            right_orange = sorted_positions[idx + 1]
            if right_orange < left_orange:
                left_orange, right_orange = right_orange, left_orange
            intervals.append((left_orange, right_orange))
        for left_orange, right_orange in intervals:
            if left_orange <= scan_x <= right_orange:
                details["alarm_level"] = 1
                details["reason"] = "orange_pixel_band_fallback"
                details["orange_intervals"] = [(float(a), float(b)) for a, b in intervals]
                details["matched_scan_y"] = scan_y
                return details
        if intervals:
            details["reason"] = "outside_orange_pixel_band_fallback"
            details["outside_orange"] = True
            details["orange_intervals"] = [(float(a), float(b)) for a, b in intervals]
            return details
    details["reason"] = "orange_band_unresolved"
    return details


def _classify_orange_above_line_level1_details(
    scan_x: float,
    scan_y: float,
    projected_curves: Dict[str, List[Dict[str, Any]]],
    curve_colors: Dict[str, str],
) -> Dict[str, Any]:
    details: Dict[str, Any] = {
        "alarm_level": None,
        "reason": "no_orange_line_at_x",
        "boundaries": [],
        "outside_orange": False,
    }
    orange_positions = _collect_color_positions_at_x(scan_x, projected_curves, curve_colors, "orange")
    details["boundaries"] = [(float(scan_x), float(y)) for y in orange_positions]
    if not orange_positions:
        return details
    divider_y = min(orange_positions)
    if scan_y < divider_y:
        details["alarm_level"] = 1
        details["reason"] = "orange_above_line"
        return details
    details["reason"] = "below_orange_line"
    details["outside_orange"] = True
    return details


def _classify_between_orange_x_scanline_details(
    scan_x: float,
    scan_y: float,
    projected_curves: Dict[str, List[Dict[str, Any]]],
    curve_colors: Dict[str, str],
) -> Dict[str, Any]:
    """
    垂直扫描线策略：在目标物 x 列上找橙线的 y 交点，判断目标 y 是否夹在两个交点之间。

    适用于两条大致横向的平行橙线之间的保护区（如龙王庙的高架桥场景）：
    - 沿 scan_x 位置画一条垂直线
    - 找出这条垂直线与所有橙线的 y 交点
    - 若目标 scan_y 在最上 y 交点和最下 y 交点之间 -> level 1
    - 否则 -> 不报警
    """
    details: Dict[str, Any] = {
        "alarm_level": None,
        "reason": "no_orange_line_at_x",
        "boundaries": [],
        "outside_orange": False,
    }
    orange_y_positions = _collect_color_positions_at_x(scan_x, projected_curves, curve_colors, "orange")
    details["boundaries"] = [(float(scan_x), float(y)) for y in orange_y_positions]

    # 至少要有两条橙线才能构成"上下夹住"的判断
    if len(orange_y_positions) < 2:
        details["reason"] = "not_enough_orange_at_x"
        return details

    top_y = min(orange_y_positions)
    bottom_y = max(orange_y_positions)

    if top_y <= scan_y <= bottom_y:
        details["alarm_level"] = 1
        details["reason"] = "between_two_orange_x_scanline"
        return details

    details["reason"] = "outside_two_orange_x_scanline"
    details["outside_orange"] = True
    return details


def _classify_stream_specific_details(
    stream_name: str,
    scan_x: float,
    scan_y: float,
    boundaries: List[Tuple[float, str]],
    projected_curves: Dict[str, List[Dict[str, Any]]],
    curve_colors: Dict[str, str],
) -> Optional[Dict[str, Any]]:
    strategy = _resolve_stream_strategy(stream_name)
    if strategy == "default":
        return None
    if strategy == "orange_above_line_level1":
        return _classify_orange_above_line_level1_details(scan_x, scan_y, projected_curves, curve_colors)
    if strategy == "orange_enclosed_level1":
        return _classify_orange_enclosed_level1_details(scan_x, boundaries, projected_curves, curve_colors, scan_y)
    if strategy == "orange_enclosed_level1_strict":
        return _classify_orange_enclosed_level1_details(scan_x, boundaries, projected_curves, curve_colors, scan_y)
    if strategy == "between_orange_x_scanline_level1":
        return _classify_between_orange_x_scanline_details(scan_x, scan_y, projected_curves, curve_colors)
    if strategy == "luojiaji_mixed":
        orange_positions = [bx for bx, color_name in boundaries if color_name == "orange"]
        if len(orange_positions) >= 2 and min(orange_positions) <= scan_x <= max(orange_positions):
            return {
                "alarm_level": 1,
                "reason": "orange_enclosed_override",
                "boundaries": [(float(x), str(color)) for x, color in boundaries],
                "outside_orange": False,
            }
        return _classify_from_boundaries_details(scan_x, boundaries)
    return None


def _single_line_level(boundaries: List[Tuple[float, str]]) -> Optional[int]:
    unique_colors = []
    for _, color_name in boundaries:
        if color_name not in unique_colors:
            unique_colors.append(color_name)
    if len(unique_colors) != 1:
        return None
    return _LEVEL_BY_COLOR.get(unique_colors[0])


def _classify_from_boundaries(scan_x: float, boundaries: List[Tuple[float, str]]) -> Optional[int]:
    if not boundaries:
        return None

    single_level = _single_line_level(boundaries)
    if single_level is not None:
        return single_level

    left = None
    right = None
    for boundary in boundaries:
        if boundary[0] <= scan_x:
            left = boundary
            continue
        right = boundary
        break

    if left is not None and right is not None:
        pair = (str(left[1]), str(right[1]))
        pair_set = {pair[0], pair[1]}
        if pair[0] == "red" and pair[1] == "red":
            return 1
        if pair_set == {"red", "yellow"}:
            return 2
        if pair_set == {"yellow", "orange"}:
            return 3

    # 多线时，橙线外直接不报警
    orange_positions = [bx for bx, color_name in boundaries if color_name == "orange"]
    yellow_positions = [bx for bx, color_name in boundaries if color_name == "yellow"]
    if orange_positions:
        if yellow_positions:
            leftmost_orange = min(orange_positions)
            nearest_yellow_to_left_orange = min(yellow_positions, key=lambda x: abs(x - leftmost_orange))
            outward = leftmost_orange - nearest_yellow_to_left_orange
            if abs(outward) > 1e-6 and (scan_x - leftmost_orange) * outward > 0:
                return None
            rightmost_orange = max(orange_positions)
            nearest_yellow_to_right_orange = min(yellow_positions, key=lambda x: abs(x - rightmost_orange))
            outward = rightmost_orange - nearest_yellow_to_right_orange
            if abs(outward) > 1e-6 and (scan_x - rightmost_orange) * outward > 0:
                return None
        else:
            # 只有橙线但不止一条时，线外也不报警；线间仍按3级处理
            if scan_x < min(orange_positions) or scan_x > max(orange_positions):
                return None
            return 3

    return None


def _classify_from_boundaries_details(scan_x: float, boundaries: List[Tuple[float, str]]) -> Dict[str, Any]:
    details: Dict[str, Any] = {
        "alarm_level": None,
        "reason": "no_boundaries",
        "boundaries": [(float(x), str(color)) for x, color in boundaries],
        "outside_orange": False,
    }
    if not boundaries:
        return details

    single_level = _single_line_level(boundaries)
    if single_level is not None:
        details["alarm_level"] = int(single_level)
        only_color = str(boundaries[0][1]) if boundaries else "unknown"
        details["reason"] = f"single_line:{only_color}"
        return details

    left = None
    right = None
    for boundary in boundaries:
        if boundary[0] <= scan_x:
            left = boundary
            continue
        right = boundary
        break

    if left is not None and right is not None:
        pair = (str(left[1]), str(right[1]))
        pair_set = {pair[0], pair[1]}
        if pair[0] == "red" and pair[1] == "red":
            details["alarm_level"] = 1
            details["reason"] = "band:red_red"
            return details
        if pair_set == {"red", "yellow"}:
            details["alarm_level"] = 2
            details["reason"] = "band:red_yellow"
            return details
        if pair_set == {"yellow", "orange"}:
            details["alarm_level"] = 3
            details["reason"] = "band:yellow_orange"
            return details

    orange_positions = [bx for bx, color_name in boundaries if color_name == "orange"]
    yellow_positions = [bx for bx, color_name in boundaries if color_name == "yellow"]
    if orange_positions:
        if yellow_positions:
            leftmost_orange = min(orange_positions)
            nearest_yellow_to_left_orange = min(yellow_positions, key=lambda x: abs(x - leftmost_orange))
            outward = leftmost_orange - nearest_yellow_to_left_orange
            if abs(outward) > 1e-6 and (scan_x - leftmost_orange) * outward > 0:
                details["reason"] = "outside_orange:left"
                details["outside_orange"] = True
                return details
            rightmost_orange = max(orange_positions)
            nearest_yellow_to_right_orange = min(yellow_positions, key=lambda x: abs(x - rightmost_orange))
            outward = rightmost_orange - nearest_yellow_to_right_orange
            if abs(outward) > 1e-6 and (scan_x - rightmost_orange) * outward > 0:
                details["reason"] = "outside_orange:right"
                details["outside_orange"] = True
                return details
        else:
            if scan_x < min(orange_positions) or scan_x > max(orange_positions):
                details["reason"] = "outside_orange:orange_only"
                details["outside_orange"] = True
                return details
            details["alarm_level"] = 3
            details["reason"] = "band:orange_only"
            return details

    details["reason"] = "unresolved_multi_line"
    return details


def classify_point_alarm_level_uv_details(
    pt: Tuple[float, float],
    projected_curves: Optional[Dict[str, List[Dict[str, Any]]]] = None,
    border_json_path: str = "",
    stream_name: str = "",
) -> Dict[str, Any]:
    details: Dict[str, Any] = {
        "alarm_level": None,
        "reason": "invalid_point",
        "matched_scan_y": None,
        "visible_colors": [],
        "boundaries": [],
        "scan_x": None,
        "scan_y": None,
        "outside_orange": False,
    }
    try:
        scan_x = float(pt[0])
        scan_y = float(pt[1])
    except Exception:
        return details
    details["scan_x"] = scan_x
    details["scan_y"] = scan_y
    if not projected_curves:
        details["reason"] = "no_projected_curves"
        return details

    curve_colors = curve_colors_from_border_file(border_json_path)
    details["visible_colors"] = visible_boundary_colors_from_projected(projected_curves, border_json_path)
    strategy = _resolve_stream_strategy(stream_name)
    strict_scanline_strategies = {"orange_enclosed_level1_strict", "between_orange_x_scanline_level1"}
    use_strict_scanline = strategy in strict_scanline_strategies
    y_offsets = [0.0]
    if not use_strict_scanline:
        for delta in (12.0, 24.0, 40.0, 64.0, 96.0, 140.0, 200.0):
            y_offsets.extend((-delta, delta))

    last_details = None
    for delta in y_offsets:
        boundaries = _collect_boundaries_at_y(
            scan_y + float(delta),
            projected_curves,
            curve_colors,
            extend_endpoints=not use_strict_scanline,
        )
        cur = _classify_stream_specific_details(
            stream_name,
            scan_x,
            scan_y + float(delta),
            boundaries,
            projected_curves,
            curve_colors,
        )
        if cur is None:
            cur = _classify_from_boundaries_details(scan_x, boundaries)
        cur["matched_scan_y"] = scan_y + float(delta)
        last_details = cur
        if cur.get("alarm_level") is not None or cur.get("outside_orange"):
            details.update(cur)
            return details

    if isinstance(last_details, dict):
        details.update(last_details)
    if details.get("reason") == "invalid_point":
        details["reason"] = "unresolved"
    return details


def classify_point_alarm_level_uv(
    pt: Tuple[float, float],
    projected_curves: Optional[Dict[str, List[Dict[str, Any]]]] = None,
    border_json_path: str = "",
    stream_name: str = "",
) -> Optional[int]:
    details = classify_point_alarm_level_uv_details(
        pt,
        projected_curves=projected_curves,
        border_json_path=border_json_path,
        stream_name=stream_name,
    )
    level = details.get("alarm_level")
    return int(level) if level is not None else None
