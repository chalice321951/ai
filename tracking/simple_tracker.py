from dataclasses import dataclass
import threading
from typing import Dict, List, Optional, Tuple


def _iou(box_a: Tuple[int, int, int, int], box_b: Tuple[int, int, int, int]) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    if inter_area <= 0:
        return 0.0

    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    denom = area_a + area_b - inter_area
    if denom <= 0:
        return 0.0
    return float(inter_area / denom)


def _center(box: Tuple[int, int, int, int]) -> Tuple[float, float]:
    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def _area(box: Tuple[int, int, int, int]) -> float:
    x1, y1, x2, y2 = box
    return float(max(0, x2 - x1) * max(0, y2 - y1))


def _center_distance_ratio(box_a: Tuple[int, int, int, int], box_b: Tuple[int, int, int, int]) -> float:
    ax, ay = _center(box_a)
    bx, by = _center(box_b)
    aw = max(1.0, float(box_a[2] - box_a[0]))
    ah = max(1.0, float(box_a[3] - box_a[1]))
    bw = max(1.0, float(box_b[2] - box_b[0]))
    bh = max(1.0, float(box_b[3] - box_b[1]))
    norm = max(1.0, ((aw + bw) / 2.0 + (ah + bh) / 2.0) / 2.0)
    dx = ax - bx
    dy = ay - by
    return float((dx * dx + dy * dy) ** 0.5 / norm)


def _area_ratio(box_a: Tuple[int, int, int, int], box_b: Tuple[int, int, int, int]) -> float:
    area_a = _area(box_a)
    area_b = _area(box_b)
    if area_a <= 0 or area_b <= 0:
        return 0.0
    return float(min(area_a, area_b) / max(area_a, area_b))


@dataclass
class _Track:
    track_id: int
    bbox: Tuple[int, int, int, int]
    class_name: str
    color: Tuple[int, int, int]
    algo_id: str
    confidence: float = 0.0
    missed: int = 0


class SimpleTracker:
    """Per-stream lightweight IoU tracker used after shared detection inference."""

    def __init__(self, max_missed: int = 20, min_iou: float = 0.3):
        self.max_missed = max(1, int(max_missed or 20))
        self.min_iou = float(min_iou or 0.3)
        self._next_id = 1
        self._tracks: Dict[int, _Track] = {}
        self._lock = threading.Lock()

    def reset(self):
        with self._lock:
            self._next_id = 1
            self._tracks.clear()

    def _match_score(self, track: _Track, det: dict) -> Optional[float]:
        det_bbox = tuple(det.get('xyxy', (0, 0, 0, 0)))
        det_class = str(det.get('class_name', '') or '')
        iou = _iou(track.bbox, det_bbox)
        center_ratio = _center_distance_ratio(track.bbox, det_bbox)
        area_ratio = _area_ratio(track.bbox, det_bbox)
        same_class = track.class_name == det_class

        # 远距离小目标在无人机视角下 IoU 很容易掉得很低，这里优先保证同位置目标续用旧 id
        geometry_match = (
            iou >= self.min_iou
            or (center_ratio <= 0.75 and area_ratio >= 0.20)
            or (same_class and center_ratio <= 1.10 and area_ratio >= 0.12)
            or (track.missed <= 1 and center_ratio <= 1.40 and area_ratio >= 0.08)
        )
        if not geometry_match:
            return None

        return float(
            iou * 1.4
            + max(0.0, 1.0 - min(center_ratio, 1.0)) * 0.9
            + area_ratio * 0.5
            + (0.15 if same_class else 0.0)
            - min(float(track.missed), 3.0) * 0.10
        )

    def update(self, detections: List[dict], frame: Optional[object] = None) -> List[Optional[int]]:
        with self._lock:
            assignments: List[Optional[int]] = [None] * len(detections)
            track_ids = list(self._tracks.keys())
            unmatched_tracks = set(track_ids)
            unmatched_detections = set(range(len(detections)))

            candidates = []
            for det_idx, det in enumerate(detections):
                for track_id in track_ids:
                    track = self._tracks[track_id]
                    score = self._match_score(track, det)
                    if score is not None:
                        candidates.append((score, det_idx, track_id))

            candidates.sort(key=lambda item: item[0], reverse=True)
            for _, det_idx, track_id in candidates:
                if det_idx not in unmatched_detections or track_id not in unmatched_tracks:
                    continue
                det_bbox = tuple(detections[det_idx].get('xyxy', (0, 0, 0, 0)))
                det_class = str(detections[det_idx].get('class_name', '') or '')
                det_conf = float(detections[det_idx].get('confidence', 0.0) or 0.0)
                track = self._tracks[track_id]
                track.bbox = det_bbox
                if track.class_name == det_class or det_conf >= max(track.confidence + 0.1, 0.85):
                    track.class_name = det_class or track.class_name
                track.color = tuple(detections[det_idx].get('color', track.color))
                track.algo_id = str(detections[det_idx].get('algo_id', '') or track.algo_id)
                track.confidence = float(detections[det_idx].get('confidence', track.confidence) or 0.0)
                track.missed = 0
                assignments[det_idx] = track_id
                unmatched_detections.remove(det_idx)
                unmatched_tracks.remove(track_id)

            for track_id in list(unmatched_tracks):
                track = self._tracks.get(track_id)
                if track is None:
                    continue
                track.missed += 1
                if track.missed > self.max_missed:
                    self._tracks.pop(track_id, None)

            for det_idx in list(unmatched_detections):
                det = detections[det_idx]
                det_bbox = tuple(det.get('xyxy', (0, 0, 0, 0)))
                det_class = str(det.get('class_name', '') or '')
                track_id = self._next_id
                self._next_id += 1
                self._tracks[track_id] = _Track(
                    track_id=track_id,
                    bbox=det_bbox,
                    class_name=det_class,
                    color=tuple(det.get('color', (0, 255, 0))),
                    algo_id=str(det.get('algo_id', '') or ''),
                    confidence=float(det.get('confidence', 0.0) or 0.0),
                    missed=0,
                )
                assignments[det_idx] = track_id

            return assignments

    def get_active_tracks(self) -> List[dict]:
        with self._lock:
            overlays: List[dict] = []
            for track_id, track in sorted(list(self._tracks.items()), key=lambda item: item[0]):
                if int(track.missed) > 0:
                    continue
                text = f"{track.class_name} {track.confidence:.2f}"
                overlays.append({
                    'xyxy': track.bbox,
                    'text': text,
                    'color': track.color,
                    'class_name': track.class_name,
                    'algo_id': track.algo_id,
                    'confidence': track.confidence,
                    'track_id': track_id,
                })
            return overlays
