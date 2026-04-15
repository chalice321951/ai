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

    def update(self, detections: List[dict], frame: Optional[object] = None) -> List[Optional[int]]:
        with self._lock:
            assignments: List[Optional[int]] = [None] * len(detections)
            track_ids = list(self._tracks.keys())
            unmatched_tracks = set(track_ids)
            unmatched_detections = set(range(len(detections)))

            candidates = []
            for det_idx, det in enumerate(detections):
                det_bbox = tuple(det.get('xyxy', (0, 0, 0, 0)))
                det_class = str(det.get('class_name', '') or '')
                for track_id in track_ids:
                    track = self._tracks[track_id]
                    if track.class_name != det_class:
                        continue
                    score = _iou(track.bbox, det_bbox)
                    if score >= self.min_iou:
                        candidates.append((score, det_idx, track_id))

            candidates.sort(key=lambda item: item[0], reverse=True)
            for _, det_idx, track_id in candidates:
                if det_idx not in unmatched_detections or track_id not in unmatched_tracks:
                    continue
                det_bbox = tuple(detections[det_idx].get('xyxy', (0, 0, 0, 0)))
                track = self._tracks[track_id]
                track.bbox = det_bbox
                track.class_name = str(detections[det_idx].get('class_name', '') or track.class_name)
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
