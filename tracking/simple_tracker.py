from dataclasses import dataclass
import threading
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np


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
        self._prev_gray: Optional[np.ndarray] = None
        self._lock = threading.Lock()

    def reset(self):
        with self._lock:
            self._next_id = 1
            self._tracks.clear()
            self._prev_gray = None

    def set_reference_frame(self, frame: Optional[np.ndarray]):
        with self._lock:
            if frame is None:
                self._prev_gray = None
                return
            self._prev_gray = self._to_gray(frame)

    def predict(self, frame: Optional[np.ndarray]):
        with self._lock:
            gray = self._to_gray(frame)
            if gray is None:
                return
            if self._prev_gray is None:
                self._prev_gray = gray
                return

            height, width = gray.shape[:2]
            for track in list(self._tracks.values()):
                predicted = self._predict_bbox(track.bbox, self._prev_gray, gray, width, height)
                if predicted is not None:
                    track.bbox = predicted
                else:
                    track.missed += 1

            stale_track_ids = [track_id for track_id, track in self._tracks.items() if track.missed > self.max_missed]
            for track_id in stale_track_ids:
                self._tracks.pop(track_id, None)

            self._prev_gray = gray

    def update(self, detections: List[dict], frame: Optional[np.ndarray] = None) -> List[Optional[int]]:
        with self._lock:
            gray = self._to_gray(frame)
            if gray is not None and self._prev_gray is None:
                self._prev_gray = gray

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

            if gray is not None:
                self._prev_gray = gray

            return assignments

    def get_active_tracks(self) -> List[dict]:
        with self._lock:
            overlays: List[dict] = []
            for track_id, track in sorted(list(self._tracks.items()), key=lambda item: item[0]):
                text = f"{track.algo_id}:ID{track_id} {track.class_name} {track.confidence:.2f}"
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

    def _to_gray(self, frame: Optional[np.ndarray]) -> Optional[np.ndarray]:
        if frame is None:
            return None
        try:
            return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        except Exception:
            return None

    def _predict_bbox(
        self,
        bbox: Tuple[int, int, int, int],
        prev_gray: np.ndarray,
        gray: np.ndarray,
        width: int,
        height: int,
    ) -> Optional[Tuple[int, int, int, int]]:
        x1, y1, x2, y2 = map(int, bbox)
        if x2 <= x1 or y2 <= y1:
            return None

        points = np.array([
            [x1, y1],
            [x2, y1],
            [x1, y2],
            [x2, y2],
            [(x1 + x2) / 2.0, (y1 + y2) / 2.0],
            [(x1 + x2) / 2.0, y1],
            [(x1 + x2) / 2.0, y2],
            [x1, (y1 + y2) / 2.0],
            [x2, (y1 + y2) / 2.0],
        ], dtype=np.float32).reshape(-1, 1, 2)

        try:
            next_points, status, _ = cv2.calcOpticalFlowPyrLK(
                prev_gray,
                gray,
                points,
                None,
                winSize=(21, 21),
                maxLevel=2,
                criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03),
            )
        except Exception:
            return None

        if next_points is None or status is None:
            return None

        valid = status.reshape(-1) == 1
        if int(valid.sum()) < 3:
            return None

        deltas = next_points.reshape(-1, 2)[valid] - points.reshape(-1, 2)[valid]
        dx, dy = np.median(deltas, axis=0)
        dx = int(round(float(dx)))
        dy = int(round(float(dy)))
        if dx == 0 and dy == 0:
            return bbox

        box_w = x2 - x1
        box_h = y2 - y1
        nx1 = min(max(0, x1 + dx), max(0, width - 1))
        ny1 = min(max(0, y1 + dy), max(0, height - 1))
        nx2 = min(width, max(nx1 + 1, nx1 + box_w))
        ny2 = min(height, max(ny1 + 1, ny1 + box_h))
        return (int(nx1), int(ny1), int(nx2), int(ny2))
