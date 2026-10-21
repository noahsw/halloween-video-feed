"""Ignore zones: rectangles where detections are dropped (e.g. a skeleton DJ prop
that the model mistakes for a person)."""

import json
from pathlib import Path
from typing import List

import numpy as np


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


class IgnoreZones:
    """Zones are stored normalized (0..1) in camera coordinates, so they survive
    resolution changes."""

    def __init__(self, path: Path, iou_thresh: float = 0.4):
        self.path = path
        self.iou_thresh = iou_thresh
        self.zones: List[List[float]] = []
        if path.exists():
            self.zones = json.loads(path.read_text()).get("zones", [])

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps({"zones": self.zones}, indent=2))

    def add(self, x: int, y: int, w: int, h: int, frame_w: int, frame_h: int) -> None:
        self.zones.append([x / frame_w, y / frame_h, (x + w) / frame_w, (y + h) / frame_h])
        self.save()

    def clear(self) -> None:
        self.zones = []
        self.save()

    def pixel_boxes(self, frame_w: int, frame_h: int) -> List[np.ndarray]:
        return [np.array([x1 * frame_w, y1 * frame_h, x2 * frame_w, y2 * frame_h]) for x1, y1, x2, y2 in self.zones]

    def should_ignore(self, box: np.ndarray, zone_boxes: List[np.ndarray]) -> bool:
        return any(_iou(box, z) >= self.iou_thresh for z in zone_boxes)
