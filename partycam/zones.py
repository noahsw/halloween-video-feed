"""Ignore zones: rectangles where detections are dropped (e.g. a skeleton DJ prop
that the model mistakes for a person)."""

import json
from pathlib import Path
from typing import List

import numpy as np


def _area(b: np.ndarray) -> float:
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])


def _intersection(a: np.ndarray, b: np.ndarray) -> float:
    return _area(np.array([max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])]))


class IgnoreZones:
    """Zones are stored normalized (0..1) in camera coordinates, so they survive
    resolution changes."""

    def __init__(self, path: Path, inside_thresh: float = 0.7, min_size: float = 0.25):
        self.path = path
        self.inside_thresh = inside_thresh
        self.min_size = min_size
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
        return in_any_zone(box, zone_boxes, self.inside_thresh, self.min_size)


def in_any_zone(box: np.ndarray, zone_boxes: List[np.ndarray], inside_thresh: float = 0.7, min_size: float = 0.25) -> bool:
    """True if the detection sits mostly inside a zone and is at least a decent fraction of
    its size. The model sometimes boxes only part of a prop (e.g. its upper body), while a real
    person walking in front of the prop is closer to the camera, so they look bigger and spill
    out of the zone."""
    area = _area(box)
    if area <= 0:
        return False
    return any(_intersection(box, z) / area >= inside_thresh and area >= min_size * _area(z) for z in zone_boxes)
