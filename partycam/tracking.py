"""Per-dancer keypoint smoothing so skeletons don't jitter."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

CONF_THRESH = 0.30


@dataclass
class Dancer:
    kpts: np.ndarray  # (17, 3): x, y, confidence
    box: np.ndarray  # (4,): x1, y1, x2, y2
    track_id: Optional[int] = None

    @property
    def height(self) -> float:
        return float(self.box[3] - self.box[1])


@dataclass
class _Track:
    dancer: Dancer
    missing_kpt_frames: np.ndarray = field(default_factory=lambda: np.zeros(17, int))
    unseen_frames: int = 0


class PoseSmoother:
    """Adaptive exponential smoothing per tracked dancer.

    Small movements get heavily smoothed (kills jitter), big movements pass through
    almost directly (keeps up with fast dance moves). Keypoints that briefly drop
    out, which happens a lot when people face away from the camera, are held in place
    for a few frames instead of blinking off.
    """

    def __init__(self, hold_kpt_frames: int = 6, hold_dancer_frames: int = 3):
        self.hold_kpt_frames = hold_kpt_frames
        self.hold_dancer_frames = hold_dancer_frames
        self.tracks: Dict[int, _Track] = {}

    def update(self, dancers: List[Dancer]) -> List[Dancer]:
        out: List[Dancer] = []
        seen = set()
        for d in dancers:
            if d.track_id is None:
                out.append(d)
                continue
            seen.add(d.track_id)
            track = self.tracks.get(d.track_id)
            if track is None:
                self.tracks[d.track_id] = _Track(dancer=d)
                out.append(d)
                continue
            track.dancer = self._smooth(track, d)
            track.unseen_frames = 0
            out.append(track.dancer)

        for tid in list(self.tracks):
            if tid in seen:
                continue
            track = self.tracks[tid]
            track.unseen_frames += 1
            if track.unseen_frames > self.hold_dancer_frames:
                del self.tracks[tid]
            else:
                out.append(track.dancer)
        return out

    def _smooth(self, track: _Track, cur: Dancer) -> Dancer:
        prev = track.dancer.kpts
        new = cur.kpts.copy()
        scale = max(cur.height, 20.0)
        for i in range(len(new)):
            cur_ok = new[i, 2] >= CONF_THRESH
            prev_ok = prev[i, 2] >= CONF_THRESH
            if cur_ok and prev_ok:
                dist = float(np.hypot(*(new[i, :2] - prev[i, :2])))
                alpha = min(1.0, 0.35 + dist / (0.08 * scale))
                new[i, :2] = prev[i, :2] + alpha * (new[i, :2] - prev[i, :2])
                track.missing_kpt_frames[i] = 0
            elif prev_ok and track.missing_kpt_frames[i] < self.hold_kpt_frames:
                new[i] = prev[i]
                track.missing_kpt_frames[i] += 1
        box = track.dancer.box + 0.6 * (cur.box - track.dancer.box)
        return Dancer(kpts=new, box=box, track_id=cur.track_id)
