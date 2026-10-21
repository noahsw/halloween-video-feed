"""Per-dancer smoothing so skeletons move like they're really in the room.

Cheap-looking AR is given away by motion more than by drawing, so this handles:
  * jitter          - adaptive smoothing: small movements are damped, big ones pass through
  * popping         - skeletons fade in and out instead of blinking on/off with detection
  * left/right flips- common from behind; detected and undone so limbs don't fly across
  * a sliding skull - the head is smoothed relative to the shoulders, so it stays attached
  * floating feet   - ankles and knees are smoothed harder and ignore sub-pixel wobble
"""

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from .anatomy import CONF_THRESH, LR_PAIRS, Head, estimate_head, shoulder_mid

FADE_IN_FRAMES = 4
FADE_OUT_FRAMES = 6

# Base smoothing per keypoint: lower = steadier. Legs are steadiest so feet stay planted.
BASE_ALPHA = np.full(17, 0.35)
BASE_ALPHA[[13, 14]] = 0.25  # knees
BASE_ALPHA[[15, 16]] = 0.18  # ankles


@dataclass
class Dancer:
    kpts: np.ndarray  # (17, 3): x, y, confidence
    box: np.ndarray  # (4,): x1, y1, x2, y2
    track_id: Optional[int] = None
    opacity: float = 1.0
    head: Optional[Head] = None

    @property
    def height(self) -> float:
        return float(self.box[3] - self.box[1])


@dataclass
class _Track:
    dancer: Dancer
    missing_kpt_frames: np.ndarray = field(default_factory=lambda: np.zeros(17, int))
    head_offset: Optional[np.ndarray] = None  # head center relative to the shoulder midpoint
    head_radius: float = 0.0
    head_roll: float = 0.0
    head_missing: int = 0


def _unflip(prev: np.ndarray, cur: np.ndarray) -> np.ndarray:
    """If the model swapped left and right since last frame, swap them back."""
    direct = swapped = 0.0
    pairs = 0
    for a, b in LR_PAIRS:
        if min(prev[a, 2], prev[b, 2], cur[a, 2], cur[b, 2]) < CONF_THRESH:
            continue
        direct += np.hypot(*(cur[a, :2] - prev[a, :2])) + np.hypot(*(cur[b, :2] - prev[b, :2]))
        swapped += np.hypot(*(cur[a, :2] - prev[b, :2])) + np.hypot(*(cur[b, :2] - prev[a, :2]))
        pairs += 1
    if pairs < 2 or swapped >= 0.6 * direct:
        return cur
    out = cur.copy()
    for a, b in LR_PAIRS:
        out[[a, b]] = cur[[b, a]]
    return out


def _lerp_angle(a: float, b: float, t: float) -> float:
    return a + t * math.atan2(math.sin(b - a), math.cos(b - a))


class PoseSmoother:
    def __init__(self, hold_kpt_frames: int = 6, hold_head_frames: int = 8):
        self.hold_kpt_frames = hold_kpt_frames
        self.hold_head_frames = hold_head_frames
        self.tracks: Dict[int, _Track] = {}

    def update(self, dancers: List[Dancer]) -> List[Dancer]:
        out: List[Dancer] = []
        seen = set()
        for d in dancers:
            if d.track_id is None:
                d.head = estimate_head(d.kpts)
                out.append(d)
                continue
            seen.add(d.track_id)
            track = self.tracks.get(d.track_id)
            if track is None:
                track = _Track(dancer=d)
                d.opacity = 1.0 / FADE_IN_FRAMES
                self.tracks[d.track_id] = track
            else:
                opacity = min(1.0, track.dancer.opacity + 1.0 / FADE_IN_FRAMES)
                track.dancer = self._smooth(track, d)
                track.dancer.opacity = opacity
            self._smooth_head(track)
            out.append(track.dancer)

        # Dancers that weren't detected this frame hold their last pose while fading out
        for tid in list(self.tracks):
            if tid in seen:
                continue
            track = self.tracks[tid]
            track.dancer.opacity -= 1.0 / FADE_OUT_FRAMES
            if track.dancer.opacity <= 1e-6:
                del self.tracks[tid]
            else:
                out.append(track.dancer)
        return out

    def _smooth(self, track: _Track, cur: Dancer) -> Dancer:
        prev = track.dancer.kpts
        new = _unflip(prev, cur.kpts)
        scale = max(cur.height, 20.0)
        dead_zone = 0.006 * scale
        for i in range(len(new)):
            cur_ok = new[i, 2] >= CONF_THRESH
            prev_ok = prev[i, 2] >= CONF_THRESH
            if cur_ok and prev_ok:
                dist = float(np.hypot(*(new[i, :2] - prev[i, :2])))
                alpha = min(1.0, BASE_ALPHA[i] + dist / (0.08 * scale))
                if dist < dead_zone:
                    alpha *= dist / dead_zone  # ignore sub-pixel wobble
                new[i, :2] = prev[i, :2] + alpha * (new[i, :2] - prev[i, :2])
                track.missing_kpt_frames[i] = 0
            elif prev_ok and track.missing_kpt_frames[i] < self.hold_kpt_frames:
                new[i] = prev[i]
                track.missing_kpt_frames[i] += 1
        box = track.dancer.box + 0.6 * (cur.box - track.dancer.box)
        return Dancer(kpts=new, box=box, track_id=cur.track_id)

    def _smooth_head(self, track: _Track) -> None:
        d = track.dancer
        est = estimate_head(d.kpts)
        sh_mid = shoulder_mid(d.kpts)
        if est is None:
            # Keep the last head attached to the shoulders for a moment
            track.head_missing += 1
            if track.head_offset is not None and sh_mid is not None and track.head_missing <= self.hold_head_frames:
                d.head = (sh_mid + track.head_offset, track.head_radius, track.head_roll)
            return
        center, radius, roll = est
        track.head_missing = 0
        if sh_mid is None:
            d.head = est
            return
        offset = center - sh_mid
        if track.head_offset is None:
            track.head_offset, track.head_radius, track.head_roll = offset, radius, roll
        else:
            # The shoulders already move smoothly with the body; only the head's offset from
            # them (which jumps when the estimate switches between eyes/ears/shoulders) is damped.
            track.head_offset = track.head_offset + 0.3 * (offset - track.head_offset)
            track.head_radius += 0.15 * (radius - track.head_radius)
            track.head_roll = _lerp_angle(track.head_roll, roll, 0.3)
        d.head = (sh_mid + track.head_offset, track.head_radius, track.head_roll)
