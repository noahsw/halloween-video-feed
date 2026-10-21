"""Smooth skeleton motion: draw at 30 fps from a 20 fps camera.

With people erased, the room behind the skeletons barely changes, and a skeleton is just a
set of joint positions. So between camera frames we can draw in-between poses: each dancer's
joints are blended between their last two detected poses (matched by track id).

The display runs about one camera frame behind real time, so it is always blending between
two poses it already has, never guessing ahead. Timing follows when frames actually arrived,
so even when drawing can't keep up with 30 fps, motion stays evenly paced instead of jerky.
"""

import math
from typing import List, Optional, Tuple

import numpy as np

from .anatomy import CONF_THRESH
from .tracking import Dancer


def _lerp_head(a, b, t: float):
    if a is None or b is None:
        return b if b is not None else a
    (ca, ra, rolla), (cb, rb, rollb) = a, b
    roll = rolla + t * math.atan2(math.sin(rollb - rolla), math.cos(rollb - rolla))
    return ca + t * (cb - ca), ra + t * (rb - ra), roll


def blend_dancers(prev: List[Dancer], cur: List[Dancer], t: float) -> List[Dancer]:
    """Poses a fraction t of the way from prev to cur (0 = prev, 1 = cur)."""
    if t >= 1.0:
        return cur
    by_id = {d.track_id: d for d in prev if d.track_id is not None}
    out = []
    for d in cur:
        p = by_id.get(d.track_id) if d.track_id is not None else None
        if p is None:  # just appeared: it's fading in already, so just draw it where it is
            out.append(d)
            continue
        kpts = d.kpts.copy()
        both = (p.kpts[:, 2] >= CONF_THRESH) & (d.kpts[:, 2] >= CONF_THRESH)
        kpts[both, :2] = p.kpts[both, :2] + t * (d.kpts[both, :2] - p.kpts[both, :2])
        out.append(
            Dancer(
                kpts=kpts,
                box=p.box + t * (d.box - p.box),
                track_id=d.track_id,
                opacity=p.opacity + t * (d.opacity - p.opacity),
                head=_lerp_head(p.head, d.head, t),
            )
        )
    return out


class Interpolator:
    """Holds the last two analyzed camera frames and answers "what do the dancers look like now?"."""

    def __init__(self) -> None:
        self.prev = None
        self.cur = None
        self.latency = 0.05  # camera frame arrival -> analysis ready, smoothed
        self.interval = 0.05  # time between camera frames, smoothed

    def push(self, analysis, ready_at: float) -> None:
        if self.cur is not None:
            gap = analysis.t - self.cur.t
            if 0.01 < gap < 0.25:  # ignore reconnects and the like
                self.interval += 0.1 * (gap - self.interval)
        self.latency += 0.1 * (min(0.3, max(0.0, ready_at - analysis.t)) - self.latency)
        self.prev, self.cur = self.cur, analysis

    def pose_at(self, now: float) -> Tuple[Optional[object], List[Dancer]]:
        """(analysis whose background to draw on, dancers interpolated for `now`)."""
        if self.cur is None:
            return None, []
        if self.prev is None:
            return self.cur, self.cur.dancers
        # One frame interval behind the newest analysis, so `target` lands between prev and cur
        target = now - self.latency - self.interval
        span = self.cur.t - self.prev.t
        t = 1.0 if span <= 1e-3 else float(np.clip((target - self.prev.t) / span, 0.0, 1.0))
        return self.cur, blend_dancers(self.prev.dancers, self.cur.dancers, t)
