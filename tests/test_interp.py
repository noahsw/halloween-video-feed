from types import SimpleNamespace

import numpy as np
import pytest

from partycam.interp import Interpolator, blend_dancers
from partycam.tracking import Dancer


def dancer(tid, x, opacity=1.0):
    k = np.zeros((17, 3), np.float32)
    k[:, 0], k[:, 1], k[:, 2] = x, 300, 0.9
    return Dancer(k, np.array([x - 50, 100, x + 50, 600], np.float32), tid, opacity, (np.array([x, 150.0]), 30.0, 0.0))


def test_halfway_pose():
    mid = blend_dancers([dancer(1, 100)], [dancer(1, 200)], 0.5)[0]
    assert mid.kpts[0, 0] == pytest.approx(150)
    assert mid.box[0] == pytest.approx(100)
    assert mid.head[0][0] == pytest.approx(150)


def test_new_dancer_is_drawn_where_it_is():
    out = blend_dancers([dancer(1, 100)], [dancer(1, 200), dancer(2, 900, opacity=0.25)], 0.5)
    new = next(d for d in out if d.track_id == 2)
    assert new.kpts[0, 0] == 900 and new.opacity == 0.25


def test_hidden_keypoints_are_not_blended():
    prev, cur = dancer(1, 100), dancer(1, 200)
    prev.kpts[3, 2] = 0.0  # not detected last frame
    mid = blend_dancers([prev], [cur], 0.5)[0]
    assert mid.kpts[3, 0] == 200


def test_at_t1_returns_current_poses_unchanged():
    cur = [dancer(1, 200)]
    assert blend_dancers([dancer(1, 100)], cur, 1.0) is cur


def test_display_runs_one_frame_behind_and_never_extrapolates():
    it = Interpolator()
    it.push(SimpleNamespace(t=10.00, dancers=[dancer(1, 100)]), ready_at=10.05)
    it.push(SimpleNamespace(t=10.05, dancers=[dancer(1, 200)]), ready_at=10.10)
    x_at = lambda now: it.pose_at(now)[1][0].kpts[0, 0]
    assert x_at(10.100) == pytest.approx(100, abs=1)  # newest pose just arrived: start from the previous one
    assert x_at(10.125) == pytest.approx(150, abs=1)  # glide toward it
    assert x_at(10.150) == pytest.approx(200, abs=1)  # reach it when the next frame is due
    assert x_at(10.300) == pytest.approx(200, abs=1)  # next frame late: hold, don't guess ahead


def test_first_frame_is_shown_as_is():
    it = Interpolator()
    assert it.pose_at(1.0) == (None, [])
    a = SimpleNamespace(t=1.0, dancers=[dancer(1, 100)])
    it.push(a, ready_at=1.05)
    bg, dancers = it.pose_at(1.1)
    assert bg is a and dancers is a.dancers
