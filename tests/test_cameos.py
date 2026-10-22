import random

import numpy as np
import pytest

from partycam.cameos import BATS, DOG, Cameos, Dog, FloorScale, Flock
from partycam.skeleton import PALETTE_NAMES, SkeletonCanvas
from partycam.tracking import Dancer

W, H = 1920, 1080


def dancer(foot_x: float, foot_y: float, height: float) -> Dancer:
    kpts = np.zeros((17, 3), np.float32)
    kpts[15:17] = (foot_x, foot_y, 0.9)  # both ankles seen
    return Dancer(kpts=kpts, box=np.array([foot_x - 60, foot_y - height, foot_x + 60, foot_y], np.float32))


def run(dog: Dog, seconds: float, dt: float = 1 / 30):
    for _ in range(int(seconds / dt)):
        dog.update(dt)
        yield dog


def test_floor_scale_learns_perspective_and_floor_extent():
    floor = FloorScale()
    rng = random.Random(0)
    for _ in range(200):
        y = rng.uniform(500, 1000)
        floor.observe([dancer(rng.uniform(700, 1300), y, 0.8 * y - 50)])
    assert floor.person_height(800) == pytest.approx(590, rel=0.02)
    lo, hi = floor.floor_span(800, W, H)
    assert 690 < lo < 760 and 1240 < hi < 1310
    assert all(500 <= floor.floor_y(rng, H) <= 1000 for _ in range(50))


def test_floor_scale_ignores_dancers_without_feet():
    floor = FloorScale()
    d = dancer(900, 800, 500)
    d.kpts[15:17, 2] = 0.1
    floor.observe([d] * 100)
    assert not floor.learned
    assert floor.person_height(800) == pytest.approx(FloorScale.FALLBACK * 800)


def test_planted_paws_dont_slide_while_trotting():
    dog = Dog(random.Random(1), W, H, FloorScale())
    planted = []
    for d in run(dog, 0.6):
        p = d.phase
        if p < Dog.STANCE - 0.05 and d.state == "in" and d.m == 1.0:  # near front paw is on the floor
            planted.append(float(d.world(d.paw(Dog.FRONT, p))[0]))
        elif planted:
            break
    assert len(planted) > 3
    assert np.ptp(planted) < 1.0


def test_dog_crosses_the_floor_stops_once_and_fades_in_and_out():
    floor = FloorScale()
    dog = Dog(random.Random(2), W, H, floor)
    lo, hi = floor.floor_span(dog.gy, W, H)
    assert dog.opacity == 0.0
    states, xs, opacities = [], [], []
    for d in run(dog, 15):
        states.append(d.state)
        xs.append(d.x)
        opacities.append(d.opacity)
        if d.done:
            break
    assert dog.done
    assert [s for i, s in enumerate(states) if i == 0 or s != states[i - 1]] == ["in", "brake", "pause", "out"]
    assert states.count("pause") / 30 == pytest.approx(Dog.PAUSE, abs=0.1)
    step = dog.speed / 30  # it can overshoot the end by the last frame's step, already invisible
    assert min(xs) >= lo - step and max(xs) <= hi + step  # never leaves the floor
    assert max(opacities) == 1.0 and opacities[-1] < 0.05


@pytest.mark.parametrize("palette", PALETTE_NAMES)
def test_dog_draws_in_every_style(palette):
    dog = Dog(random.Random(3), W, H, FloorScale())
    for d in run(dog, 1.5):
        pass
    canvas = SkeletonCanvas((H, W, 3))
    dog.draw(canvas, palette)
    ys, xs = np.nonzero(canvas.alpha)
    assert len(xs) > 0
    assert abs(np.median(xs) - dog.x) < dog.S and ys.max() <= dog.gy + 0.1 * dog.S


def test_bats_only_darken_and_fly_off():
    rng = random.Random(4)
    frame = np.full((H, W, 3), 180, np.uint8)
    flock = Flock(rng, W, H, start=0.0)
    drew = False
    t = 0.0
    while not flock.done(t):
        out = frame.copy()
        flock.draw(out, t)
        assert out.max() <= 180
        drew |= bool((out < 180).any())
        t += 1 / 30
        assert t < 8
    assert drew


def test_cameos_wait_for_quiet_and_never_repeat_the_dog():
    cameos = Cameos(every=10.0, seed=5)
    t, kinds, waited = 0.0, [], False
    while len(kinds) < 12:
        quiet = int(t) % 20 >= 8  # a banner up for 8 s of every 20
        before = cameos.active
        cameos.update(t, (H, W, 3), [], quiet=quiet)
        if cameos.active and not before:
            assert quiet
            kinds.append(DOG if cameos.dog else BATS)
        waited |= not quiet and not cameos.active and t >= cameos.next_at
        t += 1 / 30
    assert waited
    assert DOG in kinds and BATS in kinds
    assert all(not (a == b == DOG) for a, b in zip(kinds, kinds[1:]))


def test_trigger_runs_now_even_when_cameos_are_off():
    cameos = Cameos(enabled=False, seed=6)
    for i in range(30 * 200):
        cameos.update(i / 30, (H, W, 3), [])
    assert not cameos.active
    cameos.trigger(DOG)
    cameos.update(200.0, (H, W, 3), [], quiet=False)
    assert cameos.dog is not None
