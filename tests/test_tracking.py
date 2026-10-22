import numpy as np

from partycam.anatomy import L_AN, L_SH, LR_PAIRS, R_SH
from partycam.tracking import FADE_IN_FRAMES, FADE_OUT_FRAMES, Dancer, PoseSmoother

# A person ~560 px tall, seen from behind (ears but no face)
POSE = {5: (450, 300), 6: (550, 300), 7: (430, 400), 8: (570, 400), 9: (420, 480), 10: (580, 480),
        11: (465, 500), 12: (535, 500), 13: (460, 620), 14: (540, 620), 15: (455, 740), 16: (545, 740),
        3: (480, 230), 4: (520, 230)}  # fmt: skip
BOX = np.array([400, 200, 600, 760], np.float32)


def pose(dx=0.0, jitter=0.0, rng=None):
    k = np.zeros((17, 3), np.float32)
    for i, (x, y) in POSE.items():
        k[i] = (x + dx, y, 0.9)
    if jitter:
        k[:, :2] += rng.normal(0, jitter, (17, 2)) * (k[:, 2:] > 0)
    return k


def step(smoother, kpts, tid=1, box=BOX):
    return smoother.update([Dancer(kpts, box.copy(), track_id=tid)] if kpts is not None else [])


def test_new_dancer_fades_in():
    s = PoseSmoother()
    opacities = [step(s, pose())[0].opacity for _ in range(FADE_IN_FRAMES + 2)]
    assert opacities[0] == 1.0 / FADE_IN_FRAMES
    assert opacities[FADE_IN_FRAMES - 1] == 1.0
    assert all(a <= b for a, b in zip(opacities, opacities[1:]))


def test_lost_dancer_holds_pose_while_fading_out():
    s = PoseSmoother()
    for _ in range(FADE_IN_FRAMES):
        last = step(s, pose())[0]
    # The smoother updates a fading dancer in place (the app snapshots each frame's dancers),
    # so record what each frame looked like as it goes
    frames = []
    for _ in range(FADE_OUT_FRAMES):
        out = step(s, None)
        frames.append([(d.opacity, d.kpts[L_SH, 0]) for d in out])
    # Opacity drops 1/FADE_OUT_FRAMES per frame; it's drawn while above zero, then removed
    assert all(len(f) == 1 for f in frames[:-1])
    assert frames[0][0][1] == last.kpts[L_SH, 0]  # in its last pose
    opacities = [f[0][0] for f in frames[:-1]]
    assert all(a > b for a, b in zip(opacities, opacities[1:]))
    assert frames[-1] == []


def test_left_right_swap_is_undone():
    s = PoseSmoother()
    for _ in range(5):
        step(s, pose())
    swapped = pose()
    for a, b in LR_PAIRS:
        swapped[[a, b]] = swapped[[b, a]]
    out = step(s, swapped)[0]
    assert abs(out.kpts[L_SH, 0] - 450) < 5 and abs(out.kpts[R_SH, 0] - 550) < 5


def test_jitter_is_smoothed():
    rng = np.random.default_rng(0)
    s = PoseSmoother()
    ankles = []
    for i in range(40):
        out = step(s, pose(jitter=2.0, rng=rng))[0]
        if i >= 10:
            ankles.append(out.kpts[L_AN, :2].copy())
    assert np.std(ankles, axis=0).mean() < 1.0  # from 2 px of input jitter


def test_big_moves_pass_through_quickly():
    s = PoseSmoother()
    for _ in range(5):
        step(s, pose())
    out = step(s, pose(dx=120))[0]  # a fast dance move
    assert out.kpts[L_SH, 0] > 450 + 100


def test_briefly_missing_keypoint_is_held():
    s = PoseSmoother()
    for _ in range(5):
        step(s, pose())
    k = pose()
    k[L_AN, 2] = 0.05  # ankle hidden behind someone this frame
    out = step(s, k)[0]
    assert out.kpts[L_AN, 2] >= 0.3 and abs(out.kpts[L_AN, 0] - 455) < 1


def test_skull_stays_attached_when_the_body_moves():
    s = PoseSmoother()
    for _ in range(5):
        before = step(s, pose())[0]
    after = step(s, pose(dx=120))[0]
    head_dx = after.head[0][0] - before.head[0][0]
    shoulder_dx = (after.kpts[L_SH, 0] + after.kpts[R_SH, 0]) / 2 - (before.kpts[L_SH, 0] + before.kpts[R_SH, 0]) / 2
    assert abs(head_dx - shoulder_dx) < 2  # no lag behind the body


def test_untracked_dancers_pass_straight_through():
    s = PoseSmoother()
    out = s.update([Dancer(pose(), BOX.copy(), track_id=None)])
    assert len(out) == 1 and out[0].opacity == 1.0 and out[0].head is not None
