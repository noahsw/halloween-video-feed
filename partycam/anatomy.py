"""COCO keypoint indices and geometry helpers shared by tracking and drawing."""

import math
from typing import Optional, Tuple

import numpy as np

CONF_THRESH = 0.30

NOSE, L_EYE, R_EYE, L_EAR, R_EAR = 0, 1, 2, 3, 4
L_SH, R_SH, L_EL, R_EL, L_WR, R_WR = 5, 6, 7, 8, 9, 10
L_HIP, R_HIP, L_KN, R_KN, L_AN, R_AN = 11, 12, 13, 14, 15, 16

# Left/right keypoint pairs, for un-flipping when the model swaps sides
LR_PAIRS = [(L_EYE, R_EYE), (L_EAR, R_EAR), (L_SH, R_SH), (L_EL, R_EL), (L_WR, R_WR), (L_HIP, R_HIP), (L_KN, R_KN), (L_AN, R_AN)]

Head = Tuple[np.ndarray, float, float]  # center, radius, roll (radians)

# Shoulders are roughly as wide as the torso is long. Side-on, the two shoulder keypoints
# nearly overlap, so the body is never drawn narrower than this (a stick isn't a skeleton).
MIN_WIDTH_TO_TORSO = 0.55


def kpt(kpts: np.ndarray, idx: int) -> Optional[np.ndarray]:
    """Returns the keypoint as an (x, y) array, or None if it isn't confidently detected."""
    x, y, c = kpts[idx]
    if c < CONF_THRESH or x <= 1 or y <= 1:
        return None
    return np.array([x, y], np.float32)


def unit(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-6 else np.array([0.0, -1.0], np.float32)


def shoulder_mid(kpts: np.ndarray) -> Optional[np.ndarray]:
    l_sh, r_sh = kpt(kpts, L_SH), kpt(kpts, R_SH)
    return (l_sh + r_sh) * 0.5 if l_sh is not None and r_sh is not None else None


def body_width(kpts: np.ndarray) -> float:
    """Shoulder width, floored by torso length so side-on dancers keep a full-size body."""
    l_sh, r_sh = kpt(kpts, L_SH), kpt(kpts, R_SH)
    if l_sh is None or r_sh is None:
        return 0.0
    width = float(np.linalg.norm(l_sh - r_sh))
    l_hip, r_hip = kpt(kpts, L_HIP), kpt(kpts, R_HIP)
    if l_hip is not None and r_hip is not None:
        torso_len = float(np.linalg.norm((l_sh + r_sh) * 0.5 - (l_hip + r_hip) * 0.5))
        width = max(width, MIN_WIDTH_TO_TORSO * torso_len)
    return width


def estimate_head(kpts: np.ndarray) -> Optional[Head]:
    """Returns (center, radius, roll_radians) for the skull.

    The camera mostly sees people from behind, where the eyes and nose are hidden,
    so this falls back to the ears and finally to the shoulders.
    """
    l_hip, r_hip = kpt(kpts, L_HIP), kpt(kpts, R_HIP)
    sh_mid = shoulder_mid(kpts)
    sh_w = body_width(kpts)

    up = np.array([0.0, -1.0], np.float32)
    if sh_mid is not None and l_hip is not None and r_hip is not None:
        up = unit(sh_mid - (l_hip + r_hip) * 0.5)
    roll = math.atan2(up[0], -up[1])

    l_eye, r_eye, nose = kpt(kpts, L_EYE), kpt(kpts, R_EYE), kpt(kpts, NOSE)
    l_ear, r_ear = kpt(kpts, L_EAR), kpt(kpts, R_EAR)

    if l_eye is not None and r_eye is not None:
        center = (l_eye + r_eye) * 0.5
        eye_d = float(np.linalg.norm(l_eye - r_eye))
        radius = max(eye_d * 1.5, sh_w * 0.32)
        roll = math.atan2(*(l_eye - r_eye)[::-1])
        if math.cos(roll) < 0:
            roll += math.pi
    elif l_ear is not None and r_ear is not None:
        ear_d = float(np.linalg.norm(l_ear - r_ear))
        center = (l_ear + r_ear) * 0.5 + up * ear_d * 0.1
        radius = max(ear_d * 0.62, sh_w * 0.3)
    elif nose is not None:
        center = nose
        radius = sh_w * 0.32
    elif sh_mid is not None:
        center = sh_mid + up * sh_w * 0.62
        radius = sh_w * 0.3
    else:
        return None
    if radius < 6:
        return None
    return center, radius, roll
