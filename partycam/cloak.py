"""Invisibility cloak: erases people by painting the empty dance floor over them.

The camera never moves, so we keep a "background plate" of what the room looks like
with nobody in it. Pixels are learned into the plate whenever no one is covering them,
so the plate builds itself up as people move around and keeps adapting to the room.

Party lights change the room's color constantly, which would make a stale plate stick
out. So before filling, the plate is relit with a smooth per-region color gain that's
measured on the visible (non-person) parts of the current frame.
"""

from pathlib import Path
from typing import Optional

import cv2
import numpy as np


class InvisibilityCloak:
    def __init__(self, plate_path: Optional[Path] = None, learn_rate: float = 0.03):
        self.plate_path = plate_path
        self.learn_rate = learn_rate
        self.plate: Optional[np.ndarray] = None  # float32 HxWx3
        self.known: Optional[np.ndarray] = None  # uint8 HxW, 255 where the plate has seen the empty floor

    @property
    def coverage(self) -> float:
        return 0.0 if self.known is None else float(np.count_nonzero(self.known)) / self.known.size

    def reset(self) -> None:
        self.plate = None
        self.known = None

    def load(self, shape) -> bool:
        if self.plate_path is None or not self.plate_path.exists():
            return False
        img = cv2.imread(str(self.plate_path))
        if img is None:
            return False
        if img.shape[:2] != shape[:2]:
            img = cv2.resize(img, (shape[1], shape[0]), interpolation=cv2.INTER_AREA)
        self.plate = img.astype(np.float32)
        self.known = np.full(shape[:2], 255, np.uint8)
        return True

    def save(self) -> None:
        if self.plate_path is None or self.plate is None or self.coverage < 0.97:
            return
        self.plate_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(self.plate_path), np.clip(self.plate, 0, 255).astype(np.uint8))

    def apply(self, frame: np.ndarray, person_mask: np.ndarray, alpha: np.ndarray) -> np.ndarray:
        """person_mask: uint8, 255 where people are (dilated). alpha: float32 feathered version
        of it (0..1). Returns the frame with people painted over by the empty floor."""
        h, w = frame.shape[:2]
        if self.plate is None or self.plate.shape[:2] != (h, w):
            if not self.load(frame.shape):
                self.plate = frame.astype(np.float32)
                self.known = cv2.bitwise_not(person_mask)

        free = cv2.bitwise_not(person_mask)
        newly_seen = cv2.bitwise_and(free, cv2.bitwise_not(self.known))
        if cv2.countNonZero(newly_seen):
            cv2.copyTo(frame.astype(np.float32), newly_seen, self.plate)
            cv2.bitwise_or(self.known, newly_seen, self.known)
        cv2.accumulateWeighted(frame, self.plate, self.learn_rate, mask=free)

        x, y, rw, rh = cv2.boundingRect(cv2.max(person_mask, (alpha > 0.004).astype(np.uint8)))
        if rw == 0 or rh == 0:
            return frame

        gain = self._gain_map(frame, free)
        roi = (slice(y, y + rh), slice(x, x + rw))
        fill = cv2.convertScaleAbs(cv2.multiply(self.plate[roi], gain[roi]))
        hidden_unknown = cv2.bitwise_and(person_mask[roi], cv2.bitwise_not(self.known[roi]))
        if cv2.countNonZero(hidden_unknown):
            fill = self._inpaint_unknown(fill, hidden_unknown)

        out = frame.copy()
        a = alpha[roi]
        out[roi] = cv2.blendLinear(frame[roi], fill, 1.0 - a, a)
        return out

    def _gain_map(self, frame: np.ndarray, free: np.ndarray) -> np.ndarray:
        h, w = frame.shape[:2]
        sw, sh = max(16, w // 8), max(9, h // 8)
        f_s = cv2.resize(frame, (sw, sh), interpolation=cv2.INTER_AREA).astype(np.float32)
        p_s = cv2.resize(self.plate, (sw, sh), interpolation=cv2.INTER_AREA)
        valid = cv2.bitwise_and(free, self.known)
        m_s = (cv2.resize(valid, (sw, sh), interpolation=cv2.INTER_AREA) > 250).astype(np.float32)

        ratio = np.clip((f_s + 6.0) / (p_s + 6.0), 0.25, 4.0)
        m3 = m_s[..., None]
        total = float(m_s.sum())
        global_gain = (ratio * m3).sum(axis=(0, 1)) / total if total > 0 else np.ones(3, np.float32)

        sigma = sw / 20.0
        num = cv2.GaussianBlur(ratio * m3, (0, 0), sigma)
        den = cv2.GaussianBlur(m_s, (0, 0), sigma)[..., None]
        # Where no nearby pixels are visible, fall back to the frame-wide gain
        prior = 0.05
        gain = (num + global_gain * prior) / (den + prior)
        return cv2.resize(gain.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)

    def _inpaint_unknown(self, fill: np.ndarray, hidden_unknown: np.ndarray) -> np.ndarray:
        """Floor the plate has never seen (someone has been standing there since startup)."""
        h, w = fill.shape[:2]
        sw, sh = max(16, w // 6), max(9, h // 6)
        small = cv2.resize(fill, (sw, sh), interpolation=cv2.INTER_AREA)
        small_mask = cv2.resize(hidden_unknown, (sw, sh), interpolation=cv2.INTER_NEAREST)
        painted = cv2.inpaint(small, small_mask, 5, cv2.INPAINT_TELEA)
        painted = cv2.resize(painted, (w, h), interpolation=cv2.INTER_LINEAR)
        cv2.copyTo(painted, hidden_unknown, fill)
        return fill
