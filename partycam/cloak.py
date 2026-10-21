"""Invisibility cloak: erases people by painting the empty dance floor over them.

The camera never moves, so we keep a "background plate" of what the room looks like
with nobody in it. Pixels are learned into the plate whenever no one is covering them,
so the plate builds itself up as people move around and keeps adapting to the room.

Party lights change the room's color constantly, which would make a stale plate stick
out. So before filling, the plate is relit with a smooth per-region color gain that's
measured on the visible (non-person) parts of the current frame.
"""

import threading
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

GAIN_GRID = 8  # 1920x1080 divides evenly, which keeps INTER_AREA on its fast path


def upscale_region(small: np.ndarray, grid: int, x: int, y: int, w: int, h: int) -> np.ndarray:
    """Upscales just the (x, y, w, h) full-resolution region of a smooth map stored at 1/grid
    scale, instead of the whole frame."""
    sx0, sy0 = x // grid, y // grid
    sx1, sy1 = min(small.shape[1], (x + w) // grid + 2), min(small.shape[0], (y + h) // grid + 2)
    big = cv2.resize(small[sy0:sy1, sx0:sx1], ((sx1 - sx0) * grid, (sy1 - sy0) * grid), interpolation=cv2.INTER_LINEAR)
    big = big[y - sy0 * grid : y - sy0 * grid + h, x - sx0 * grid : x - sx0 * grid + w]
    if big.shape[:2] != (h, w):  # frame size isn't a multiple of the grid at the right/bottom edge
        big = cv2.copyMakeBorder(big, 0, h - big.shape[0], 0, w - big.shape[1], cv2.BORDER_REPLICATE)
    return big


class InvisibilityCloak:
    def __init__(self, plate_path: Optional[Path] = None, learn_rate: float = 0.03):
        self.plate_path = plate_path
        self.learn_rate = learn_rate
        self.plate: Optional[np.ndarray] = None  # float32 HxWx3
        self.known: Optional[np.ndarray] = None  # uint8 HxW, 255 where the plate has seen the empty floor
        self.frame_count = 0

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

    def save(self, background: bool = False) -> None:
        """With background=True the PNG is written on another thread, so the periodic save
        doesn't stall a frame (encoding a 1080p PNG takes ~100 ms)."""
        if self.plate_path is None or self.plate is None or self.coverage < 0.97:
            return
        img, path = cv2.convertScaleAbs(self.plate), self.plate_path

        def write() -> None:
            path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(path), img, [cv2.IMWRITE_PNG_COMPRESSION, 1])

        if background:
            threading.Thread(target=write, daemon=True).start()
        else:
            write()

    def apply(self, frame: np.ndarray, person_mask: np.ndarray, alpha: np.ndarray, feather_px: int = 12) -> np.ndarray:
        """person_mask: uint8, 255 where people are (dilated). alpha: float32 feathered version
        of it (0..1), reaching at most feather_px beyond the mask. Returns the frame with people
        painted over by the empty floor."""
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
        # Learning every other frame at twice the rate is the same speed of adaptation, at half the cost
        self.frame_count += 1
        if self.frame_count % 2 == 0:
            cv2.accumulateWeighted(frame, self.plate, min(1.0, self.learn_rate * 2), mask=free)

        x, y, rw, rh = cv2.boundingRect(person_mask)
        if rw == 0 or rh == 0:
            return frame
        x, y = max(0, x - feather_px), max(0, y - feather_px)
        rw, rh = min(w - x, rw + 2 * feather_px), min(h - y, rh + 2 * feather_px)

        gain = upscale_region(self._gain_map(frame, free), GAIN_GRID, x, y, rw, rh)
        roi = (slice(y, y + rh), slice(x, x + rw))
        fill = cv2.convertScaleAbs(cv2.multiply(self.plate[roi], gain))
        hidden_unknown = cv2.bitwise_and(person_mask[roi], cv2.bitwise_not(self.known[roi]))
        if cv2.countNonZero(hidden_unknown):
            fill = self._inpaint_unknown(fill, hidden_unknown)

        out = frame.copy()
        a = alpha[roi]
        out[roi] = cv2.blendLinear(frame[roi], fill, 1.0 - a, a)
        return out

    def _gain_map(self, frame: np.ndarray, free: np.ndarray) -> np.ndarray:
        """Per-region color gain (current light / plate's light), at 1/GAIN_GRID scale."""
        h, w = frame.shape[:2]
        sw, sh = max(1, w // GAIN_GRID), max(1, h // GAIN_GRID)
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
        return gain.astype(np.float32)

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
