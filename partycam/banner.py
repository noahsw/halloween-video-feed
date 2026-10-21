"""Rotating "come join the dance floor" banner."""

import math
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

MESSAGES = [
    "NO BODY DANCES ALONE - JOIN THE SKELETON CREW!",
    "THE DEAD ARE DANCING. WHAT'S YOUR EXCUSE?",
    "SHAKE YOUR BONES ON THE DANCE FLOOR!",
    "DON'T BE A DEADBEAT - COME DANCE!",
    "THE DJ IS DEAD. THE PARTY ISN'T.",
    "GET IN HERE BEFORE YOU GHOST US",
    "BONE-AFIDE DANCE PARTY IN PROGRESS",
    "RATTLE THOSE BONES - GET ON THE FLOOR!",
]

FONT_CANDIDATES = [
    "/System/Library/Fonts/Supplemental/Chalkduster.ttf",
    "/System/Library/Fonts/Supplemental/Impact.ttf",
    "/Library/Fonts/Arial Bold.ttf",
]

TEXT_RGB = (236, 255, 220)
GLOW_RGB = (120, 255, 60)


def _find_font() -> Optional[str]:
    return next((p for p in FONT_CANDIDATES if Path(p).exists()), None)


class JoinBanner:
    def __init__(self, seconds_per_message: float = 8.0):
        self.seconds = seconds_per_message
        self.font_path = _find_font()
        self._cache: Dict[Tuple[int, int], Tuple[np.ndarray, np.ndarray]] = {}

    def _render(self, idx: int, frame_w: int) -> Tuple[np.ndarray, np.ndarray]:
        """Returns (premultiplied BGR float32, alpha float32) for message idx."""
        key = (idx, frame_w)
        if key in self._cache:
            return self._cache[key]
        text = MESSAGES[idx]
        size = max(18, frame_w // 30)
        font = ImageFont.truetype(self.font_path, size) if self.font_path else ImageFont.load_default(size)
        while size > 14:
            x0, y0, x1, y1 = font.getbbox(text)
            if x1 - x0 <= frame_w * 0.9:
                break
            size = int(size * 0.92)
            font = ImageFont.truetype(self.font_path, size) if self.font_path else ImageFont.load_default(size)

        pad = size
        x0, y0, x1, y1 = font.getbbox(text)
        img_w, img_h = x1 - x0 + pad * 2, y1 - y0 + pad * 2
        origin = (pad - x0, pad - y0)

        glow = Image.new("L", (img_w, img_h), 0)
        ImageDraw.Draw(glow).text(origin, text, font=font, fill=255, stroke_width=max(2, size // 10), stroke_fill=255)
        glow = glow.filter(ImageFilter.GaussianBlur(size / 4))

        crisp = Image.new("L", (img_w, img_h), 0)
        ImageDraw.Draw(crisp).text(origin, text, font=font, fill=255)
        outline = Image.new("L", (img_w, img_h), 0)
        ImageDraw.Draw(outline).text(origin, text, font=font, fill=255, stroke_width=max(1, size // 18), stroke_fill=255)

        g = np.asarray(glow, np.float32) / 255.0
        c = np.asarray(crisp, np.float32) / 255.0
        o = np.asarray(outline, np.float32) / 255.0
        alpha = np.clip(np.maximum(g * 0.85, o), 0, 1)
        rgb = (
            np.array(GLOW_RGB, np.float32) * (g * 0.85)[..., None] * (1 - o[..., None])
            + np.array((10, 20, 10), np.float32) * (o - c).clip(0)[..., None]
            + np.array(TEXT_RGB, np.float32) * c[..., None]
        )
        result = (rgb[..., ::-1].copy(), alpha)
        self._cache[key] = result
        return result

    def draw(self, frame: np.ndarray, t: float) -> None:
        """Composites the current message onto the bottom of the frame, in place."""
        h, w = frame.shape[:2]
        idx = int(t // self.seconds) % len(MESSAGES)
        phase = (t % self.seconds) / self.seconds
        fade = min(1.0, phase / 0.08, (1.0 - phase) / 0.08)
        flicker = 0.9 + 0.1 * math.sin(t * 11.0) * math.sin(t * 3.7)
        strength = max(0.0, fade) * flicker
        if strength <= 0.01:
            return

        rgb, alpha = self._render(idx, w)
        bh, bw = alpha.shape
        bob = int(math.sin(t * 2.2) * h * 0.004)
        x = max(0, (w - bw) // 2)
        y = max(0, h - bh - int(h * 0.02) + bob)
        bh, bw = min(bh, h - y), min(bw, w - x)
        roi = frame[y : y + bh, x : x + bw].astype(np.float32)
        a = alpha[:bh, :bw, None] * strength
        roi = roi * (1 - a) + rgb[:bh, :bw] * strength
        frame[y : y + bh, x : x + bw] = np.clip(roi, 0, 255).astype(np.uint8)
