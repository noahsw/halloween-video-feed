"""Missing-person posters that cut into the feed every so often, like an emergency broadcast.

Drop poster images in local/missing/ (gitignored: they're photos of your guests). About every
10 minutes one cuts in through a burst of static, under a red "MISSING PERSON ALERT" bar, holds
for a minute with a slow push-in, and cuts back out. Every poster shows once before any
repeats. They wait for a banner message or cameo to finish, so there's only ever one thing on
screen, and keep the next banner or cameo back for a few seconds after.
"""

import math
import random
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

from .banner import ShuffleBag

IMAGE_TYPES = {".jpg", ".jpeg", ".png", ".webp"}
FONT_CANDIDATES = [
    "/System/Library/Fonts/Supplemental/Impact.ttf",
    "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
    "/Library/Fonts/Arial Bold.ttf",
]

ALERT = "MISSING PERSON ALERT"
CAPTION = "FULL CASE FILE POSTED IN THE RESTROOMS"
ALERT_RGB = (196, 22, 28)
WARN_RGB = (255, 206, 38)
CAPTION_RGB = (232, 232, 232)


def find_posters(folder: Path) -> List[Path]:
    if not folder.is_dir():
        return []
    return sorted(p for p in folder.iterdir() if p.suffix.lower() in IMAGE_TYPES and not p.name.startswith("."))


def _font(size: int) -> ImageFont.FreeTypeFont:
    path = next((p for p in FONT_CANDIDATES if Path(p).exists()), None)
    return ImageFont.truetype(path, size) if path else ImageFont.load_default(size)


def glitch(frame: np.ndarray, amount: float, rng: np.random.Generator) -> np.ndarray:
    """TV snow and horizontal tearing over the frame; `amount` 0 is clean, 1 is all snow."""
    out = frame.copy()
    h, w = out.shape[:2]
    for _ in range(int(amount * 14)):
        y = int(rng.integers(0, h))
        band = slice(y, y + int(rng.integers(2, max(3, h // 10))))
        out[band] = np.roll(out[band], int(rng.integers(-w // 6, w // 6 + 1)), axis=1)
    # Chunky snow, like an old set losing the signal
    snow = rng.integers(0, 256, (h // 4 + 1, w // 4 + 1), dtype=np.uint8)
    snow = cv2.cvtColor(cv2.resize(snow, (w, h), interpolation=cv2.INTER_NEAREST), cv2.COLOR_GRAY2BGR)
    return cv2.addWeighted(out, 1.0 - amount, snow, amount, 0)


def _warning_sign(draw: ImageDraw.ImageDraw, cx: float, cy: float, size: float, font: ImageFont.FreeTypeFont) -> None:
    half = size / 2
    draw.polygon([(cx, cy - half), (cx + half * 1.1, cy + half * 0.85), (cx - half * 1.1, cy + half * 0.85)], fill=WARN_RGB)
    draw.text((cx, cy + half * 0.2), "!", font=font, fill=(20, 20, 20), anchor="mm")


class MissingPosters:
    """Decides when a poster cuts in and draws it. Call update() once per output frame, then
    draw() on the finished frame; while one is up it replaces the whole picture."""

    FIRST_AFTER = (90.0, 180.0)  # seconds after startup
    GLITCH_IN = 1.5  # seconds of static cutting in: long enough to read as "we interrupt this program"
    GLITCH_OUT = 0.6
    PUSH_IN = 0.05  # slow zoom over the showing, so it never looks like a frozen frame
    COOLDOWN = 5.0  # seconds after a poster before a banner or cameo may start
    BAR = 0.065  # alert bar height, as a fraction of the frame
    CAPTION_BAR = 0.05

    def __init__(self, folder: Path, every: float = 600.0, show_for: float = 60.0, enabled: bool = True, seed: Optional[int] = None):
        self.paths = find_posters(folder)
        self.every = max(every, show_for + 1.0)
        self.show_for = show_for
        self.enabled = enabled
        self.rng = random.Random(seed)
        self.noise = np.random.default_rng(seed)
        self.deck = ShuffleBag([str(p) for p in self.paths], self.rng) if self.paths else None
        self.current: Optional[str] = None
        self.started = 0.0
        self.ended = -math.inf
        self.next_at: Optional[float] = None
        self.pending = False
        self._assets: Optional[Tuple] = None

    def trigger(self) -> None:
        """Cut to the next poster now (even if posters are off, or one is already up)."""
        self.pending = True

    def busy(self, now: float) -> bool:
        """True while a poster is up, and for a few seconds after."""
        return self.current is not None or now - self.ended < self.COOLDOWN

    def update(self, now: float, quiet: bool = True) -> None:
        """Ends the current poster or starts one when it's due. `quiet` is False while something
        else (a banner message, a cameo) is on screen; a poster that comes due then waits."""
        if self.next_at is None:
            self.next_at = now + self.rng.uniform(*self.FIRST_AFTER)
        if self.current is not None and now - self.started >= self.show_for:
            self.current = None
            self.ended = now
            # Uneven gaps, so it never feels like it's on a timer
            self.next_at = now + (self.every - self.show_for) * self.rng.uniform(0.85, 1.15)
        if self.deck is None:
            self.pending = False
            return
        if self.pending or (self.enabled and quiet and self.current is None and now >= self.next_at):
            self.current = self.deck.next()
            self.started = now
            self.pending = False

    def draw(self, frame: np.ndarray, now: float) -> None:
        """Replaces the frame with the current poster (if one is up), in place."""
        if self.current is None:
            return
        age = now - self.started
        poster = self._compose(frame.shape, age)
        if age < self.GLITCH_IN:
            p = age / self.GLITCH_IN
            base = frame if p < 0.5 else poster  # the swap happens under full snow
            # Snow builds fast, rolls for a while with the picture flickering through, then clears
            amount = min(1.0, p / 0.2, (1.0 - p) / 0.3) * self.noise.uniform(0.85, 1.0)
        elif age > self.show_for - self.GLITCH_OUT:
            p = (age - (self.show_for - self.GLITCH_OUT)) / self.GLITCH_OUT
            base = poster if p < 0.5 else frame
            amount = 1.0 - abs(2.0 * p - 1.0)
        else:
            frame[:] = poster
            return
        frame[:] = glitch(base, amount, self.noise)

    def _compose(self, shape: Tuple[int, ...], age: float) -> np.ndarray:
        h, w = shape[:2]
        key = (self.current, h, w)
        if self._assets is None or self._assets[0] != key:
            self._assets = (key, *self._build(Path(self.current), h, w))
        _, backgrounds, poster, (y0, rh) = self._assets
        out = backgrounds[int(age * 3) % 2].copy()  # the warning signs blink
        # Zoom from fit to fit + PUSH_IN; the stored poster is the fully zoomed size. Sub-pixel
        # placement via warpAffine keeps such a slow zoom from stepping a pixel at a time.
        s = (1.0 + self.PUSH_IN * min(1.0, age / self.show_for)) / (1.0 + self.PUSH_IN)
        ph, pw = poster.shape[:2]
        m = np.float32([[s, 0, (w - s * pw) / 2], [0, s, (rh - s * ph) / 2]])
        band = out[y0 : y0 + rh]
        cv2.warpAffine(poster, m, (w, rh), dst=band, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_TRANSPARENT)
        return out

    def _build(self, path: Path, h: int, w: int) -> Tuple[List[np.ndarray], np.ndarray, Tuple[int, int]]:
        """Returns ([background with the signs lit, unlit], poster at full zoom, (poster band top, height))."""
        img = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
        src = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
        ih, iw = src.shape[:2]
        bar_h, cap_h = round(h * self.BAR), round(h * self.CAPTION_BAR)
        y0, rh = bar_h, h - bar_h - cap_h

        fit = min(rh / ih, w * 0.9 / iw)
        big = fit * (1.0 + self.PUSH_IN)
        poster = cv2.resize(src, (round(iw * big), round(ih * big)), interpolation=cv2.INTER_AREA)

        # Behind it, the same poster blown up to cover the frame, blurred and dark (blurred at
        # 1/8 size: it's smooth anyway, and a full-size blur this wide is slow)
        cover = max(w / iw, h / ih)
        cw, ch = math.ceil(iw * cover), math.ceil(ih * cover)
        small = cv2.resize(src, (max(1, cw // 8), max(1, ch // 8)), interpolation=cv2.INTER_AREA)
        bg = cv2.resize(cv2.GaussianBlur(small, (0, 0), 5), (cw, ch), interpolation=cv2.INTER_LINEAR)
        x, y = (cw - w) // 2, (ch - h) // 2
        bg = cv2.convertScaleAbs(bg[y : y + h, x : x + w], alpha=0.3)
        # Soft shadow so the poster lifts off the blur
        shadow = np.zeros((h, w), np.uint8)
        fw, fh = iw * fit, ih * fit
        cv2.rectangle(shadow, (int((w - fw) / 2), int(y0 + (rh - fh) / 2)), (int((w + fw) / 2), int(y0 + (rh + fh) / 2)), 255, -1)
        shadow = cv2.GaussianBlur(shadow, (0, 0), h / 50)
        bg = cv2.multiply(bg, cv2.merge([255 - (shadow * 0.7).astype(np.uint8)] * 3), scale=1.0 / 255.0)

        base = Image.fromarray(cv2.cvtColor(bg, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(base)
        draw.rectangle([0, 0, w, bar_h], fill=ALERT_RGB)
        alert_font = _font(round(bar_h * 0.62))
        draw.text((w / 2, bar_h / 2), ALERT, font=alert_font, fill=(255, 255, 255), anchor="mm")
        draw.rectangle([0, h - cap_h, w, h], fill=(14, 14, 14))
        draw.rectangle([0, h - cap_h, w, h - cap_h + max(2, h // 400)], fill=ALERT_RGB)
        draw.text((w / 2, h - cap_h / 2), CAPTION, font=_font(round(cap_h * 0.55)), fill=CAPTION_RGB, anchor="mm")

        lit = base.copy()
        tx0, _, tx1, _ = draw.textbbox((w / 2, bar_h / 2), ALERT, font=alert_font, anchor="mm")
        sign = bar_h * 0.62
        sign_font = _font(round(sign * 0.7))
        lit_draw = ImageDraw.Draw(lit)
        for cx in (tx0 - sign * 1.2, tx1 + sign * 1.2):
            _warning_sign(lit_draw, cx, bar_h / 2, sign, sign_font)
        backgrounds = [cv2.cvtColor(np.asarray(im), cv2.COLOR_RGB2BGR) for im in (lit, base)]
        return backgrounds, poster, (y0, rh)
