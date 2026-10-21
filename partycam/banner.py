"""Funny "come join the dance floor" messages that pop up every so often.

Messages come from shuffled decks (see messages.py), so every message is shown once
before any repeats. Half the time, if one fits, the banner picks a message written for
the moment instead: how busy the floor is, the time of night, or a live skeleton count.
"""

import math
import random
from datetime import datetime
from collections import OrderedDict, deque
from pathlib import Path
from typing import Deque, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

from . import messages

FONT_CANDIDATES = [
    "/System/Library/Fonts/Supplemental/Chalkduster.ttf",
    "/System/Library/Fonts/Supplemental/Impact.ttf",
    "/Library/Fonts/Arial Bold.ttf",
]

TEXT_RGB = (236, 255, 220)
GLOW_RGB = (120, 255, 60)


def _find_font() -> Optional[str]:
    return next((p for p in FONT_CANDIDATES if Path(p).exists()), None)


class ShuffleBag:
    """Hands out every item once, in random order, before repeating any. When it reshuffles,
    the most recently shown third goes to the back of the new deck, so a message from the end
    of one round can't pop up again right at the start of the next."""

    def __init__(self, items: List[str], rng: random.Random):
        self.items = list(items)
        self.rng = rng
        self.bag: List[str] = []  # popped from the end
        self.recent: Deque[str] = deque(maxlen=max(1, len(items) // 3))

    def next(self) -> str:
        if not self.bag:
            fresh = [m for m in self.items if m not in self.recent]
            held_back = [m for m in self.items if m in self.recent]
            self.rng.shuffle(fresh)
            self.rng.shuffle(held_back)
            self.bag = held_back + fresh
        item = self.bag.pop()
        self.recent.append(item)
        return item


def _floor_pool(most: int) -> str:
    if most == 0:
        return "EMPTY"
    if most == 1:
        return "SOLO"
    if most <= 5:
        return "FEW"
    if most <= 19:
        return "CROWD"
    return "PACKED"


def _clock_pools(now: datetime) -> List[str]:
    minutes = now.hour * 60 + now.minute
    pools = []
    if 17 * 60 <= minutes < 21 * 60:
        pools.append("EARLY")
    if minutes >= 23 * 60 + 45 or minutes < 15:
        pools.append("WITCHING")
    if 60 <= minutes < 5 * 60:
        pools.append("LATE")
    if 105 <= minutes < 5 * 60:
        pools.append("LAST_CALL")
    return pools


def _clock_text(now: datetime) -> str:
    return f"{now.hour % 12 or 12}:{now.minute:02d} {'AM' if now.hour < 12 else 'PM'}"


class JoinBanner:
    SITUATION_WINDOW = 5.0  # seconds of dancer counts used to judge the moment
    SITUATION_CHANCE = 0.5  # how often a fitting situational message wins over a general one
    # Situational decks are small, so each is rationed (seconds between uses)
    COOLDOWNS = {"EMPTY": 150, "SOLO": 150, "FEW": 150, "CROWD": 150, "PACKED": 150,
                 "EARLY": 300, "WITCHING": 240, "LATE": 300, "LAST_CALL": 300,
                 "LIVE_COUNT": 420, "LIVE_TIME": 900}  # fmt: skip
    FADE = 0.6

    def __init__(self, every: float = 30.0, show_for: float = 8.0, seed: Optional[int] = None):
        self.every = max(every, show_for + 1.0)
        self.show_for = show_for
        self.rng = random.Random(seed)
        self.bags = {name: ShuffleBag(getattr(messages, name), self.rng) for name in ["GENERAL", *self.COOLDOWNS]}
        self.last_used: Dict[str, float] = {name: -math.inf for name in self.COOLDOWNS}
        self.font_path = _find_font()
        self._cache: "OrderedDict[Tuple[str, int], Tuple[np.ndarray, np.ndarray]]" = OrderedDict()
        self.counts: Deque[Tuple[float, int]] = deque()
        self.current: Optional[str] = None
        self.started = 0.0
        self.next_at: Optional[float] = None

    def _situations(self, t: float, most: int) -> List[str]:
        """Situational pools that fit right now and aren't on cooldown."""
        now = datetime.fromtimestamp(t)
        fits = [_floor_pool(most), *_clock_pools(now)]
        minutes = now.hour * 60 + now.minute
        if not 17 * 60 <= minutes < 21 * 60 + 30:
            fits.append("LIVE_TIME")  # announcing the time only gets fun once the night is going
        if most >= 2:
            fits.append("LIVE_COUNT")
        return [p for p in fits if t - self.last_used[p] >= self.COOLDOWNS[p]]

    def _pick(self, t: float) -> str:
        most = max((c for _, c in self.counts), default=0)
        candidates = self._situations(t, most)
        if candidates and self.rng.random() < self.SITUATION_CHANCE:
            pool = self.rng.choice(candidates)
            self.last_used[pool] = t
        else:
            pool = "GENERAL"
        text = self.bags[pool].next()
        if pool.startswith("LIVE_"):
            text = text.format(n=most, next=most + 1, time=_clock_text(datetime.fromtimestamp(t)))
        return text

    def _schedule(self, t: float) -> None:
        if self.current is None and self.next_at is None:
            self.next_at = t + 3.0  # first message shortly after startup
        if self.current is not None and t - self.started >= self.show_for:
            self.current = None
            # A little randomness in the gap so it doesn't feel like a metronome
            self.next_at = t + (self.every - self.show_for) * self.rng.uniform(0.75, 1.25)
        if self.current is None and self.next_at is not None and t >= self.next_at:
            self.current = self._pick(t)
            self.started = t
            self.next_at = None

    def _render(self, text: str, frame_w: int) -> Tuple[np.ndarray, np.ndarray]:
        """Returns (premultiplied BGR float32, alpha float32) for a message."""
        key = (text, frame_w)
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]
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
        while len(self._cache) > 3:  # each rendered message is a few MB
            self._cache.popitem(last=False)
        return result

    def draw(self, frame: np.ndarray, t: float, dancer_count: int = 0) -> None:
        """Composites the current message (if one is showing) onto the bottom of the frame, in place."""
        self.counts.append((t, dancer_count))
        while self.counts and self.counts[0][0] < t - self.SITUATION_WINDOW:
            self.counts.popleft()
        self._schedule(t)
        if self.current is None:
            return

        h, w = frame.shape[:2]
        age = t - self.started
        fade = min(1.0, age / self.FADE, (self.show_for - age) / self.FADE)
        flicker = 0.9 + 0.1 * math.sin(t * 11.0) * math.sin(t * 3.7)
        strength = max(0.0, fade) * flicker
        if strength <= 0.01:
            return

        rgb, alpha = self._render(self.current, w)
        bh, bw = alpha.shape
        bob = int(math.sin(t * 2.2) * h * 0.004)
        x = max(0, (w - bw) // 2)
        y = max(0, h - bh - int(h * 0.02) + bob)
        bh, bw = min(bh, h - y), min(bw, w - x)
        roi = frame[y : y + bh, x : x + bw].astype(np.float32)
        a = alpha[:bh, :bw, None] * strength
        roi = roi * (1 - a) + rgb[:bh, :bw] * strength
        frame[y : y + bh, x : x + bw] = np.clip(roi, 0, 255).astype(np.uint8)
