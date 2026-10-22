"""Rare cameos: a few bats flitting along the ceiling, or a skeleton dog trotting across the floor.

They're meant as a "wait, did you see that?" moment, not a show competing with the dance floor:
a few minutes apart, a few seconds long, and never while a banner message is up. The dog is
drawn with the same pens as the dancers, so the room lights it, it casts a floor shadow, and it
passes behind or in front of dancers by depth. The bats are soft dark silhouettes up high.
"""

import math
import random
from collections import deque
from typing import Deque, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from .anatomy import CONF_THRESH, L_AN, R_AN
from .banner import ShuffleBag
from .skeleton import PALETTES, SkeletonCanvas, make_pen
from .tracking import Dancer

BATS, DOG = "bats", "dog"
# Bats are the quick one; the dog is the punchline, so it comes up less often
DECK = [BATS, BATS, DOG]

BAT_COLOR = np.array((16, 11, 14), np.float32)  # BGR: near-black with a hint of purple
BAT_OPACITY = 0.9

Shadow = Tuple[Tuple[float, float], Tuple[float, float], float]  # center, half-axes (pixels), strength 0-1


def _rotate(p: np.ndarray, angle: float) -> np.ndarray:
    c, s = math.cos(angle), math.sin(angle)
    return np.stack([p[..., 0] * c - p[..., 1] * s, p[..., 0] * s + p[..., 1] * c], axis=-1)


def _smoothstep(u: float) -> float:
    u = min(1.0, max(0.0, u))
    return u * u * (3 - 2 * u)


class FloorScale:
    """Learns where the dance floor is and how tall a person looks on each part of it (lower in
    frame = nearer = taller) from dancers' feet, so the dog walks on the floor at the right size."""

    MIN_SAMPLES = 60
    FALLBACK = 0.62  # person height / foot y, measured in the party room before anyone danced

    def __init__(self) -> None:
        self.samples: Deque[Tuple[float, float, float]] = deque(maxlen=3000)  # foot x, foot y, height

    def observe(self, dancers: Sequence[Dancer]) -> None:
        for d in dancers:
            k = d.kpts
            if d.opacity >= 0.95 and min(k[L_AN, 2], k[R_AN, 2]) >= CONF_THRESH:
                self.samples.append((float(d.box[0] + d.box[2]) / 2, float(d.box[3]), d.height))

    @property
    def learned(self) -> bool:
        return len(self.samples) >= self.MIN_SAMPLES

    def person_height(self, foot_y: float) -> float:
        if not self.learned:
            return self.FALLBACK * foot_y
        ys, hs = np.array(self.samples)[:, 1:].T
        if np.ptp(ys) > 80:
            slope, intercept = np.polyfit(ys, hs, 1)
            if slope > 0:
                return max(40.0, slope * foot_y + intercept)
        return float(np.median(hs / np.maximum(ys, 1.0))) * foot_y

    def floor_y(self, rng: random.Random, h: int) -> float:
        """Somewhere people have actually stood, so the dog walks on the dance floor."""
        y = rng.choice(self.samples)[1] if self.learned else rng.uniform(0.62, 0.85) * h
        return float(np.clip(y, 0.45 * h, 0.92 * h))

    def floor_span(self, y: float, w: int, h: int) -> Tuple[float, float]:
        """Left and right ends of the floor at height y: where people have stood at that depth.
        The fallback is the rug in the party room."""
        if self.learned:
            pts = np.array(self.samples)
            xs = pts[np.abs(pts[:, 1] - y) < 0.06 * h, 0]
            if len(xs) >= 10:
                lo, hi = np.percentile(xs, [3, 97])
                return float(lo), float(hi)
        return 0.3 * w, 0.75 * w


# A bat seen from below with its wings out, wingspan 1. Each wing point is (x, y, how much it
# follows the wingtip up and down); the trailing edge is scalloped between the finger bones.
BAT_WING = np.array([
    (0.03, -0.03, 0.0), (0.20, -0.09, 0.45), (0.50, -0.03, 1.0), (0.41, 0.05, 0.8), (0.34, 0.01, 0.68),
    (0.26, 0.09, 0.5), (0.18, 0.04, 0.35), (0.10, 0.09, 0.15), (0.03, 0.06, 0.0),
], np.float32)  # fmt: skip
BAT_EAR = np.array([(-0.032, -0.07), (-0.026, -0.125), (-0.006, -0.085)], np.float32)


def bat_polygons(flap: float, depth: float) -> List[np.ndarray]:
    """Bat outline (wingspan 1) at wingbeat phase `flap`. `depth` (0-1) is how hard it's
    flapping; 0 is a glide with wings flat out."""
    lift = -0.24 * math.sin(flap) * depth
    # Wings look shorter at the top and bottom of the stroke, where they're angled to the camera
    fold = 1.0 - 0.38 * abs(math.sin(flap)) * depth
    right = np.stack([BAT_WING[:, 0] * fold, BAT_WING[:, 1] + lift * BAT_WING[:, 2]], axis=1)
    left = right * np.array([-1.0, 1.0], np.float32)
    body = cv2.ellipse2Poly((0, 10), (35, 75), 0, 0, 360, 30).astype(np.float32) / 1000.0
    head = cv2.ellipse2Poly((0, -60), (30, 28), 0, 0, 360, 30).astype(np.float32) / 1000.0
    ears = [BAT_EAR, BAT_EAR * np.array([-1.0, 1.0], np.float32)]
    return [right, left[::-1], body, head, *ears]


class Bat:
    def __init__(self, rng: random.Random, w: int, h: int, direction: int, y: float, delay: float):
        self.span = w * 0.036 * rng.uniform(0.7, 1.3)
        self.direction = direction
        self.w = w
        self.speed = (w + 2 * self.span) / rng.uniform(2.2, 3.4)  # px/s
        self.delay = delay
        self.y = y
        self.flap_hz = rng.uniform(5.5, 7.5)
        self.phase = rng.uniform(0, 2 * math.pi)
        # Bats flit: two overlapping wobbles up and down, plus spells of gliding
        self.wobble = [(h * rng.uniform(0.015, 0.035), rng.uniform(0.5, 0.9), rng.uniform(0, 2 * math.pi)),
                       (h * rng.uniform(0.006, 0.012), rng.uniform(1.8, 2.8), rng.uniform(0, 2 * math.pi))]  # fmt: skip
        self.glide = (rng.uniform(0.3, 0.6), rng.uniform(0, 2 * math.pi))

    def position(self, s: float) -> Tuple[float, float]:
        travelled = s * self.speed - self.span
        x = travelled if self.direction > 0 else self.w - travelled
        y = self.y + sum(a * math.sin(2 * math.pi * f * s + p) for a, f, p in self.wobble)
        return x, y

    def done(self, s: float) -> bool:
        return s * self.speed > self.w + 2 * self.span

    def draw(self, frame: np.ndarray, s: float) -> None:
        if s < 0 or self.done(s):
            return
        x, y = self.position(s)
        _, y_next = self.position(s + 0.05)
        bank = math.atan2(y_next - y, self.speed * 0.05) * 0.6 * self.direction
        depth = 0.55 + 0.45 * math.sin(2 * math.pi * self.glide[0] * s + self.glide[1])
        polys = [_rotate(p, bank) * self.span + (x, y) for p in bat_polygons(self.phase + 2 * math.pi * self.flap_hz * s, depth)]

        h, w = frame.shape[:2]
        pad = 4
        allpts = np.concatenate(polys)
        x0, y0 = np.floor(allpts.min(axis=0)).astype(int) - pad
        x1, y1 = np.ceil(allpts.max(axis=0)).astype(int) + pad
        cx0, cy0, cx1, cy1 = max(0, x0), max(0, y0), min(w, x1), min(h, y1)
        if cx1 <= cx0 or cy1 <= cy0:
            return
        mask = np.zeros((y1 - y0, x1 - x0), np.uint8)
        # Drawn at 4x subpixel precision so small, fast bats don't jitter by whole pixels
        cv2.fillPoly(mask, [np.round((p - (x0, y0)) * 16).astype(np.int32) for p in polys], 255, cv2.LINE_AA, shift=4)
        # As soft as the camera image, and a little more for motion blur
        mask = cv2.GaussianBlur(mask, (0, 0), max(0.7, self.span / 80))
        a = mask[cy0 - y0 : cy1 - y0, cx0 - x0 : cx1 - x0, None].astype(np.float32) * (BAT_OPACITY / 255.0)
        roi = frame[cy0:cy1, cx0:cx1]
        roi[:] = (roi * (1 - a) + BAT_COLOR * a).astype(np.uint8)


class Flock:
    def __init__(self, rng: random.Random, w: int, h: int, start: float):
        direction = rng.choice([-1, 1])
        center = h * rng.uniform(0.07, 0.24)
        # A lone bat is spookier than a swarm, so sometimes that's all it is
        n = rng.choice([1, 3, 4, 5, 6, 7])
        self.bats = [Bat(rng, w, h, direction, center + h * rng.uniform(-0.06, 0.06), rng.uniform(0, 1.4) if i else 0.0)
                     for i in range(n)]  # fmt: skip
        self.start = start

    def done(self, now: float) -> bool:
        return all(b.done(now - self.start - b.delay) for b in self.bats)

    def draw(self, frame: np.ndarray, now: float) -> None:
        for b in self.bats:
            b.draw(frame, now - self.start - b.delay)


# Dog skull in profile, length 1, origin at the back of the skull, snout along +x, y down
DOG_CRANIUM = np.array([
    (0.0, 0.02), (0.04, -0.12), (0.18, -0.2), (0.36, -0.17), (0.5, -0.08), (0.75, -0.03),
    (0.98, 0.0), (1.0, 0.07), (0.94, 0.12), (0.55, 0.15), (0.32, 0.2), (0.12, 0.16),
], np.float32)  # fmt: skip
DOG_JAW = np.array([(0.22, 0.16), (0.94, 0.13), (0.92, 0.19), (0.6, 0.23), (0.3, 0.27), (0.2, 0.22)], np.float32)
DOG_JAW_HINGE = np.array([0.24, 0.17], np.float32)
DOG_EYE = np.array([(0.35, -0.08), (0.46, -0.09), (0.49, -0.01), (0.41, 0.04), (0.34, 0.0)], np.float32)
DOG_TEMPLE = np.array([(0.16, -0.02), (0.28, -0.04), (0.3, 0.08), (0.18, 0.1)], np.float32)
DOG_NOSE = np.array([(0.92, 0.01), (0.98, 0.03), (0.95, 0.07)], np.float32)
# Dog backbone from tail to neck, in the dog's own frame (see Dog)
SPINE_X = np.array([-0.60, -0.48, -0.05, 0.34, 0.48])
SPINE_Y = np.array([-0.93, -0.97, -1.0, -1.03, -1.0])
PELVIS = np.array([(-0.36, -1.0), (-0.42, -1.04), (-0.58, -0.98), (-0.56, -0.88), (-0.44, -0.83), (-0.40, -0.92)])
SCAPULA = np.array([(0.28, -1.02), (0.44, -1.02), (0.43, -0.74), (0.37, -0.74)])


def _two_bone(root: np.ndarray, target: np.ndarray, l1: float, l2: float, bend: float) -> np.ndarray:
    """Middle joint of a two-bone limb reaching from root toward target. bend = +1 puts the
    joint behind the root-target line (y down, facing +x), -1 in front."""
    d = target - root
    dist = float(np.clip(np.linalg.norm(d), 1e-6, l1 + l2 - 1e-6))
    u = d / max(1e-6, float(np.linalg.norm(d)))
    along = (l1 * l1 - l2 * l2 + dist * dist) / (2 * dist)
    out = math.sqrt(max(0.0, l1 * l1 - along * along))
    return root + u * along + np.array([-u[1], u[0]]) * out * bend


class Dog:
    """A ghostly skeleton dog: it fades in at one end of the dance floor, trots partway across,
    stops to look up at the dancers and bark twice with its tail going, then trots on and fades
    out at the other end. Fading in and out on the floor keeps it from walking through furniture.

    Lengths are in shoulder heights (S) in the dog's own frame: x forward, y down, origin on
    the floor under its middle.
    """

    TROT_PERIOD = 0.42  # seconds per stride
    STRIDE = 0.55  # how far a planted paw travels back under the body, in S
    STANCE = 0.55  # fraction of each stride a paw is on the floor
    BRAKE = 0.4  # seconds to slow to a stop (or get going again)
    PAUSE = 3.0
    BARKS = (0.9, 1.6)  # seconds into the pause
    FADE = 1.2  # distance (in S) over which it materializes and vanishes
    FRONT, HIND = 0.44, -0.46  # where the paws stand
    # (leg, near side, stride phase): a trot moves diagonal pairs together
    LEGS = (("front", True, 0.0), ("hind", False, 0.0), ("front", False, 0.5), ("hind", True, 0.5))

    def __init__(self, rng: random.Random, w: int, h: int, floor: FloorScale):
        self.gy = floor.floor_y(rng, h)
        # A medium dog stands about a third as tall as a person
        self.S = float(np.clip(0.32 * floor.person_height(self.gy), 30.0, 0.3 * h))
        lo, hi = floor.floor_span(self.gy, w, h)
        if hi - lo < 4 * self.S:
            mid = (lo + hi) / 2
            lo, hi = max(0.0, mid - 2 * self.S), min(float(w), mid + 2 * self.S)
        self.facing = rng.choice([-1, 1])
        self.start_x, self.end_x = (lo, hi) if self.facing > 0 else (hi, lo)
        self.x = self.start_x
        self.stop_x = self.start_x + self.facing * (hi - lo) * rng.uniform(0.4, 0.6)
        self.speed = self.STRIDE * self.S / (self.STANCE * self.TROT_PERIOD)  # px/s: planted paws don't slide
        self.state = "in"
        self.m = 1.0  # 0 = standing still, 1 = trotting
        self.phase = rng.random()
        self.pause_t = 0.0

    @property
    def done(self) -> bool:
        return self.state == "out" and (self.end_x - self.x) * self.facing <= 0

    @property
    def opacity(self) -> float:
        fade = self.FADE * self.S
        return _smoothstep((self.x - self.start_x) * self.facing / fade) * _smoothstep((self.end_x - self.x) * self.facing / fade)

    def update(self, dt: float) -> None:
        if self.state == "in" and (self.stop_x - self.x) * self.facing <= self.speed * self.BRAKE / 2:
            self.state = "brake"
        if self.state == "brake":
            self.m = max(0.0, self.m - dt / self.BRAKE)
            if self.m == 0.0:
                self.state = "pause"
        elif self.state == "pause":
            self.pause_t += dt
            if self.pause_t >= self.PAUSE:
                self.state = "out"
        elif self.state == "out":
            self.m = min(1.0, self.m + dt / self.BRAKE)
        self.x += self.facing * self.speed * self.m * dt
        self.phase = (self.phase + self.m * dt / self.TROT_PERIOD) % 1.0

    def paw(self, neutral: float, phase: float) -> np.ndarray:
        p = phase % 1.0
        if p < self.STANCE:  # planted, sliding back under the body as it moves forward
            off, lift = 0.5 - p / self.STANCE, 0.0
        else:  # swinging forward through the air
            u = (p - self.STANCE) / (1 - self.STANCE)
            off, lift = -0.5 + _smoothstep(u), 0.12 * math.sin(math.pi * u)
        return np.array([neutral + self.STRIDE * off * self.m, -lift * self.m])

    def world(self, p: np.ndarray) -> np.ndarray:
        return np.stack([self.x + self.facing * p[..., 0] * self.S, self.gy + p[..., 1] * self.S], axis=-1)

    def shadow(self) -> Shadow:
        return (self.x, self.gy), (0.8 * self.S, 0.12 * self.S), self.opacity

    # Pause choreography: look up, two barks, tail wagging the whole time
    def _look(self) -> float:
        if self.state != "pause":
            return 0.0
        return _smoothstep(self.pause_t / 0.35) * _smoothstep((self.PAUSE - self.pause_t) / 0.35)

    def _jaw(self) -> float:
        if self.state != "pause":
            return 0.0
        return max((0.42 * math.sin(math.pi * (self.pause_t - b) / 0.22) if 0 <= self.pause_t - b <= 0.22 else 0.0)
                   for b in self.BARKS)  # fmt: skip

    def draw(self, canvas: SkeletonCanvas, palette: str) -> None:
        opacity = self.opacity
        if opacity <= 0.01:
            return
        # Slighter bones than a person's
        t = float(np.clip(self.S / 20.0, 2.0, 12.0))
        color = PALETTES.get(palette, PALETTES["bone"])[0]
        pen = make_pen(canvas, palette, color, t, opacity)
        far_pen = make_pen(canvas, palette, color, t * 0.85, opacity)
        bob = -0.02 * abs(math.sin(2 * math.pi * self.phase)) * self.m
        body = np.array([0.0, bob])
        far = np.array([0.04, -0.05])  # the camera is up high, so the far side sits a little higher

        def P(*pts) -> List[Tuple[int, int]]:
            return [tuple(int(round(v)) for v in self.world(np.asarray(p, np.float32))) for p in pts]

        def leg(pn, kind: str, offset: np.ndarray, phase: float) -> None:
            if kind == "front":
                root = np.array([0.42, -0.72]) + body + offset
                paw = self.paw(self.FRONT, self.phase + phase) + offset
                wrist = paw + (-0.03, -0.12)
                elbow = _two_bone(root, wrist, 0.28, 0.40, bend=1.0)
                joints = [root, elbow, wrist, paw, paw + (0.09, 0.0)]
            else:
                root = np.array([-0.48, -0.88]) + body + offset
                paw = self.paw(self.HIND, self.phase + phase) + offset
                hock = paw + (-0.10, -0.24)
                knee = _two_bone(root, hock, 0.34, 0.38, bend=-1.0)
                joints = [root, knee, hock, paw, paw + (0.08, 0.0)]
            pts = P(*joints)
            for (a, b), sc in zip(zip(pts, pts[1:]), (0.8, 0.65, 0.5, 0.4)):
                pn.bone(a, b, sc)
            for j, sc in zip(pts[:3], (0.75, 0.6, 0.5)):
                pn.knob(j, sc)

        for kind, near, phase in self.LEGS:
            if not near:
                leg(far_pen, kind, far, phase)

        # Tail: carried level and swaying while trotting, wagging hard while it stops to bark
        if self.state == "pause":
            lift = 0.45 + 0.3 * math.sin(2 * math.pi * 3.5 * self.pause_t)
        else:
            lift = 0.15 + 0.08 * math.sin(2 * math.pi * self.phase)
        tail = [np.array([SPINE_X[0], SPINE_Y[0]]) + body]
        for _ in range(10):
            lift += 0.035
            tail.append(tail[-1] + np.array([-math.cos(lift), -math.sin(lift)]) * 0.06)
        tail_px = P(*tail)
        pen.curve(np.array(tail_px), 0.3)
        for i, p in enumerate(tail_px[1:]):
            pen.knob(p, 0.45 - 0.025 * i)

        # Spine, ribs (sweeping back, the chest deep in front and tucked up behind) and breastbone
        pen.curve(np.array(P(*(np.stack([SPINE_X, SPINE_Y], axis=1) + body))), 0.45)
        for x in np.linspace(SPINE_X[-1], SPINE_X[0], 14):
            pen.knob(P((x, np.interp(x, SPINE_X, SPINE_Y) + bob))[0], 0.5)
        bottoms = []
        for x in np.linspace(0.30, -0.22, 8):
            top = float(np.interp(x, SPINE_X, SPINE_Y)) + bob
            bot = float(np.interp(x, [-0.22, 0.30], [-0.68, -0.47])) + bob
            span = bot - top
            rib = P((x, top), (x - 0.09, top + 0.3 * span), (x - 0.08, top + 0.75 * span), (x, bot))
            pen.curve(np.array(rib), 0.4)
            bottoms.append(rib[-1])
        pen.curve(np.array(bottoms[:6]), 0.4)
        pen.solid(np.array(P(*(PELVIS + body))))

        # Neck and skull
        look = self._look()
        neck_top = np.array([0.66, -1.16 - 0.1 * look]) + body
        neck = P(*np.linspace(np.array([SPINE_X[-1], SPINE_Y[-1]]) + body, neck_top, 5))
        pen.curve(np.array(neck), 0.7)
        for p in neck[1:-1]:
            pen.knob(p, 0.8)
        length = 0.5
        angle = 0.5 - 0.6 * look  # nose down while trotting; up to look at the dancers

        def skull(pts: np.ndarray, jaw: float = 0.0) -> np.ndarray:
            if jaw:
                pts = _rotate(pts - DOG_JAW_HINGE, jaw) + DOG_JAW_HINGE
            return np.array(P(*(neck_top + _rotate(pts, angle) * length)))

        pen.solid(skull(DOG_JAW, self._jaw()))
        pen.solid(skull(DOG_CRANIUM))
        pen.hole(skull(DOG_TEMPLE))
        pen.hole(skull(DOG_EYE))
        pen.hole(skull(DOG_NOSE))
        for x in (0.6, 0.72, 0.84):  # teeth, which show when it barks
            a, b = skull(np.array([(x, 0.12), (x - 0.01, 0.17)], np.float32))
            pen.bone((int(a[0]), int(a[1])), (int(b[0]), int(b[1])), 0.25)
        eye = skull(np.array([(0.41, -0.02)], np.float32))[0]
        pen.eye_glow((int(eye[0]), int(eye[1])), max(1, int(length * self.S * 0.045)))

        pen.solid(np.array(P(*(SCAPULA + body))))
        for kind, near, phase in self.LEGS:
            if near:
                leg(pen, kind, np.zeros(2), phase)


class Cameos:
    """Decides when a cameo happens and runs it. Call update() once per output frame, draw the
    dog into the skeleton canvas, and draw_overlay() on the finished frame for the bats."""

    FIRST_AFTER = (45.0, 90.0)  # seconds after startup

    def __init__(self, every: float = 180.0, enabled: bool = True, seed: Optional[int] = None):
        self.every = every
        self.enabled = enabled
        self.rng = random.Random(seed)
        self.deck = ShuffleBag(DECK, self.rng)
        self.floor = FloorScale()
        self.flock: Optional[Flock] = None
        self.dog: Optional[Dog] = None
        self.next_at: Optional[float] = None
        self.pending: Optional[str] = None
        self.last: Optional[float] = None

    @property
    def active(self) -> bool:
        return self.flock is not None or self.dog is not None

    def trigger(self, kind: str) -> None:
        """Run this cameo now (or as soon as the current one ends), even if cameos are off."""
        self.pending = kind

    def update(self, now: float, shape: Tuple[int, ...], dancers: Sequence[Dancer], quiet: bool = True) -> None:
        """Advances the current cameo, or starts one when it's due. `quiet` is False while a
        banner message is on screen; a cameo that comes due then waits until it's gone."""
        self.floor.observe(dancers)
        dt = 0.0 if self.last is None else min(0.2, max(0.0, now - self.last))
        self.last = now
        if self.next_at is None:
            self.next_at = now + self.rng.uniform(*self.FIRST_AFTER)

        if self.dog is not None:
            self.dog.update(dt)
            if self.dog.done:
                self.dog = None
                self._schedule(now)
        if self.flock is not None and self.flock.done(now):
            self.flock = None
            self._schedule(now)
        if self.active:
            return

        kind = self.pending
        if kind is None and self.enabled and quiet and now >= self.next_at:
            kind = self.deck.next()
        self.pending = None
        h, w = shape[:2]
        if kind == BATS:
            self.flock = Flock(self.rng, w, h, now)
        elif kind == DOG:
            self.dog = Dog(self.rng, w, h, self.floor)

    def _schedule(self, now: float) -> None:
        # Uneven gaps, so it never feels like it's on a timer
        self.next_at = now + self.every * self.rng.uniform(0.7, 1.3)

    def draw_overlay(self, frame: np.ndarray, now: float) -> None:
        if self.flock is not None:
            self.flock.draw(frame, now)
