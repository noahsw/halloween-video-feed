"""Skeleton drawing.

Two looks share the same anatomy code through a "pen":
  * BonePen  - ivory bones with dark outlines, later lit by the room's light (looks like a real skeleton)
  * NeonPen  - the glowing neon tube look from the original script

Pens draw into a SkeletonCanvas: an opaque layer (premultiplied color + alpha) that gets
alpha-composited over the background, plus an additive glow layer that gets bloomed.
"""

import math
from typing import List, Sequence, Tuple

import cv2
import numpy as np

from .anatomy import L_AN, L_EL, L_HIP, L_KN, L_SH, L_WR, R_AN, R_EL, R_HIP, R_KN, R_SH, R_WR, MIN_WIDTH_TO_TORSO
from .anatomy import estimate_head, unit
from .anatomy import kpt as _pt
from .tracking import Dancer

Point = Tuple[int, int]

# BGR colors
BONE_IVORY = (200, 226, 238)
BONE_SHADOW = (38, 40, 46)
HOLE_DARK = (14, 14, 18)

PALETTES = {
    # "bone" colors are only used for the glowing eyes; bones stay ivory.
    "bone": [(80, 255, 57), (255, 60, 190), (0, 140, 255), (255, 230, 0)],
    "multi": [
        (80, 255, 57),  # Ecto Neon Green
        (255, 230, 0),  # Electric Cyan
        (255, 50, 190),  # Ultraviolet Magenta
        (0, 140, 255),  # Hellfire Orange
        (130, 255, 0),  # Toxic Lime
        (255, 90, 90),  # Spectral Blue
    ],
    "ecto": [(80, 255, 57)],
    "phantom": [(255, 240, 0), (255, 60, 180)],
    "inferno": [(0, 140, 255), (0, 60, 255), (0, 215, 255)],
}
PALETTE_NAMES = list(PALETTES)

# Below this bone thickness (a dancer roughly 240 px tall, i.e. at the back of the room),
# fine details like finger bones and paired forearm bones are a few pixels wide and unreadable
# on the TV, so they're skipped. That keeps a packed dance floor cheap to draw.
DETAIL_MIN_T = 5.0

# Dark edge around each bone, as a fraction of bone thickness (0 = none). It separates bones
# from bright backgrounds, but too thick reads as a fuzzy halo once softened and AirPlayed.
BONE_OUTLINE = 0.15


class SkeletonCanvas:
    def __init__(self, shape: Tuple[int, ...]):
        h, w = shape[:2]
        self.color = np.zeros((h, w, 3), np.uint8)  # premultiplied by alpha
        self.alpha = np.zeros((h, w), np.uint8)
        self.glow = np.zeros((h, w, 3), np.uint8)


def _brighten(color: Tuple[int, int, int], base: int = 140) -> Tuple[int, int, int]:
    return tuple(min(255, int(c * 0.45 + base)) for c in color)  # type: ignore[return-value]


def _ipt(p: np.ndarray | Sequence[float]) -> Point:
    return int(round(p[0])), int(round(p[1]))


def _scale(color: Tuple[int, int, int], k: float) -> Tuple[int, int, int]:
    return tuple(int(c * k) for c in color)  # type: ignore[return-value]


class BonePen:
    def __init__(self, canvas: SkeletonCanvas, eye_color: Tuple[int, int, int], t: float, opacity: float = 1.0):
        # Colors are premultiplied by opacity, which is how the canvas stores them
        self.c = canvas
        self.eye_color = _scale(eye_color, opacity)
        self.ivory = _scale(BONE_IVORY, opacity)
        self.shadow = _scale(BONE_SHADOW, opacity)
        self.dark = _scale(HOLE_DARK, opacity)
        self.a = int(255 * opacity)
        self.opacity = opacity
        self.t = t
        self.detailed = t >= DETAIL_MIN_T
        self.outline = int(round(t * BONE_OUTLINE))

    def _opaque(self, draw, color, thickness: int) -> None:
        draw(self.c.color, color, thickness)
        draw(self.c.alpha, self.a, thickness)

    def bone(self, p1: Point, p2: Point, scale: float = 1.0) -> None:
        th = max(1, int(round(self.t * scale)))
        line = lambda img, col, w: cv2.line(img, p1, p2, col, w, cv2.LINE_AA)
        self._opaque(line, self.shadow, th + 2 * self.outline)
        line(self.c.color, self.ivory, th)  # the outline pass already set coverage here

    def knob(self, p: Point, scale: float = 1.0) -> None:
        r = max(1, int(round(self.t * 0.75 * scale)))
        circ = lambda img, col, w: cv2.circle(img, p, w, col, -1, cv2.LINE_AA)
        self._opaque(circ, self.shadow, r + self.outline)
        circ(self.c.color, self.ivory, r)

    def curve(self, pts: np.ndarray, scale: float = 1.0, closed: bool = False) -> None:
        th = max(1, int(round(self.t * scale)))
        pts = pts.astype(np.int32).reshape((-1, 1, 2))
        poly = lambda img, col, w: cv2.polylines(img, [pts], closed, col, w, cv2.LINE_AA)
        self._opaque(poly, self.shadow, th + 2 * self.outline)
        poly(self.c.color, self.ivory, th)

    def solid(self, pts: np.ndarray) -> None:
        """Filled bone surface (cranium, pelvis)."""
        pts = pts.astype(np.int32).reshape((-1, 1, 2))
        cv2.fillPoly(self.c.color, [pts], self.ivory, cv2.LINE_AA)
        cv2.fillPoly(self.c.alpha, [pts], self.a, cv2.LINE_AA)
        if self.outline > 0:
            cv2.polylines(self.c.color, [pts], True, self.shadow, self.outline * 2, cv2.LINE_AA)
            cv2.polylines(self.c.alpha, [pts], True, self.a, self.outline * 2, cv2.LINE_AA)

    def hole(self, pts: np.ndarray) -> None:
        """Dark cavity (eye sockets, nose)."""
        pts = pts.astype(np.int32).reshape((-1, 1, 2))
        cv2.fillPoly(self.c.color, [pts], self.dark, cv2.LINE_AA)
        cv2.fillPoly(self.c.alpha, [pts], self.a, cv2.LINE_AA)

    def crack(self, p1: Point, p2: Point) -> None:
        """Thin dark line on top of bone (teeth gaps)."""
        cv2.line(self.c.color, p1, p2, self.shadow, max(1, self.outline), cv2.LINE_AA)

    def eye_glow(self, p: Point, r: int) -> None:
        # The one thing that glows: eyes are a light source, not lit by the room
        cv2.circle(self.c.glow, p, r * 3, self.eye_color, -1, cv2.LINE_AA)
        cv2.circle(self.c.color, p, max(1, r), _brighten(self.eye_color, int(120 * self.opacity)), -1, cv2.LINE_AA)


class NeonPen:
    """The original glowing-tube look: a colored aura with a hot white-tinted core."""

    def __init__(self, canvas: SkeletonCanvas, color: Tuple[int, int, int], t: float, opacity: float = 1.0):
        self.c = canvas
        self.color = _scale(color, opacity)
        self.core = _scale(_brighten(color), opacity)
        self.dark = _scale(HOLE_DARK, opacity)
        self.white = _scale((255, 255, 255), opacity)
        self.a = int(255 * opacity)
        self.t = t
        self.detailed = t >= DETAIL_MIN_T

    def _core(self, draw, thickness: int) -> None:
        draw(self.c.color, self.core, thickness)
        draw(self.c.alpha, self.a, thickness)

    def bone(self, p1: Point, p2: Point, scale: float = 1.0) -> None:
        th = max(1, int(round(self.t * scale * 0.7)))
        cv2.line(self.c.glow, p1, p2, self.color, th * 4, cv2.LINE_AA)
        self._core(lambda img, col, w: cv2.line(img, p1, p2, col, w, cv2.LINE_AA), th)

    def knob(self, p: Point, scale: float = 1.0) -> None:
        r = max(1, int(round(self.t * 0.6 * scale)))
        cv2.circle(self.c.glow, p, r * 3, self.color, -1, cv2.LINE_AA)
        self._core(lambda img, col, w: cv2.circle(img, p, w, col, -1, cv2.LINE_AA), r)

    def curve(self, pts: np.ndarray, scale: float = 1.0, closed: bool = False) -> None:
        th = max(1, int(round(self.t * scale * 0.7)))
        pts = pts.astype(np.int32).reshape((-1, 1, 2))
        cv2.polylines(self.c.glow, [pts], closed, self.color, th * 4, cv2.LINE_AA)
        self._core(lambda img, col, w: cv2.polylines(img, [pts], closed, col, w, cv2.LINE_AA), th)

    def solid(self, pts: np.ndarray) -> None:
        pts = pts.astype(np.int32).reshape((-1, 1, 2))
        cv2.fillPoly(self.c.color, [pts], self.dark, cv2.LINE_AA)
        cv2.fillPoly(self.c.alpha, [pts], self.a, cv2.LINE_AA)
        self.curve(pts, closed=True)

    def hole(self, pts: np.ndarray) -> None:
        pts = pts.astype(np.int32).reshape((-1, 1, 2))
        cv2.fillPoly(self.c.glow, [pts], self.color, cv2.LINE_AA)
        cv2.polylines(self.c.color, [pts], True, self.core, max(1, int(self.t * 0.5)), cv2.LINE_AA)

    def crack(self, p1: Point, p2: Point) -> None:
        self.bone(p1, p2, 0.5)

    def eye_glow(self, p: Point, r: int) -> None:
        cv2.circle(self.c.color, p, max(2, r), self.white, -1, cv2.LINE_AA)


def _perp(u: np.ndarray) -> np.ndarray:
    return np.array([-u[1], u[0]], np.float32)


def draw_skull(pen, center: np.ndarray, radius: float, roll: float) -> None:
    cos_a, sin_a = math.cos(roll), math.sin(roll)

    def rp(lx: float, ly: float) -> Point:
        return _ipt(
            (center[0] + (lx * cos_a - ly * sin_a) * radius, center[1] + (lx * sin_a + ly * cos_a) * radius)
        )

    def poly(local: Sequence[Tuple[float, float]]) -> np.ndarray:
        return np.array([rp(x, y) for x, y in local], np.int32)

    cranium = [
        (-0.78, -0.35), (-0.92, -0.05), (-0.84, 0.28), (-0.55, 0.48), (-0.46, 0.92),
        (0.0, 1.02), (0.46, 0.92), (0.55, 0.48), (0.84, 0.28), (0.92, -0.05),
        (0.78, -0.35), (0.58, -0.78), (0.0, -0.96), (-0.58, -0.78),
    ]  # fmt: skip
    pen.solid(poly(cranium))

    for side in (-1.0, 1.0):
        socket = [(side * x, y) for x, y in [(0.12, -0.10), (0.55, -0.12), (0.62, 0.10), (0.42, 0.24), (0.16, 0.18)]]
        pen.hole(poly(socket))
        pen.eye_glow(rp(side * 0.37, 0.06), max(1, int(radius * 0.07)))

    pen.hole(poly([(0.0, 0.26), (-0.13, 0.48), (0.13, 0.48)]))

    teeth_top, teeth_bot = 0.62, 0.84
    pen.hole(poly([(-0.36, teeth_top), (0.36, teeth_top), (0.30, teeth_bot), (-0.30, teeth_bot)]))
    pen.solid(poly([(-0.32, teeth_top + 0.02), (0.32, teeth_top + 0.02), (0.27, teeth_bot - 0.02), (-0.27, teeth_bot - 0.02)]))
    for tx in (-0.18, -0.06, 0.06, 0.18):
        pen.crack(rp(tx, teeth_top), rp(tx * 0.9, teeth_bot))
    pen.crack(rp(-0.30, (teeth_top + teeth_bot) / 2), rp(0.30, (teeth_top + teeth_bot) / 2))


def _draw_ribcage(pen, sh_mid, hip_mid, l_sh, r_sh, show_ribs: bool) -> None:
    spine_vec = hip_mid - sh_mid
    torso_len = float(np.linalg.norm(spine_vec))
    if torso_len < 10:
        return
    u_spine = spine_vec / torso_len
    u_perp = _perp(u_spine)
    sh_w = max(float(np.linalg.norm(l_sh - r_sh)), MIN_WIDTH_TO_TORSO * torso_len)

    # Vertebrae
    n_vert = 9 if pen.detailed else 5
    for i in range(n_vert + 1):
        pen.knob(_ipt(sh_mid + spine_vec * (i / n_vert)), 0.55)

    if show_ribs:
        ribs = zip([0.12, 0.23, 0.34, 0.45, 0.55], [0.40, 0.47, 0.48, 0.44, 0.36])
        if not pen.detailed:
            ribs = zip([0.15, 0.32, 0.5], [0.42, 0.48, 0.40])
        for frac, w_scale in ribs:
            center = sh_mid + u_spine * torso_len * frac
            half_w = sh_w * w_scale
            dip = u_spine * torso_len * 0.06
            for sign in (-1.0, 1.0):
                pts = np.array([
                    center,
                    center + sign * u_perp * half_w * 0.5 - dip * 0.5,
                    center + sign * u_perp * half_w * 0.95,
                    center + sign * u_perp * half_w * 0.9 + dip,
                ])  # fmt: skip
                pen.curve(pts, 0.55)

    # Pelvis: iliac wings, the pubic ring with its two holes, and the sacrum in the middle
    half_w = sh_w * 0.5

    def L(px: float, py: float) -> np.ndarray:
        return hip_mid + u_perp * px * half_w + u_spine * py * torso_len

    for s in (-1.0, 1.0):
        wing = [(0, -0.14), (s * 0.55, -0.22), (s * 1.0, -0.12), (s * 0.85, 0.02), (s * 0.6, 0.08), (s * 0.25, 0.12), (0, 0.06)]
        pen.solid(np.array([L(*p) for p in wing]))
        if not pen.detailed:
            continue
        ring = [(s * 0.6, 0.08), (s * 0.58, 0.22), (s * 0.2, 0.26), (0, 0.2)]
        pen.curve(np.array([L(*p) for p in ring]), 0.6)
        hole = [(s * 0.48, 0.12), (s * 0.44, 0.2), (s * 0.24, 0.2), (s * 0.28, 0.13)]
        pen.hole(np.array([L(*p) for p in hole]))
    pen.solid(np.array([L(-0.2, -0.14), L(0.2, -0.14), L(0, 0.12)]))


def _draw_limb_pair(pen, a: np.ndarray, b: np.ndarray, gap: float) -> None:
    """Forearm (radius + ulna) and shin (tibia + fibula) are two thin parallel bones."""
    if not pen.detailed:
        pen.bone(_ipt(a), _ipt(b), 0.7)
        return
    off = _perp(unit(b - a)) * gap
    pen.bone(_ipt(a + off), _ipt(b + off), 0.55)
    pen.bone(_ipt(a - off), _ipt(b - off), 0.45)


def _draw_hand(pen, elbow: np.ndarray, wrist: np.ndarray) -> None:
    fwd = wrist - elbow
    length = float(np.linalg.norm(fwd)) * 0.38
    if not pen.detailed or length < 3:
        pen.knob(_ipt(wrist), 0.8)
        return
    base_angle = math.atan2(fwd[1], fwd[0])
    for spread in (-0.5, -0.17, 0.17, 0.5):
        ang = base_angle + spread
        tip = wrist + np.array([math.cos(ang), math.sin(ang)]) * length * (1.0 - abs(spread) * 0.4)
        pen.bone(_ipt(wrist), _ipt(tip), 0.3)
    pen.knob(_ipt(wrist), 0.8)


def draw_dancer(pen, d: Dancer, show_skull: bool = True, show_ribs: bool = True) -> None:
    k = d.kpts
    P = lambda i: _pt(k, i)
    l_sh, r_sh, l_hip, r_hip = P(L_SH), P(R_SH), P(L_HIP), P(R_HIP)
    have_shoulders = l_sh is not None and r_sh is not None
    have_hips = l_hip is not None and r_hip is not None
    sh_mid = (l_sh + r_sh) * 0.5 if have_shoulders else None
    hip_mid = (l_hip + r_hip) * 0.5 if have_hips else None

    # Back to front: torso, legs, arms, skull
    if sh_mid is not None and hip_mid is not None:
        _draw_ribcage(pen, sh_mid, hip_mid, l_sh, r_sh, show_ribs)
    elif l_sh is not None and r_hip is not None:
        pen.bone(_ipt(l_sh), _ipt(r_hip))
    elif r_sh is not None and l_hip is not None:
        pen.bone(_ipt(r_sh), _ipt(l_hip))

    gap = pen.t * 0.45
    for hip, knee, ankle in ((L_HIP, L_KN, L_AN), (R_HIP, R_KN, R_AN)):
        h, kn, an = P(hip), P(knee), P(ankle)
        if h is not None and kn is not None:
            pen.bone(_ipt(h), _ipt(kn), 1.15)
            pen.knob(_ipt(h), 1.1)
        if kn is not None and an is not None:
            _draw_limb_pair(pen, kn, an, gap)
            foot = an + unit(an - kn) * float(np.linalg.norm(an - kn)) * 0.18
            pen.bone(_ipt(an), _ipt(foot), 0.8)
        if kn is not None:
            pen.knob(_ipt(kn), 1.0)

    if have_shoulders:
        neck_base = sh_mid + (unit(hip_mid - sh_mid) if hip_mid is not None else np.array([0, 1])) * pen.t
        for sh in (l_sh, r_sh):
            pen.bone(_ipt(sh), _ipt(neck_base), 0.7)

    for sh_i, el_i, wr_i in ((L_SH, L_EL, L_WR), (R_SH, R_EL, R_WR)):
        sh, el, wr = P(sh_i), P(el_i), P(wr_i)
        if sh is not None and el is not None:
            pen.bone(_ipt(sh), _ipt(el), 0.95)
        if el is not None and wr is not None:
            _draw_limb_pair(pen, el, wr, gap * 0.8)
            _draw_hand(pen, el, wr)
        for j in (sh, el):
            if j is not None:
                pen.knob(_ipt(j), 0.9)

    head = d.head if d.head is not None else estimate_head(k)
    if head is None:
        return
    center, radius, roll = head
    if sh_mid is not None:
        chin = center + np.array([-math.sin(roll), math.cos(roll)]) * radius * 0.9
        pen.bone(_ipt(sh_mid), _ipt(chin), 0.6)
    if show_skull:
        draw_skull(pen, center, radius, roll)


def make_pen(canvas: SkeletonCanvas, palette_name: str, color: Tuple[int, int, int], t: float, opacity: float = 1.0):
    pen_cls = BonePen if palette_name == "bone" else NeonPen
    return pen_cls(canvas, color, t, opacity)


def thickness_for(d: Dancer) -> float:
    """Bone thickness scales with how big the dancer appears (near vs far from the camera)."""
    return float(np.clip(d.height / 48.0, 2.0, 18.0))


def draw_dancers(
    canvas: SkeletonCanvas,
    dancers: List[Dancer],
    palette_name: str,
    show_skulls: bool,
    show_ribs: bool,
) -> None:
    colors = PALETTES.get(palette_name, PALETTES["bone"])
    # Draw far-away (smaller, higher in frame) dancers first so nearer ones overlap them
    for i, d in enumerate(sorted(dancers, key=lambda d: d.box[3])):
        idx = d.track_id if d.track_id is not None else i
        color = colors[idx % len(colors)]
        draw_dancer(make_pen(canvas, palette_name, color, thickness_for(d), d.opacity), d, show_skulls, show_ribs)
