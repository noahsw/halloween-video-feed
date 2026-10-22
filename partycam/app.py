#!/usr/bin/env python3
"""
Halloween Dance Party Skeleton Cam

Turns everyone on the dance floor into dancing skeletons. In the default mode the people
themselves vanish (the empty dance floor is painted over them) so only the skeletons are
left dancing in your room. Built for Apple Silicon (MPS) + a Tapo RTSP camera, and meant to
be AirPlayed to a TV.

Usage:
    uv run party-cam                      # camera from .env (see .env.example)
    uv run party-cam --source 0           # MacBook webcam

Live Party Hotkeys (while window is focused):
    1 / 2 / 3 / 4 : Background (1 = Invisible dancers, 2 = Dimmed Room, 3 = Pitch-Black Void, 4 = Spectral Night-Vision)
    s         : Toggle Skull Heads on/off
    r         : Toggle Ribcage on/off
    t         : Cycle Phantom Trail persistence (Medium -> Long Ghost Echo -> Short -> Off); off by default for bone style
    c         : Cycle Skeleton Style (Bone, Multi-Dancer Neon, Ecto-Green, Phantom Cyan/Purple, Hellfire Orange)
    j         : Toggle "Join the dance floor" banner
    b         : Re-learn the empty dance floor (use if the camera got bumped)
    z         : Draw an ignore zone (e.g. around a skeleton prop); x clears all zones
    m         : Toggle Horizontal Mirroring
    f         : Toggle Fullscreen (macOS native; drag the window onto the TV first)
    h         : Toggle help / FPS overlay
    q or ESC  : Quit
"""

import argparse
import dataclasses
import hashlib
import signal
import threading
import time
import traceback
from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np

from .banner import JoinBanner
from .cloak import InvisibilityCloak, upscale_region
from .config import LOCAL_DIR, MODELS_DIR, load_env, redact, rtsp_url_from_env
from .interp import Interpolator
from .macos import keep_process_fast, keep_thread_fast, thermal_state
from .anatomy import CONF_THRESH, L_AN, L_KN, R_AN, R_KN
from .skeleton import PALETTE_NAMES, SkeletonCanvas, draw_dancers
from .modelprocs import POSE, SEGMENT, ModelProcess, SharedFrame, detect_poses, half_shape, segment_people
from .stream import LowLatencyStreamReader
from .tracking import Dancer, PoseSmoother
from .zones import IgnoreZones

WINDOW_NAME = "Halloween Skeleton Party Cam"

BG_INVISIBLE, BG_DIM, BG_VOID, BG_SPECTRAL = 1, 2, 3, 4
TRAIL_PRESETS = [0.80, 0.90, 0.65, 0.0]  # Medium, Long Ghost Echo, Short, Off
TRAIL_OFF = TRAIL_PRESETS.index(0.0)
SEG_CONF = 0.25
# Blur (sigma, px) so bones are as soft as the camera image instead of razor sharp
SKELETON_SOFTEN = 0.4

# Limb segments painted into the person mask, on top of the segmentation mask, so
# fast-moving arms the segmenter misses still get erased.
MASK_LIMBS = [(5, 7), (7, 9), (6, 8), (8, 10), (11, 13), (13, 15), (12, 14), (14, 16), (5, 6), (11, 12), (5, 11), (6, 12)]


@dataclass
class Analysis:
    frame: np.ndarray
    dancers: List[Dancer]
    seg_mask: Optional[np.ndarray]  # half-resolution mask of everyone the segmentation model found
    t: float = 0.0  # when the camera frame arrived
    # Filled in once per camera frame by prepare_background(), then reused by every output frame
    bg: Optional[np.ndarray] = None  # the room with people erased (or the chosen background mode)
    motion: Optional[np.ndarray] = None  # motion edges for the ectoplasm shimmer


class PartyCam:
    """Frame in, skeleton party frame out. Holds all the state between frames.

    analyze() is the GPU-heavy half (pose + segmentation) and render() the CPU half,
    so they can run on separate threads, overlapping consecutive frames.
    """

    def __init__(self, args: argparse.Namespace, device: str, source: str = ""):
        from ultralytics import YOLO

        self.args = args
        self.device = device
        print(f"[*] Loading models on device='{device}'...")
        # Loaded here too: this downloads missing weights, and they're the fallback if a process fails
        self.pose_model = YOLO(str(MODELS_DIR / args.pose_model))
        self.seg_model = YOLO(str(MODELS_DIR / args.seg_model))
        self.seg_frame = 0
        self.last_seg_mask: Optional[np.ndarray] = None
        # Both models run in their own processes, at the same time (see modelprocs.py)
        self.frame_buf: Optional[SharedFrame] = SharedFrame()
        self.procs: dict = {}
        for kind, weights, imgsz, conf in [(POSE, args.pose_model, args.imgsz, args.conf),
                                           (SEGMENT, args.seg_model, args.seg_imgsz, SEG_CONF)]:  # fmt: skip
            try:
                warm = kind == POSE and not args.no_gpu_warm  # one process keeping the GPU busy is enough
                self.procs[kind] = ModelProcess(kind, str(MODELS_DIR / weights), imgsz, conf, device, self.frame_buf, warm)
            except Exception as e:
                print(f"[!] Couldn't start the {kind} process ({e}); running it in-thread (slower)")
        for kind, proc in list(self.procs.items()):
            try:
                proc.wait_ready()
            except Exception as e:
                self._drop_proc(kind, e)

        self.smoother = PoseSmoother()
        # One saved floor image per video source, so a webcam test never pollutes the party camera's
        source_key = hashlib.sha1(redact(source).encode()).hexdigest()[:10]
        self.cloak = InvisibilityCloak(LOCAL_DIR / f"floor_{source_key}.png")
        if args.fresh_background:
            self.cloak.plate_path.unlink(missing_ok=True)
        self.zones = IgnoreZones(LOCAL_DIR / "ignore_zones.json")
        self.banner = JoinBanner(every=args.banner_every)

        self.bg_mode = BG_INVISIBLE
        self.show_skulls = True
        self.show_ribs = True
        self.show_banner = not args.no_banner
        self.mirror = args.mirror
        self.palette_idx = 0
        # Each style remembers its own trail setting. Glowing neon suits ghost trails; on lit bones
        # they read as double vision, so bone starts with them off.
        self.trail_by_style = {name: TRAIL_OFF if name == "bone" else 0 for name in PALETTE_NAMES}

        self.trail: Optional[np.ndarray] = None
        self.prev_mask: Optional[np.ndarray] = None
        self.prev_gray: Optional[np.ndarray] = None
        self.grain = 0.0  # estimated camera noise level, matched on the skeletons
        self.last_render: Optional[float] = None
        self.dancer_count = 0
        self.timings: dict = {}

    @property
    def trail_idx(self) -> int:
        return self.trail_by_style[PALETTE_NAMES[self.palette_idx]]

    @trail_idx.setter
    def trail_idx(self, idx: int) -> None:
        self.trail_by_style[PALETTE_NAMES[self.palette_idx]] = idx

    def reset_buffers(self) -> None:
        self.trail = None
        self.prev_mask = None
        self.prev_gray = None

    def prepare(self, raw: np.ndarray) -> np.ndarray:
        h, w = raw.shape[:2]
        if w > self.args.width:
            # INTER_AREA is ~20x slower here (non-integer 2560->1920) for no visible gain
            raw = cv2.resize(raw, (self.args.width, int(h * self.args.width / w)), interpolation=cv2.INTER_LINEAR)
        return raw

    def detect(self, frame: np.ndarray) -> Tuple[List[Dancer], Optional[np.ndarray]]:
        return self.detect_collect(self.detect_submit(frame))

    def detect_submit(self, frame: np.ndarray) -> dict:
        """Starts both models on the frame (they run in other processes). Returns a job for
        detect_collect(); the caller can do CPU work in between."""
        h, w = frame.shape[:2]
        zone_boxes = self.zones.pixel_boxes(w, h)

        # The GPU is saturated running both models every frame, so with --seg-every 2
        # segmentation runs every other frame and the mask in between is reused (person_mask
        # also holds the previous frame's mask and paints everyone's current limbs on top, so a
        # 50 ms-old outline still covers).
        need_seg = self.bg_mode == BG_INVISIBLE or self.cloak.coverage < 0.97
        self.seg_frame += 1
        reuse = (
            self.last_seg_mask is not None
            and self.last_seg_mask.shape == half_shape(h, w)
            and self.seg_frame % max(1, self.args.seg_every) != 0
        )
        if need_seg and reuse:
            need_seg = False

        # Hand the frame to both model processes; they run at the same time
        sent = set()
        if self.procs and self.frame_buf is not None:
            try:
                self.frame_buf.write(frame)
            except ValueError as e:
                for kind in list(self.procs):
                    self._drop_proc(kind, e)
        for kind in (POSE, SEGMENT) if need_seg else (POSE,):
            if kind in self.procs:
                try:
                    self.procs[kind].submit(zone_boxes)
                    sent.add(kind)
                except OSError as e:
                    self._drop_proc(kind, e)
        return dict(frame=frame, zone_boxes=zone_boxes, need_seg=need_seg, reuse=reuse, sent=sent, started=time.perf_counter())

    def detect_collect(self, job: dict) -> Tuple[List[Dancer], Optional[np.ndarray]]:
        frame, zone_boxes, need_seg, reuse, sent = (job[k] for k in ("frame", "zone_boxes", "need_seg", "reuse", "sent"))
        h, w = frame.shape[:2]
        t_pose = job["started"]
        poses = self._collect(POSE, sent)
        if poses is None:
            poses = detect_poses(self.pose_model, frame, self.args.imgsz, self.args.conf, self.device)
        dancers = [
            Dancer(kpts=k, box=b, track_id=None if tid < 0 else int(tid))
            for k, b, tid in zip(*poses)
            if not self.zones.should_ignore(b, zone_boxes)
        ]
        dancers = self.smoother.update(dancers)
        t_seg = time.perf_counter()
        self.timings["pose"] = t_seg - t_pose
        self.timings["seg_wait"] = 0.0

        if not need_seg:
            self._gpu_idle()
            if not reuse:
                return dancers, None
            # Widen the reused outline a little to cover a frame's worth of movement
            grow = max(3, h // 120) | 1
            return dancers, cv2.dilate(self.last_seg_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (grow, grow)))
        seg_mask = self._collect(SEGMENT, sent)
        self._gpu_idle()
        self.timings["seg_wait"] = time.perf_counter() - t_seg
        if seg_mask is None:
            seg_mask = np.zeros(half_shape(h, w), np.uint8)
            segment_people(self.seg_model, frame, self.args.seg_imgsz, SEG_CONF, self.device, zone_boxes, seg_mask)
        self.last_seg_mask = seg_mask
        return dancers, seg_mask

    def _gpu_idle(self) -> None:
        if POSE in self.procs and not self.args.no_gpu_warm:
            try:
                self.procs[POSE].idle()
            except OSError as e:
                self._drop_proc(POSE, e)

    def _collect(self, kind: str, sent: set):
        if kind not in sent:
            return None
        try:
            return self.procs[kind].result()
        except (RuntimeError, OSError, EOFError) as e:
            self._drop_proc(kind, e)
            return None

    def _drop_proc(self, kind: str, error: Exception) -> None:
        print(f"[!] The {kind} process failed ({error}); running it in-thread from now on (slower)")
        proc = self.procs.pop(kind, None)
        if proc is not None:
            proc.close()

    def close(self) -> None:
        for proc in self.procs.values():
            proc.close()
        self.procs = {}
        if self.frame_buf is not None:
            self.frame_buf.close()
            self.frame_buf = None

    def person_mask(self, shape, dancers: List[Dancer], seg_mask: Optional[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        """Returns (binary mask, feathered float alpha) covering everyone, at full resolution.
        Built at half resolution since it gets dilated and blurred anyway."""
        h, w = shape[:2]
        sh, sw = h // 2, w // 2
        mask = seg_mask.copy() if seg_mask is not None and seg_mask.shape == (sh, sw) else np.zeros((sh, sw), np.uint8)
        for d in dancers:
            th = max(2, int(d.height * 0.065))
            k = d.kpts
            for a, b in MASK_LIMBS:
                if k[a, 2] >= CONF_THRESH and k[b, 2] >= CONF_THRESH:
                    cv2.line(mask, (int(k[a, 0] / 2), int(k[a, 1] / 2)), (int(k[b, 0] / 2), int(k[b, 1] / 2)), 255, th)
            # Feet: shoes extend past the ankle keypoint, and they're what leaks when people bounce
            for knee_i, ankle_i in ((L_KN, L_AN), (R_KN, R_AN)):
                if k[ankle_i, 2] >= CONF_THRESH:
                    ankle = k[ankle_i, :2]
                    down = ankle - k[knee_i, :2] if k[knee_i, 2] >= CONF_THRESH else np.array([0.0, d.height * 0.25])
                    toe = ankle + down * 0.25
                    cv2.line(mask, (int(ankle[0] / 2), int(ankle[1] / 2)), (int(toe[0] / 2), int(toe[1] / 2)), 255, th)
            if d.head is not None:
                c, rad, _ = d.head
                cv2.circle(mask, (int(c[0] / 2), int(c[1] / 2)), int(rad * 0.75), 255, -1)
        grow = max(3, sh // 70) | 1
        mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (grow, grow)))
        # Hold masks for a frame so detection flicker never reveals someone
        held = mask if self.prev_mask is None or self.prev_mask.shape != mask.shape else cv2.max(mask, self.prev_mask)
        self.prev_mask = mask

        feather = cv2.GaussianBlur(held, (0, 0), max(1.0, sh / 360))
        alpha = cv2.resize(feather, (w, h), interpolation=cv2.INTER_LINEAR).astype(np.float32) * (1.0 / 255.0)
        full = cv2.resize(held, (w, h), interpolation=cv2.INTER_NEAREST)
        return full, alpha

    def background(self, frame: np.ndarray, mask: np.ndarray, alpha: np.ndarray, gray_blur) -> np.ndarray:
        # Keep learning the empty floor in every mode, so switching back to invisible is instant
        cloaked = self.cloak.apply(frame, mask, alpha)
        if self.bg_mode == BG_INVISIBLE:
            return cv2.convertScaleAbs(cloaked, alpha=self.args.room_brightness)
        if self.bg_mode == BG_DIM:
            dark = cv2.convertScaleAbs(frame, alpha=0.28, beta=-10)
            b, g, r = cv2.split(dark)
            return cv2.merge([cv2.add(b, 18), g, r])
        if self.bg_mode == BG_VOID:
            return np.zeros_like(frame)
        edges = cv2.Canny(gray_blur, 35, 110)
        bg = np.zeros_like(frame)
        bg[edges > 0] = (75, 30, 55)
        return cv2.add(bg, cv2.convertScaleAbs(frame, alpha=0.15))

    def analyze(self, raw: np.ndarray, t: Optional[float] = None) -> Analysis:
        frame = self.prepare(raw)
        t0 = time.perf_counter()
        dancers, seg_mask = self.detect(frame)
        self.timings["detect"] = time.perf_counter() - t0
        return self.make_analysis(frame, dancers, seg_mask, t)

    def make_analysis(self, frame, dancers: List[Dancer], seg_mask, t: Optional[float]) -> Analysis:
        # Snapshot the dancers: the smoother keeps updating its own copies (e.g. fade-outs), and
        # interpolation needs this frame's poses exactly as they were
        dancers = [dataclasses.replace(d) for d in dancers]
        return Analysis(frame, dancers, seg_mask, t=time.time() if t is None else t)

    def prepare_background(self, a: Analysis) -> None:
        """The expensive per-camera-frame half of rendering: erase people (and learn the floor)."""
        if a.bg is not None:
            return
        t0 = time.perf_counter()
        frame = a.frame
        need_gray = self.bg_mode != BG_INVISIBLE
        gray_blur = cv2.GaussianBlur(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (9, 9), 0) if need_gray else None
        mask, alpha = self.person_mask(frame.shape, a.dancers, a.seg_mask)
        a.bg = self.background(frame, mask, alpha, gray_blur)
        # Ghostly ectoplasm shimmer where things move (not in invisible mode: it would outline people)
        if gray_blur is not None and self.prev_gray is not None and self.prev_gray.shape == gray_blur.shape:
            _, motion = cv2.threshold(cv2.absdiff(gray_blur, self.prev_gray), 22, 255, cv2.THRESH_BINARY)
            a.motion = cv2.Canny(motion, 50, 150)
        self.prev_gray = gray_blur
        if PALETTE_NAMES[self.palette_idx] == "bone":
            self.update_grain(frame)
        self.timings["background"] = time.perf_counter() - t0

    def render(self, a: Analysis, now: Optional[float] = None, dancers: Optional[List[Dancer]] = None) -> np.ndarray:
        """Draws skeletons over a's (erased) background. `dancers` are interpolated poses for this
        moment; by default, a's own."""
        now = time.time() if now is None else now
        self.prepare_background(a)
        t0 = time.perf_counter()
        dancers = a.dancers if dancers is None else dancers
        self.dancer_count = len(dancers)
        if self.trail is None or self.trail.shape != a.frame.shape:
            self.trail = np.zeros_like(a.frame)

        bg = a.bg
        canvas = SkeletonCanvas(a.frame.shape)
        palette = PALETTE_NAMES[self.palette_idx]
        draw_dancers(canvas, dancers, palette, self.show_skulls, self.show_ribs)
        if palette == "bone":
            # Neon emits its own light; bones should look like they're in the room
            bg = self.floor_shadows(bg, dancers)
            self.light_skeletons(canvas, bg)
        if a.motion is not None:
            canvas.glow[a.motion > 0] = (90, 45, 15)

        dt = 0.05 if self.last_render is None else min(0.2, max(0.001, now - self.last_render))
        self.last_render = now
        out = self.composite(bg, canvas, palette, dt)
        # Everything above runs un-mirrored so the learned floor and ignore zones never depend on it
        if self.mirror:
            out = cv2.flip(out, 1)
        if self.show_banner:
            self.banner.draw(out, now, self.dancer_count)
        self.timings["render"] = time.perf_counter() - t0
        return out

    def floor_shadows(self, bg: np.ndarray, dancers: List[Dancer]) -> np.ndarray:
        """Soft contact shadow under each skeleton's feet, so they stand on the floor."""
        h, w = bg.shape[:2]
        sw, sh = w // 4, h // 4
        shadow = np.zeros((sh, sw), np.uint8)
        for d in dancers:
            k = d.kpts
            feet = [k[i, :2] for i in (L_AN, R_AN) if k[i, 2] >= CONF_THRESH]
            base = np.mean(feet, axis=0) if feet else np.array([(d.box[0] + d.box[2]) / 2, d.box[3]])
            spread = float(np.linalg.norm(feet[0] - feet[1])) if len(feet) == 2 else 0.0
            half_w = max(spread * 0.6 + d.height * 0.05, d.height * 0.12)
            center = (int(base[0] / 4), int((base[1] + d.height * 0.015) / 4))
            axes = (max(1, int(half_w / 4)), max(1, int(half_w * 0.3 / 4)))
            cv2.ellipse(shadow, center, axes, 0, 0, 360, int(120 * d.opacity), -1)
        if not shadow.any():
            return bg
        shadow = cv2.resize(cv2.GaussianBlur(shadow, (0, 0), max(1.0, sh / 150)), (w, h), interpolation=cv2.INTER_LINEAR)
        return cv2.multiply(bg, cv2.merge([cv2.bitwise_not(shadow)] * 3), scale=1.0 / 255.0)

    def update_grain(self, frame: np.ndarray) -> None:
        """Estimates the camera's noise from a center crop (robust median of the high-pass)."""
        h, w = frame.shape[:2]
        crop = cv2.cvtColor(frame[h // 2 - 160 : h // 2 + 160, w // 2 - 160 : w // 2 + 160], cv2.COLOR_BGR2GRAY)
        highpass = cv2.absdiff(crop, cv2.GaussianBlur(crop, (3, 3), 0))
        sigma = min(10.0, 1.85 * float(np.median(highpass)))
        self.grain += 0.05 * (sigma - self.grain)

    def light_skeletons(self, canvas: SkeletonCanvas, bg: np.ndarray) -> None:
        """Tints and dims the bones by the light around them (purple party light makes purple
        bones, dark corners make darker bones), then matches the camera's softness and grain."""
        x, y, rw, rh = cv2.boundingRect(canvas.alpha)
        if rw == 0 or rh == 0:
            return
        h, w = bg.shape[:2]
        G = 8
        # 1/8 scale: 1920x1080 divides evenly, which keeps INTER_AREA on its fast path
        small = cv2.resize(bg, (max(1, w // G), max(1, h // G)), interpolation=cv2.INTER_AREA).astype(np.float32)
        small = cv2.GaussianBlur(small, (0, 0), 4.0)
        luma = small @ np.array([0.114, 0.587, 0.299], np.float32)
        # Bones stay brighter than their surroundings so they read, but follow the room's light
        intensity = np.clip(0.5 + luma / 120.0, 0.5, 1.1)
        tint = np.clip(1.0 + 0.6 * ((small + 12.0) / (luma[..., None] + 12.0) - 1.0), 0.5, 1.5)
        # As 8-bit with 128 = 1.0, so the full-size multiply below stays on fast integer paths
        light_small = np.clip(tint * intensity[..., None] * 128.0, 0, 255).astype(np.uint8)

        # Pad so the softening blur isn't clipped at the box edge
        x, y = max(0, x - 3), max(0, y - 3)
        rw, rh = min(w - x, rw + 6), min(h - y, rh + 6)
        roi = (slice(y, y + rh), slice(x, x + rw))
        # Upscale only the part of the (very smooth) light map that covers the skeletons
        light = upscale_region(light_small, G, x, y, rw, rh)
        soft = SKELETON_SOFTEN
        color = cv2.multiply(cv2.GaussianBlur(canvas.color[roi], (0, 0), soft), light, scale=1.0 / 128.0)
        alpha = cv2.GaussianBlur(canvas.alpha[roi], (0, 0), soft)
        if self.grain > 0.5:
            # Monochrome grain, weighted by coverage (premultiplied), added/subtracted with saturation
            noise = np.empty((rh, rw), np.float32)
            cv2.randn(noise, 0.0, self.grain)
            noise = cv2.multiply(noise, alpha, scale=1.0 / 255.0, dtype=cv2.CV_32F)
            color = cv2.add(color, cv2.merge([cv2.convertScaleAbs(cv2.max(noise, 0.0))] * 3))
            color = cv2.subtract(color, cv2.merge([cv2.convertScaleAbs(cv2.min(noise, 0.0))] * 3))
        canvas.color[roi] = color
        canvas.alpha[roi] = alpha

    def process(self, raw: np.ndarray, now: Optional[float] = None) -> np.ndarray:
        return self.render(self.analyze(raw), now)

    def composite(self, bg: np.ndarray, canvas: SkeletonCanvas, palette: str, dt: float = 0.05) -> np.ndarray:
        h, w = bg.shape[:2]
        # Cheap wide bloom: blur at quarter resolution
        small = cv2.resize(canvas.glow, (w // 4, h // 4), interpolation=cv2.INTER_AREA)
        small = cv2.GaussianBlur(small, (0, 0), 2.5)
        glow_gain = 0.55 if palette == "bone" else 1.35
        # Presets are per 50 ms (one camera frame); scale so trails last as long at any output rate
        decay = TRAIL_PRESETS[self.trail_idx] ** (dt / 0.05)

        if decay > 0:
            bloom = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
            out = cv2.addWeighted(bg, 1.0, bloom, glow_gain, 0)
            # Phantom trails drift upward like rising spirits
            self.trail[:-2] = self.trail[2:]
            self.trail[-2:] = 0
            ghost = cv2.addWeighted(canvas.color, 0.30, bloom, 0.25, 0)
            self.trail = cv2.addWeighted(self.trail, decay, ghost, 1.0, 0)
            out = cv2.add(out, cv2.convertScaleAbs(self.trail, alpha=0.6))
        else:
            self.trail.fill(0)
            out = bg.copy()
            # Without trails, only the area around the glow needs the bloom added
            gx, gy, gw, gh = cv2.boundingRect(small.max(axis=2))
            if gw and gh:
                pad = 8  # blur reach, in quarter-res pixels
                gx, gy = max(0, gx - pad), max(0, gy - pad)
                gw, gh = min(small.shape[1] - gx, gw + 2 * pad), min(small.shape[0] - gy, gh + 2 * pad)
                x, y, rw, rh = gx * 4, gy * 4, min(w - gx * 4, gw * 4), min(h - gy * 4, gh * 4)
                bloom = upscale_region(small, 4, x, y, rw, rh)
                out[y : y + rh, x : x + rw] = cv2.addWeighted(bg[y : y + rh, x : x + rw], 1.0, bloom, glow_gain, 0)

        # Skeletons over everything, only where they are
        x, y, rw, rh = cv2.boundingRect(canvas.alpha)
        if rw and rh:
            roi = (slice(y, y + rh), slice(x, x + rw))
            inv = cv2.merge([cv2.bitwise_not(canvas.alpha[roi])] * 3)
            out[roi] = cv2.add(cv2.multiply(out[roi], inv, scale=1.0 / 255.0), canvas.color[roi])
        return out


class DetectionWorker(threading.Thread):
    """Runs detection and people-erasing off the main thread, pipelined: while the model
    processes work on camera frame N+1 on the GPU, this thread erases people from frame N on
    the CPU, then publishes it. The main thread only draws skeletons. Costs one frame (~50 ms)
    of extra delay, which AirPlay's 1-2 s lag hides."""

    def __init__(self, cam: PartyCam, stream: LowLatencyStreamReader):
        super().__init__(daemon=True)
        self.cam = cam
        self.stream = stream
        self.running = True
        self.lock = threading.Lock()
        self.latest: Optional[Analysis] = None
        self.seq = 0

    def _publish(self, a: Analysis) -> None:
        with self.lock:
            self.latest = a
            self.seq += 1

    def run(self) -> None:
        keep_thread_fast()
        last_id = -1
        ready: Optional[Analysis] = None  # analyzed, people not erased yet
        while self.running:
            frame_id, raw, arrived = self.stream.read_timed()
            if raw is None or frame_id == last_id:
                time.sleep(0.002)
                continue
            last_id = frame_id
            try:
                frame = self.cam.prepare(raw)
                t0 = time.perf_counter()
                job = self.cam.detect_submit(frame)  # GPU starts on this frame...
                if ready is not None:  # ...while the CPU erases people from the previous one
                    self.cam.prepare_background(ready)
                    self._publish(ready)
                dancers, seg_mask = self.cam.detect_collect(job)
                self.cam.timings["detect"] = time.perf_counter() - t0
                ready = self.cam.make_analysis(frame, dancers, seg_mask, arrived)
            except Exception:
                traceback.print_exc()
                ready = None
                time.sleep(0.5)

    def get(self) -> Tuple[int, Optional[Analysis]]:
        with self.lock:
            return self.seq, self.latest

    def stop(self) -> None:
        self.running = False
        self.join(timeout=2.0)


class PerfLog:
    """Prints one line every 10 s so problems are diagnosable from the terminal, e.g.
    [perf] camera 20.0 fps -> shown 19.6 fps | frame gaps 50/58 ms (median/p95) | ..."""

    def __init__(self, stream: LowLatencyStreamReader, worker: "DetectionWorker", every: float = 10.0):
        self.stream = stream
        self.worker = worker
        self.every = every
        self.start = time.time()
        self.start_id = stream.read()[0]
        self.start_seq = worker.get()[0]
        self.show: List[float] = []
        self.pose: List[float] = []
        self.seg_wait: List[float] = []
        self.erase: List[float] = []
        self.last_shown: Optional[float] = None
        self.gaps: List[float] = []
        self.detect: List[float] = []
        self.render: List[float] = []
        self.dancers: List[int] = []

    def frame_shown(self, cam: PartyCam, show_time: float) -> None:
        self.show.append(show_time)
        now = time.time()
        if self.last_shown is not None:
            self.gaps.append(now - self.last_shown)
        self.last_shown = now
        t = cam.timings
        self.detect.append(t.get("detect", 0.0))
        self.pose.append(t.get("pose", 0.0))
        if t.get("seg_wait", 0.0) > 0:
            self.seg_wait.append(t["seg_wait"])
        self.render.append(t.get("render", 0.0))
        self.erase.append(t.get("background", 0.0))
        self.dancers.append(cam.dancer_count)
        elapsed = now - self.start
        if elapsed < self.every or not self.gaps:
            return
        frame_id, seq = self.stream.read()[0], self.worker.get()[0]
        gaps = np.array(self.gaps) * 1000
        print(
            f"[perf] camera {(frame_id - self.start_id) / elapsed:4.1f} fps -> analyzed {(seq - self.start_seq) / elapsed:4.1f}"
            f" -> shown {len(self.gaps) / elapsed:4.1f} fps | gaps {np.median(gaps):3.0f}/{np.percentile(gaps, 95):3.0f} ms"
            f" | detect {np.mean(self.detect) * 1000:3.0f} (pose {np.mean(self.pose) * 1000:3.0f},"
            f" seg wait {np.mean(self.seg_wait or [0]) * 1000:3.0f}), erase {np.mean(self.erase) * 1000:3.0f},"
            f" draw {np.mean(self.render) * 1000:3.0f},"
            f" show {np.mean(self.show) * 1000:3.0f}/{np.max(self.show) * 1000:3.0f} ms | skeletons {max(self.dancers)}"
            f" | thermal {thermal_state()}",
            flush=True,
        )
        self.start, self.start_id, self.start_seq = now, frame_id, seq
        self.gaps, self.detect, self.render, self.dancers, self.show = [], [], [], [], []
        self.pose, self.seg_wait, self.erase = [], [], []


def draw_hud(img: np.ndarray, cam: PartyCam, fps: float) -> None:
    lines = [
        f"{fps:4.1f} fps | {cam.dancer_count} skeletons | floor learned {cam.cloak.coverage * 100:3.0f}%",
        f"style={PALETTE_NAMES[cam.palette_idx]} bg={cam.bg_mode} trails={TRAIL_PRESETS[cam.trail_idx]} zones={len(cam.zones.zones)}",
        "1-4 bg | s skull | r ribs | t trail | c style | j banner | b relearn floor",
        "z ignore zone | x clear zones | m mirror | f fullscreen | q quit",
    ]
    for i, text in enumerate(lines):
        y = 28 + i * 26
        cv2.putText(img, text, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(img, text, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 255, 200), 1, cv2.LINE_AA)
    w, h = img.shape[1], img.shape[0]
    for z in cam.zones.pixel_boxes(w, h):
        x1, x2 = (w - z[2], w - z[0]) if cam.mirror else (z[0], z[2])
        cv2.rectangle(img, (int(x1), int(z[1])), (int(x2), int(z[3])), (0, 0, 255), 2)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Halloween Dance Party Skeleton Cam")
    p.add_argument("--source", help="Camera index ('0' = webcam), RTSP URL, or video file. Default: camera from .env")
    p.add_argument("--pose-model", default="yolo11s-pose.pt", help="Ultralytics pose weights (downloaded to models/)")
    p.add_argument("--seg-model", default="yolo11s-seg.pt", help="Ultralytics segmentation weights (downloaded to models/)")
    p.add_argument("--imgsz", type=int, default=1280, help="Pose inference size; bigger finds smaller/farther/partly hidden dancers")
    p.add_argument("--seg-imgsz", type=int, default=960, help="Segmentation inference size; bigger erases more of a packed crowd")
    p.add_argument("--conf", type=float, default=0.25, help="Person detection confidence threshold")
    p.add_argument("--seg-every", type=int, default=2, help="Run person segmentation every N frames (1 = every frame, slower)")
    p.add_argument("--no-gpu-warm", action="store_true",
                   help="Let the GPU idle between frames: saves power on battery, but macOS then slows it ~30%%")  # fmt: skip
    p.add_argument("--width", type=int, default=1920, help="Processing/output width (camera frames are downscaled to this)")
    p.add_argument("--room-brightness", type=float, default=0.85, help="Room brightness in invisible mode (0-1)")
    p.add_argument("--fullscreen", action="store_true",
                   help="Start fullscreen (on the display the window opens on; with the lid closed, the TV)")  # fmt: skip
    p.add_argument("--fps", type=float, default=30.0,
                   help="Output frame rate; skeleton poses are interpolated between camera frames (0 = camera rate)")  # fmt: skip
    p.add_argument("--mirror", action="store_true", help="Start mirrored")
    p.add_argument("--no-banner", action="store_true", help="Start with the 'join the dance floor' banner off")
    p.add_argument("--banner-every", type=float, default=30.0, help="Show a banner message about every N seconds")
    p.add_argument("--fresh-background", action="store_true", help="Forget the saved empty-floor image")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    load_env()

    source = args.source or rtsp_url_from_env() or "0"
    source_val: str | int = int(source) if source.isdigit() else source

    # Lazy import torch so --help is instant
    import torch

    device = "mps" if torch.backends.mps.is_available() else ("0" if torch.cuda.is_available() else "cpu")
    keep_process_fast()
    MODELS_DIR.mkdir(exist_ok=True)
    cam = PartyCam(args, device, source)

    print(f"[*] Connecting to video stream: {redact(source)}")
    stream = LowLatencyStreamReader(source_val)

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, 1280, 720)

    worker = DetectionWorker(cam, stream)
    worker.start()

    show_hud = False
    last_seq = 0
    last_save = time.time()
    fps, frames, fps_t = 0.0, 0, time.time()
    analysis: Optional[Analysis] = None
    stats = PerfLog(stream, worker)
    started_fullscreen = False
    # Output frames are drawn on their own clock (--fps), interpolating poses between camera frames
    interp = Interpolator()
    frame_dt = 1.0 / args.fps if args.fps > 0 else 0.0
    next_out = time.time()
    new_analysis = False

    # Quit cleanly (saving the learned floor) on `kill` too, not just on 'q'
    stop_requested = threading.Event()
    signal.signal(signal.SIGTERM, lambda *_: stop_requested.set())

    print("[*] Party Cam is LIVE! Press 'h' for help, 'f' for fullscreen, 'q' to quit.", flush=True)
    try:
        while not stop_requested.is_set():
            seq, latest = worker.get()
            if latest is not None and seq != last_seq:
                last_seq = seq
                analysis = latest
                interp.push(analysis, time.time())  # people already erased by the worker
                new_analysis = True

            now = time.time()
            due = new_analysis if frame_dt == 0 else (interp.cur is not None and now >= next_out)
            if due:
                new_analysis = False
                if frame_dt:
                    # Don't build up a backlog if drawing falls behind (e.g. a huge crowd)
                    next_out = max(next_out + frame_dt, now - frame_dt)
                    bg_analysis, dancers = interp.pose_at(now)
                else:
                    bg_analysis, dancers = analysis, analysis.dancers
                out = cam.render(bg_analysis, now, dancers)
                frames += 1
                if time.time() - fps_t >= 2.0:
                    fps, frames, fps_t = frames / (time.time() - fps_t), 0, time.time()
                display = out
                if show_hud:
                    display = out.copy()
                    draw_hud(display, cam, fps)
                t_show = time.perf_counter()
                cv2.imshow(WINDOW_NAME, display)
                stats.frame_shown(cam, time.perf_counter() - t_show)
                if args.fullscreen and not started_fullscreen:
                    # After the first frame, so the window exists. With the lid closed the TV is the
                    # only display, so this lands there, also after the launcher restarts the app.
                    cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
                    started_fullscreen = True

            if time.time() - last_save > 60:
                cam.cloak.save(background=True)
                last_save = time.time()

            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break
            elif key in (ord("1"), ord("2"), ord("3"), ord("4")):
                cam.bg_mode = int(chr(key))
            elif key == ord("s"):
                cam.show_skulls = not cam.show_skulls
            elif key == ord("r"):
                cam.show_ribs = not cam.show_ribs
            elif key == ord("t"):
                cam.trail_idx = (cam.trail_idx + 1) % len(TRAIL_PRESETS)
            elif key == ord("c"):
                cam.palette_idx = (cam.palette_idx + 1) % len(PALETTE_NAMES)
            elif key == ord("j"):
                cam.show_banner = not cam.show_banner
            elif key == ord("h"):
                show_hud = not show_hud
            elif key == ord("b"):
                cam.cloak.reset()
                print("[*] Re-learning the empty dance floor")
            elif key == ord("m"):
                cam.mirror = not cam.mirror
            elif key == ord("z") and analysis is not None:
                # Drawn on the un-mirrored camera view, where zones live
                ref = analysis.frame
                x, y, w, h = cv2.selectROI(WINDOW_NAME, ref, showCrosshair=False)
                if w > 0 and h > 0:
                    cam.zones.add(x, y, w, h, ref.shape[1], ref.shape[0])
                    print(f"[*] Added ignore zone ({len(cam.zones.zones)} total)")
            elif key == ord("x"):
                cam.zones.clear()
                print("[*] Cleared ignore zones")
            elif key == ord("f"):
                # OpenCV uses macOS native fullscreen, on whichever display the window is on
                is_full = cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN) == cv2.WINDOW_FULLSCREEN
                cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL if is_full else cv2.WINDOW_FULLSCREEN)
    except KeyboardInterrupt:
        pass  # Ctrl-C in the terminal: quit quietly, cleanup below still runs
    finally:
        worker.stop()
        cam.cloak.save()
        cam.close()
        stream.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
