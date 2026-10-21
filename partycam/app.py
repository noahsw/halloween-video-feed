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
    uv run party-cam --display "TV"       # fullscreen on the display whose name contains "TV"

Live Party Hotkeys (while window is focused):
    1 / 2 / 3 / 4 : Background (1 = Invisible dancers, 2 = Dimmed Room, 3 = Pitch-Black Void, 4 = Spectral Night-Vision)
    s         : Toggle Skull Heads on/off
    r         : Toggle Ribcage on/off
    t         : Cycle Phantom Trail persistence (Medium -> Long Ghost Echo -> Short -> Off)
    c         : Cycle Skeleton Style (Bone, Multi-Dancer Neon, Ecto-Green, Phantom Cyan/Purple, Hellfire Orange)
    j         : Toggle "Join the dance floor" banner
    b         : Re-learn the empty dance floor (use if the camera got bumped)
    z         : Draw an ignore zone (e.g. around a skeleton prop); x clears all zones
    m         : Toggle Horizontal Mirroring
    f         : Toggle Fullscreen;  d : Move fullscreen to the next display
    h         : Toggle help / FPS overlay
    q or ESC  : Quit
"""

import argparse
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
from .cloak import InvisibilityCloak
from .config import LOCAL_DIR, MODELS_DIR, load_env, redact, rtsp_url_from_env
from .display import FullscreenController, find_screen, screen_names
from .skeleton import PALETTE_NAMES, SkeletonCanvas, draw_dancers, estimate_head
from .stream import LowLatencyStreamReader
from .tracking import CONF_THRESH, Dancer, PoseSmoother
from .zones import IgnoreZones

WINDOW_NAME = "Halloween Skeleton Party Cam"

BG_INVISIBLE, BG_DIM, BG_VOID, BG_SPECTRAL = 1, 2, 3, 4
TRAIL_PRESETS = [0.80, 0.90, 0.65, 0.0]  # Medium, Long Ghost Echo, Short, Off

# Limb segments painted into the person mask, on top of the segmentation mask, so
# fast-moving arms the segmenter misses still get erased.
MASK_LIMBS = [(5, 7), (7, 9), (6, 8), (8, 10), (11, 13), (13, 15), (12, 14), (14, 16), (5, 6), (11, 12), (5, 11), (6, 12)]


@dataclass
class Analysis:
    frame: np.ndarray
    dancers: List[Dancer]
    segments: list  # person outline polygons from the segmentation model


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
        self.pose_model = YOLO(str(MODELS_DIR / args.pose_model))
        self.seg_model = YOLO(str(MODELS_DIR / args.seg_model))

        self.smoother = PoseSmoother()
        # One saved floor image per video source, so a webcam test never pollutes the party camera's
        source_key = hashlib.sha1(redact(source).encode()).hexdigest()[:10]
        self.cloak = InvisibilityCloak(LOCAL_DIR / f"floor_{source_key}.png")
        if args.fresh_background:
            self.cloak.plate_path.unlink(missing_ok=True)
        self.zones = IgnoreZones(LOCAL_DIR / "ignore_zones.json")
        self.banner = JoinBanner()

        self.bg_mode = BG_INVISIBLE
        self.show_skulls = True
        self.show_ribs = True
        self.show_banner = not args.no_banner
        self.mirror = args.mirror
        self.palette_idx = 0
        self.trail_idx = 0

        self.trail: Optional[np.ndarray] = None
        self.prev_mask: Optional[np.ndarray] = None
        self.prev_gray: Optional[np.ndarray] = None
        self.dancer_count = 0
        self.timings: dict = {}

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

    def detect(self, frame: np.ndarray) -> Tuple[List[Dancer], list]:
        h, w = frame.shape[:2]
        zone_boxes = self.zones.pixel_boxes(w, h)

        r = self.pose_model.track(
            frame, persist=True, imgsz=self.args.imgsz, conf=self.args.conf,
            device=self.device, verbose=False, tracker="bytetrack.yaml",
        )[0]  # fmt: skip
        dancers: List[Dancer] = []
        if r.keypoints is not None and r.boxes is not None and len(r.boxes):
            kpts = r.keypoints.data.cpu().numpy()
            boxes = r.boxes.xyxy.cpu().numpy()
            ids = r.boxes.id.int().cpu().tolist() if r.boxes.id is not None else [None] * len(boxes)
            for k, b, tid in zip(kpts, boxes, ids):
                if not self.zones.should_ignore(b, zone_boxes):
                    dancers.append(Dancer(kpts=k, box=b, track_id=tid))
        dancers = self.smoother.update(dancers)

        segments = []
        if self.bg_mode == BG_INVISIBLE or self.cloak.coverage < 0.97:
            s = self.seg_model.predict(
                frame, classes=[0], imgsz=self.args.seg_imgsz, conf=0.25, device=self.device, verbose=False
            )[0]
            if s.masks is not None:
                for b, poly in zip(s.boxes.xyxy.cpu().numpy(), s.masks.xy):
                    if len(poly) >= 3 and not self.zones.should_ignore(b, zone_boxes):
                        segments.append(poly)
        return dancers, segments

    def person_mask(self, shape, dancers: List[Dancer], segments: list) -> Tuple[np.ndarray, np.ndarray]:
        """Returns (binary mask, feathered float alpha) covering everyone, at full resolution.
        Built at half resolution since it gets dilated and blurred anyway."""
        h, w = shape[:2]
        sh, sw = h // 2, w // 2
        mask = np.zeros((sh, sw), np.uint8)
        for poly in segments:
            cv2.fillPoly(mask, [(poly * 0.5).astype(np.int32)], 255)
        for d in dancers:
            th = max(2, int(d.height * 0.065))
            k = d.kpts
            for a, b in MASK_LIMBS:
                if k[a, 2] >= CONF_THRESH and k[b, 2] >= CONF_THRESH:
                    cv2.line(mask, (int(k[a, 0] / 2), int(k[a, 1] / 2)), (int(k[b, 0] / 2), int(k[b, 1] / 2)), 255, th)
            head = estimate_head(k)
            if head is not None:
                c, rad, _ = head
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

    def analyze(self, raw: np.ndarray) -> Analysis:
        frame = self.prepare(raw)
        t0 = time.perf_counter()
        dancers, segments = self.detect(frame)
        self.timings["detect"] = time.perf_counter() - t0
        return Analysis(frame, dancers, segments)

    def render(self, a: Analysis, now: Optional[float] = None) -> np.ndarray:
        now = time.time() if now is None else now
        t0 = time.perf_counter()
        frame, dancers = a.frame, a.dancers
        self.dancer_count = len(dancers)
        if self.trail is None or self.trail.shape != frame.shape:
            self.trail = np.zeros_like(frame)

        need_gray = self.bg_mode != BG_INVISIBLE
        gray_blur = cv2.GaussianBlur(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (9, 9), 0) if need_gray else None
        mask, alpha = self.person_mask(frame.shape, dancers, a.segments)
        bg = self.background(frame, mask, alpha, gray_blur)
        t1 = time.perf_counter()

        canvas = SkeletonCanvas(frame.shape)
        palette = PALETTE_NAMES[self.palette_idx]
        draw_dancers(canvas, dancers, palette, self.show_skulls, self.show_ribs)

        # Ghostly ectoplasm shimmer where things move (not in invisible mode: it would outline people)
        if gray_blur is not None and self.prev_gray is not None and self.prev_gray.shape == gray_blur.shape:
            _, motion = cv2.threshold(cv2.absdiff(gray_blur, self.prev_gray), 22, 255, cv2.THRESH_BINARY)
            canvas.glow[cv2.Canny(motion, 50, 150) > 0] = (90, 45, 15)
        self.prev_gray = gray_blur

        out = self.composite(bg, canvas, palette)
        # Everything above runs un-mirrored so the learned floor and ignore zones never depend on it
        if self.mirror:
            out = cv2.flip(out, 1)
        if self.show_banner:
            self.banner.draw(out, now)
        t2 = time.perf_counter()
        self.timings.update(background=t1 - t0, render=t2 - t1)
        return out

    def process(self, raw: np.ndarray, now: Optional[float] = None) -> np.ndarray:
        return self.render(self.analyze(raw), now)

    def composite(self, bg: np.ndarray, canvas: SkeletonCanvas, palette: str) -> np.ndarray:
        h, w = bg.shape[:2]
        # Cheap wide bloom: blur at quarter resolution
        small = cv2.resize(canvas.glow, (w // 4, h // 4), interpolation=cv2.INTER_AREA)
        small = cv2.GaussianBlur(small, (0, 0), 2.5)
        bloom = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
        glow_gain = 0.55 if palette == "bone" else 1.35
        out = cv2.addWeighted(bg, 1.0, bloom, glow_gain, 0)

        decay = TRAIL_PRESETS[self.trail_idx]
        if decay > 0:
            # Phantom trails drift upward like rising spirits
            self.trail[:-2] = self.trail[2:]
            self.trail[-2:] = 0
            ghost = cv2.addWeighted(canvas.color, 0.30, bloom, 0.25, 0)
            self.trail = cv2.addWeighted(self.trail, decay, ghost, 1.0, 0)
            out = cv2.add(out, cv2.convertScaleAbs(self.trail, alpha=0.6))
        else:
            self.trail.fill(0)

        inv = cv2.merge([cv2.bitwise_not(canvas.alpha)] * 3)
        out = cv2.multiply(out, inv, scale=1.0 / 255.0)
        return cv2.add(out, canvas.color)


class DetectionWorker(threading.Thread):
    """Runs pose + segmentation on the newest camera frame while the main thread renders."""

    def __init__(self, cam: PartyCam, stream: LowLatencyStreamReader):
        super().__init__(daemon=True)
        self.cam = cam
        self.stream = stream
        self.running = True
        self.lock = threading.Lock()
        self.latest: Optional[Analysis] = None
        self.seq = 0

    def run(self) -> None:
        last_id = -1
        while self.running:
            frame_id, raw = self.stream.read()
            if raw is None or frame_id == last_id:
                time.sleep(0.002)
                continue
            last_id = frame_id
            try:
                result = self.cam.analyze(raw)
            except Exception:
                traceback.print_exc()
                time.sleep(0.5)
                continue
            with self.lock:
                self.latest = result
                self.seq += 1

    def get(self) -> Tuple[int, Optional[Analysis]]:
        with self.lock:
            return self.seq, self.latest

    def stop(self) -> None:
        self.running = False
        self.join(timeout=2.0)


def draw_hud(img: np.ndarray, cam: PartyCam, fps: float) -> None:
    lines = [
        f"{fps:4.1f} fps | {cam.dancer_count} skeletons | floor learned {cam.cloak.coverage * 100:3.0f}%",
        f"style={PALETTE_NAMES[cam.palette_idx]} bg={cam.bg_mode} trails={TRAIL_PRESETS[cam.trail_idx]} zones={len(cam.zones.zones)}",
        "1-4 bg | s skull | r ribs | t trail | c style | j banner | b relearn floor",
        "z ignore zone | x clear zones | m mirror | f fullscreen | d next display | q quit",
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
    p.add_argument("--imgsz", type=int, default=960, help="Pose inference size; bigger finds smaller/farther dancers")
    p.add_argument("--seg-imgsz", type=int, default=640, help="Segmentation inference size")
    p.add_argument("--conf", type=float, default=0.30, help="Person detection confidence threshold")
    p.add_argument("--width", type=int, default=1920, help="Processing/output width (camera frames are downscaled to this)")
    p.add_argument("--room-brightness", type=float, default=0.85, help="Room brightness in invisible mode (0-1)")
    p.add_argument("--display", help="Start fullscreen on this display (index or name substring, e.g. 'TV')")
    p.add_argument("--list-displays", action="store_true", help="List displays and exit")
    p.add_argument("--mirror", action="store_true", help="Start mirrored")
    p.add_argument("--no-banner", action="store_true", help="Start with the 'join the dance floor' banner off")
    p.add_argument("--fresh-background", action="store_true", help="Forget the saved empty-floor image")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    load_env()

    if args.list_displays:
        for i, name in enumerate(screen_names()):
            print(f"  {i}: {name}")
        return

    source = args.source or rtsp_url_from_env() or "0"
    source_val: str | int = int(source) if source.isdigit() else source

    # Lazy import torch so --help and --list-displays are instant
    import torch

    device = "mps" if torch.backends.mps.is_available() else ("0" if torch.cuda.is_available() else "cpu")
    MODELS_DIR.mkdir(exist_ok=True)
    cam = PartyCam(args, device, source)

    print(f"[*] Connecting to video stream: {redact(source)}")
    stream = LowLatencyStreamReader(source_val)

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, 1280, 720)
    fullscreen = FullscreenController(WINDOW_NAME)
    pending_display = None
    if args.display is not None:
        pending_display = find_screen(args.display)
        if pending_display is None:
            print(f"[!] No display matching '{args.display}'. Available: {screen_names()}")

    worker = DetectionWorker(cam, stream)
    worker.start()

    show_hud = False
    last_seq = 0
    last_save = time.time()
    fps, frames, fps_t = 0.0, 0, time.time()
    analysis: Optional[Analysis] = None

    # Quit cleanly (saving the learned floor) on `kill` too, not just on 'q'
    stop_requested = threading.Event()
    signal.signal(signal.SIGTERM, lambda *_: stop_requested.set())

    print("[*] Party Cam is LIVE! Press 'h' for help, 'f' for fullscreen, 'q' to quit.", flush=True)
    try:
        while not stop_requested.is_set():
            seq, analysis = worker.get()
            if analysis is not None and seq != last_seq:
                last_seq = seq
                out = cam.render(analysis)
                frames += 1
                if time.time() - fps_t >= 2.0:
                    fps, frames, fps_t = frames / (time.time() - fps_t), 0, time.time()
                display = out
                if show_hud:
                    display = out.copy()
                    draw_hud(display, cam, fps)
                cv2.imshow(WINDOW_NAME, display)
                if pending_display is not None:
                    fullscreen.enter(pending_display)
                    pending_display = None

            if time.time() - last_save > 60:
                cam.cloak.save()
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
                if fullscreen.is_fullscreen:
                    fullscreen.exit()
                else:
                    fullscreen.enter(fullscreen.screen_index or 0)
            elif key == ord("d"):
                idx = fullscreen.cycle()
                if idx is not None:
                    print(f"[*] Fullscreen on display {idx}: {screen_names()[idx]}")
    finally:
        worker.stop()
        cam.cloak.save()
        stream.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
