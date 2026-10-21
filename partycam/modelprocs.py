"""Runs the pose and person-segmentation models in their own processes.

Why processes and not threads: PyTorch's Metal (MPS) backend isn't thread-safe, so two
threads can't run models at once. And with a model in a thread, the drawing code on the main
thread fights it for Python's GIL, which made detection randomly ~15 ms slower. Each process
gets its own GPU context and its own interpreter.

Each camera frame is written once into shared memory that both processes read. Segmentation
returns one combined person mask (half resolution) through a second shared buffer; pose
returns its small keypoint arrays over the pipe.

GPU warm-up: between camera frames the GPU sits idle for a few ms, and macOS responds by
lowering its clock, after which every frame takes ~30% longer (a steady 45 ms became 60 ms
after half a minute). So once both models are done with a frame, the app tells the pose
process it's idle, and it runs tiny ~1.5 ms GPU jobs until the next frame arrives. That keeps
the clock up without slowing real work.
"""

import multiprocessing as mp
import signal
from multiprocessing import shared_memory
from typing import Any, List, Optional, Tuple

import cv2
import numpy as np

MAX_SHAPE = (2160, 3840, 3)  # largest frame the shared buffers can hold (4K)
POSE, SEGMENT = "pose", "segment"
WARM = "warm"  # message: both models are done with the frame, keep the GPU busy until the next

Poses = Tuple[np.ndarray, np.ndarray, np.ndarray]  # keypoints (N,17,3), boxes (N,4), track ids (N,; -1 = none)


def half_shape(h: int, w: int) -> Tuple[int, int]:
    return h // 2, w // 2


def detect_poses(model, frame: np.ndarray, imgsz: int, conf: float, device: str) -> Poses:
    """Pose + ByteTrack tracking. Shared by the pose process and the in-process fallback."""
    r = model.track(frame, persist=True, imgsz=imgsz, conf=conf, device=device, verbose=False, tracker="bytetrack.yaml")[0]
    if r.keypoints is None or r.boxes is None or not len(r.boxes):
        return np.zeros((0, 17, 3), np.float32), np.zeros((0, 4), np.float32), np.zeros(0, np.int64)
    ids = r.boxes.id.int().cpu().numpy() if r.boxes.id is not None else np.full(len(r.boxes), -1)
    return r.keypoints.data.cpu().numpy(), r.boxes.xyxy.cpu().numpy(), ids


def segment_people(model, frame: np.ndarray, imgsz: int, conf: float, device: str, zone_boxes, out: np.ndarray) -> int:
    """Fills `out` (half resolution) with 255 wherever a person is. Returns how many were found.
    Shared by the segmentation process and the in-process fallback."""
    from .zones import in_any_zone

    out.fill(0)
    r = model.predict(frame, classes=[0], imgsz=imgsz, conf=conf, device=device, verbose=False)[0]
    if r.masks is None:
        return 0
    found = 0
    for box, poly in zip(r.boxes.xyxy.cpu().numpy(), r.masks.xy):
        if len(poly) >= 3 and not in_any_zone(box, zone_boxes):
            cv2.fillPoly(out, [(poly * 0.5).astype(np.int32)], 255)
            found += 1
    return found


class SharedFrame:
    """The current camera frame, in shared memory both model processes read from."""

    def __init__(self) -> None:
        h, w, c = MAX_SHAPE
        self.shm = shared_memory.SharedMemory(create=True, size=h * w * c)
        self.shape: Optional[Tuple[int, int]] = None

    def write(self, frame: np.ndarray) -> Tuple[int, int]:
        h, w = frame.shape[:2]
        if h * w * 3 > self.shm.size:
            raise ValueError(f"frame {w}x{h} is larger than the shared buffer")
        np.ndarray((h, w, 3), np.uint8, self.shm.buf)[:] = frame
        self.shape = (h, w)
        return self.shape

    def close(self) -> None:
        self.shm.close()
        self.shm.unlink()


def _gpu_warmer(device: str):
    """Returns a function that does ~1.5 ms of GPU work, or None if there's no GPU."""
    import torch

    if device != "mps":
        return None
    a = torch.randn(1536, 1536, device=device)

    def warm() -> None:
        _ = a @ a
        torch.mps.synchronize()

    return warm


def _serve(
    conn, kind: str, model_path: str, imgsz: int, conf: float, device: str, frame_name: str, mask_name: Optional[str], warm_gpu: bool
) -> None:
    from ultralytics import YOLO

    from .macos import keep_process_fast

    # Ctrl-C in the terminal reaches this process too; let the main app shut it down instead
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    keep_process_fast(f"Party cam {kind} model")
    frame_shm = shared_memory.SharedMemory(name=frame_name)
    mask_shm = shared_memory.SharedMemory(name=mask_name) if mask_name else None
    try:
        model = YOLO(model_path)
        model.predict(np.zeros((1080, 1920, 3), np.uint8), imgsz=imgsz, device=device, verbose=False)  # warm up
        warm = _gpu_warmer(device) if warm_gpu else None
        warming = False
        conn.send("ready")
        while True:
            if warming and not conn.poll(0):
                warm()  # idle between frames: keep the GPU clock up until the next one arrives
                continue
            msg = conn.recv()
            if msg is None:
                break
            if msg == WARM:
                warming = warm is not None
                continue
            warming = False
            (h, w), zone_boxes = msg
            frame = np.ndarray((h, w, 3), np.uint8, frame_shm.buf)
            if kind == SEGMENT:
                out = np.ndarray(half_shape(h, w), np.uint8, mask_shm.buf)
                conn.send(segment_people(model, frame, imgsz, conf, device, zone_boxes, out))
            else:
                conn.send(detect_poses(model, frame, imgsz, conf, device))
    except (EOFError, KeyboardInterrupt):
        pass
    finally:
        frame_shm.close()
        if mask_shm is not None:
            mask_shm.close()


class ModelProcess:
    """One model in its own process. Per frame: frame.write() once, then submit() to each
    process, do other work, then result()."""

    def __init__(
        self, kind: str, model_path: str, imgsz: int, conf: float, device: str, frame: SharedFrame, warm_gpu: bool = False
    ):
        self.kind = kind
        self.frame = frame
        ctx = mp.get_context("spawn")
        self.mask_shm = None
        if kind == SEGMENT:
            h, w, _ = MAX_SHAPE
            self.mask_shm = shared_memory.SharedMemory(create=True, size=(h // 2) * (w // 2))
        self.conn, child_conn = ctx.Pipe()
        self.proc = ctx.Process(
            target=_serve,
            args=(child_conn, kind, model_path, imgsz, conf, device, frame.shm.name,
                  self.mask_shm.name if self.mask_shm else None, warm_gpu),  # fmt: skip
            daemon=True,
        )
        self.proc.start()
        self.pending: Optional[Tuple[int, int]] = None

    def wait_ready(self, timeout: float = 90.0) -> None:
        if not self.conn.poll(timeout) or self.conn.recv() != "ready":
            self.close()
            raise RuntimeError(f"{self.kind} process didn't start")

    def submit(self, zone_boxes: List[np.ndarray]) -> None:
        """Asks the process to work on the frame currently in the shared buffer."""
        self.pending = self.frame.shape
        self.conn.send((self.pending, zone_boxes))

    def idle(self) -> None:
        """Tells a warm_gpu process the frame is fully done, so it can keep the GPU busy."""
        self.conn.send(WARM)

    def result(self, timeout: float = 5.0) -> Any:
        if self.pending is None or not self.conn.poll(timeout):
            raise RuntimeError(f"{self.kind} process stopped responding")
        value = self.conn.recv()
        h, w = self.pending
        self.pending = None
        if self.kind == SEGMENT:
            return np.ndarray(half_shape(h, w), np.uint8, self.mask_shm.buf).copy()
        return value

    def close(self) -> None:
        try:
            if self.proc.is_alive():
                self.conn.send(None)
                self.proc.join(timeout=3.0)
            if self.proc.is_alive():
                self.proc.terminate()
        except (BrokenPipeError, OSError):
            pass
        if self.mask_shm is not None:
            self.mask_shm.close()
            self.mask_shm.unlink()
            self.mask_shm = None
