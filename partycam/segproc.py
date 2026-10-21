"""Runs the person-segmentation model in its own process, so it uses the GPU at the same
time as the pose model. PyTorch's Metal (MPS) backend isn't thread-safe, so a second thread
can't do this, but a second process gets its own GPU context.

Frames go in and masks come back through shared memory; the pipe only carries tiny messages.
The process returns one combined person mask at half resolution, which is all the
invisibility cloak needs.
"""

import multiprocessing as mp
from multiprocessing import shared_memory
from typing import List, Optional, Tuple

import cv2
import numpy as np

MAX_SHAPE = (2160, 3840, 3)  # largest frame the shared buffers can hold (4K)


def half_shape(h: int, w: int) -> Tuple[int, int]:
    return h // 2, w // 2


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


def _serve(conn, model_path: str, imgsz: int, conf: float, device: str, frame_name: str, mask_name: str) -> None:
    from ultralytics import YOLO

    frame_shm = shared_memory.SharedMemory(name=frame_name)
    mask_shm = shared_memory.SharedMemory(name=mask_name)
    try:
        model = YOLO(model_path)
        model.predict(np.zeros((1080, 1920, 3), np.uint8), imgsz=imgsz, device=device, verbose=False)  # warm up
        conn.send("ready")
        while True:
            msg = conn.recv()
            if msg is None:
                break
            (h, w), zone_boxes = msg
            frame = np.ndarray((h, w, 3), np.uint8, frame_shm.buf)
            out = np.ndarray(half_shape(h, w), np.uint8, mask_shm.buf)
            conn.send(segment_people(model, frame, imgsz, conf, device, zone_boxes, out))
    except (EOFError, KeyboardInterrupt):
        pass
    finally:
        frame_shm.close()
        mask_shm.close()


class SegmentationProcess:
    """Usage per frame: submit(frame) ... run pose meanwhile ... result() -> mask."""

    def __init__(self, model_path: str, imgsz: int, conf: float, device: str, timeout: float = 60.0):
        ctx = mp.get_context("spawn")
        h, w, c = MAX_SHAPE
        self.frame_shm = shared_memory.SharedMemory(create=True, size=h * w * c)
        self.mask_shm = shared_memory.SharedMemory(create=True, size=(h // 2) * (w // 2))
        self.conn, child_conn = ctx.Pipe()
        self.proc = ctx.Process(
            target=_serve,
            args=(child_conn, model_path, imgsz, conf, device, self.frame_shm.name, self.mask_shm.name),
            daemon=True,
        )
        self.proc.start()
        self.shape: Optional[Tuple[int, int]] = None
        if not self.conn.poll(timeout) or self.conn.recv() != "ready":
            self.close()
            raise RuntimeError("segmentation process didn't start")

    @property
    def alive(self) -> bool:
        return self.proc.is_alive()

    def submit(self, frame: np.ndarray, zone_boxes: List[np.ndarray]) -> None:
        h, w = frame.shape[:2]
        if h * w * 3 > self.frame_shm.size:
            raise ValueError(f"frame {w}x{h} is larger than the shared buffer")
        np.ndarray((h, w, 3), np.uint8, self.frame_shm.buf)[:] = frame
        self.shape = (h, w)
        self.conn.send(((h, w), zone_boxes))

    def result(self, timeout: float = 5.0) -> np.ndarray:
        if self.shape is None or not self.conn.poll(timeout):
            raise RuntimeError("segmentation process stopped responding")
        self.conn.recv()
        h, w = self.shape
        self.shape = None
        return np.ndarray(half_shape(h, w), np.uint8, self.mask_shm.buf).copy()

    def close(self) -> None:
        try:
            if self.proc.is_alive():
                self.conn.send(None)
                self.proc.join(timeout=3.0)
            if self.proc.is_alive():
                self.proc.terminate()
        except (BrokenPipeError, OSError):
            pass
        for shm in (self.frame_shm, self.mask_shm):
            shm.close()
            shm.unlink()
