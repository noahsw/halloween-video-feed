"""Low-latency camera reader."""

import os
import threading
import time
from typing import Optional, Tuple

import cv2
import numpy as np

from .config import redact
from .macos import keep_thread_fast

# Force low-latency TCP transport options for OpenCV's FFmpeg RTSP backend
os.environ.setdefault(
    "OPENCV_FFMPEG_CAPTURE_OPTIONS",
    "rtsp_transport;tcp|fflags;nobuffer|flags;low_delay|max_delay;500000",
)


class LowLatencyStreamReader:
    """Continuously drains the RTSP/webcam buffer on a background thread so
    pose inference always runs on the freshest frame with zero lag buildup."""

    def __init__(self, source: str | int):
        self.source = source
        self.cap: Optional[cv2.VideoCapture] = None
        self.frame: Optional[np.ndarray] = None
        self.frame_id = 0
        self.running = True
        self.lock = threading.Lock()
        self._connect()
        self.thread = threading.Thread(target=self._update_loop, daemon=True)
        self.thread.start()

    def _connect(self) -> None:
        if self.cap is not None:
            self.cap.release()
        if isinstance(self.source, int) or str(self.source).isdigit():
            self.cap = cv2.VideoCapture(int(self.source))
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            self.cap.set(cv2.CAP_PROP_FPS, 30)
        else:
            self.cap = cv2.VideoCapture(self.source, cv2.CAP_FFMPEG)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if not self.cap.isOpened():
            print(f"[!] Could not open {redact(self.source)}; retrying...")

    def _update_loop(self) -> None:
        keep_thread_fast()
        consecutive_failures = 0
        # Video files (handy for testing) play back at their own frame rate and loop
        is_file = os.path.isfile(str(self.source))
        file_dt = 1.0 / (self.cap.get(cv2.CAP_PROP_FPS) or 30.0) if is_file and self.cap else 0.0
        next_t = time.time()
        while self.running:
            if is_file:
                time.sleep(max(0.0, next_t - time.time()))
                next_t = max(next_t + file_dt, time.time() - file_dt)
            if self.cap is None or not self.cap.isOpened():
                time.sleep(1.0)
                self._connect()
                continue

            ok, img = self.cap.read()
            if ok and img is not None:
                consecutive_failures = 0
                with self.lock:
                    self.frame = img
                    self.frame_id += 1
            elif is_file:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            else:
                consecutive_failures += 1
                if consecutive_failures > 30:
                    # Auto-reconnect if Wi-Fi or RTSP stream blips during the party
                    consecutive_failures = 0
                    time.sleep(0.5)
                    self._connect()
                else:
                    time.sleep(0.01)

    def read(self) -> Tuple[int, Optional[np.ndarray]]:
        """Returns (frame_id, frame). frame_id only increases when a new frame arrives."""
        with self.lock:
            return self.frame_id, self.frame

    def stop(self) -> None:
        self.running = False
        self.thread.join(timeout=2.0)
        if self.cap is not None:
            self.cap.release()
