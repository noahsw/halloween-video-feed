"""Record a raw test clip from the camera, for tuning the look offline.

Clips go to local/clips/ (gitignored) and never leave your Mac. Replay one with:
    uv run party-cam --source local/clips/<clip>.mp4
"""

import argparse
import shutil
import subprocess
import sys
import time
from datetime import datetime

import cv2

from .config import LOCAL_DIR, load_env, redact, rtsp_url_from_env


def main() -> None:
    p = argparse.ArgumentParser(description="Record a raw camera clip to local/clips/")
    p.add_argument("--seconds", type=int, default=90)
    p.add_argument("--source", help="RTSP URL (default: camera from .env)")
    args = p.parse_args()

    load_env()
    url = args.source or rtsp_url_from_env()
    if not url:
        sys.exit("No camera configured. Fill in .env (see .env.example) or pass --source.")
    out_dir = LOCAL_DIR / "clips"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"clip_{datetime.now():%Y%m%d_%H%M%S}.mp4"
    print(f"[*] Recording {args.seconds}s from {redact(url)} -> {out.relative_to(LOCAL_DIR.parent)}")

    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg:
        # Copies the camera's own H.264 stream: full quality, almost no CPU. Audio is dropped.
        cmd = [ffmpeg, "-hide_banner", "-loglevel", "error", "-rtsp_transport", "tcp", "-i", url,
               "-t", str(args.seconds), "-an", "-c:v", "copy", str(out)]  # fmt: skip
        result = subprocess.run(cmd, stderr=subprocess.PIPE, text=True)
        if result.returncode != 0:
            sys.exit(f"[!] ffmpeg failed: {redact(result.stderr.strip())}")
    else:
        cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
        if not cap.isOpened():
            sys.exit(f"[!] Could not open {redact(url)}")
        fps = cap.get(cv2.CAP_PROP_FPS) or 20.0
        writer = None
        end = time.time() + args.seconds
        while time.time() < end:
            ok, frame = cap.read()
            if not ok:
                continue
            if writer is None:
                writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), fps, frame.shape[1::-1])
            writer.write(frame)
        cap.release()
        if writer is not None:
            writer.release()
    print(f"[*] Saved {out.relative_to(LOCAL_DIR.parent)}")


if __name__ == "__main__":
    main()
