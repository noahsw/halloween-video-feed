"""Camera settings, read from a local .env file that is never committed."""

import os
import re
from pathlib import Path
from typing import Optional
from urllib.parse import quote

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parent.parent
# Per-house state (background plate, ignore zones). Gitignored: it is a picture of your room.
LOCAL_DIR = REPO_ROOT / "local"
MODELS_DIR = REPO_ROOT / "models"


def load_env() -> None:
    load_dotenv(REPO_ROOT / ".env")


def rtsp_url_from_env() -> Optional[str]:
    """Builds the Tapo RTSP URL from CAMERA_* variables, or None if CAMERA_HOST is unset."""
    host = os.getenv("CAMERA_HOST", "").strip()
    if not host:
        return None
    user = os.getenv("CAMERA_USER", "")
    password = os.getenv("CAMERA_PASSWORD", "")
    port = os.getenv("CAMERA_PORT", "554").strip()
    stream = os.getenv("CAMERA_STREAM", "stream1").strip()
    auth = f"{quote(user, safe='')}:{quote(password, safe='')}@" if user else ""
    return f"rtsp://{auth}{host}:{port}/{stream}"


def redact(source: str) -> str:
    """Hides credentials so stream URLs are safe to print."""
    return re.sub(r"//[^/@]+@", "//***:***@", str(source))
