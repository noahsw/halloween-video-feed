"""Keep macOS from throttling the party cam.

macOS slows down work it thinks is in the background: App Nap, timer coalescing, and lower
CPU/GPU scheduling for processes and threads without a "user interactive" quality of service.
Unchecked, that shows up as detection randomly taking twice as long for half a minute at a
time, and choppy video. A latency-critical activity plus interactive thread QoS opts out.
The same activity keeps the system and the display from idle-sleeping: with the lid closed the TV
is the only display, and it would otherwise go black after a few minutes without input.
"""

import ctypes
import sys
from typing import Any, Optional

# From Foundation's NSProcessInfo.h
NS_ACTIVITY_USER_INITIATED = 0x00FFFFFF | (1 << 20)  # includes NSActivityIdleSystemSleepDisabled
NS_ACTIVITY_LATENCY_CRITICAL = 0xFF00000000
NS_ACTIVITY_IDLE_DISPLAY_SLEEP_DISABLED = 1 << 40
# From <sys/qos.h>
QOS_CLASS_USER_INTERACTIVE = 0x21

_activity: Optional[Any] = None  # must stay referenced, or the activity ends


def keep_process_fast(reason: str = "Party cam is live") -> None:
    """Begins a latency-critical activity for this process and raises the calling thread's QoS."""
    global _activity
    if sys.platform != "darwin":
        return
    if _activity is None:
        try:
            from Foundation import NSProcessInfo  # type: ignore[import-not-found]

            _activity = NSProcessInfo.processInfo().beginActivityWithOptions_reason_(
                NS_ACTIVITY_USER_INITIATED | NS_ACTIVITY_LATENCY_CRITICAL | NS_ACTIVITY_IDLE_DISPLAY_SLEEP_DISABLED, reason
            )
        except Exception as e:  # pyobjc missing: still run, just maybe throttled
            print(f"[!] Couldn't ask macOS not to throttle the party cam: {e}")
    keep_thread_fast()


def thermal_state() -> str:
    """macOS's own thermal verdict: nominal, fair, serious or critical ("?" if unavailable).
    At serious and above, macOS slows the CPU and GPU to cool down."""
    if sys.platform != "darwin":
        return "?"
    try:
        from Foundation import NSProcessInfo  # type: ignore[import-not-found]

        return ("nominal", "fair", "serious", "critical")[int(NSProcessInfo.processInfo().thermalState())]
    except Exception:
        return "?"


def keep_thread_fast() -> None:
    """Marks the calling thread as user-interactive. Call at the start of each worker thread."""
    if sys.platform != "darwin":
        return
    try:
        libsystem = ctypes.CDLL("/usr/lib/libSystem.dylib")
        libsystem.pthread_set_qos_class_self_np(ctypes.c_uint(QOS_CLASS_USER_INTERACTIVE), ctypes.c_int(0))
    except (OSError, AttributeError):
        pass
