"""Party-night launcher: runs the party cam and brings it back if it crashes or freezes.

    uv run party-night

Any arguments are passed straight to party-cam. Quitting with 'q' (or Ctrl-C here) stops for
good; a crash, or no [perf] heartbeat for a while, restarts it after a short pause. Keeps the
Mac awake the whole time, and logs everything to local/logs/ for a look the morning after.
"""

import os
import shutil
import signal
import subprocess
import sys
import threading
import time
from datetime import datetime

from .config import LOCAL_DIR

HEARTBEAT_TIMEOUT = 45.0  # party-cam prints a [perf] line every 10 s while it's running
STARTUP_GRACE = 120.0  # loading models and connecting can take a while
BACKOFF = [2, 5, 10, 30]  # seconds between restarts, if it keeps failing quickly


def _stop_group(proc: subprocess.Popen, grace: float = 10.0) -> None:
    """Stops the party cam and its model processes (they share its process group)."""
    for sig in (signal.SIGCONT, signal.SIGTERM):  # SIGCONT first, in case it's stopped
        try:
            os.killpg(proc.pid, sig)
        except ProcessLookupError:
            return
    try:
        proc.wait(timeout=grace)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        proc.wait()


def _stamp() -> str:
    return datetime.now().strftime("%H:%M:%S")


def main() -> None:
    log_dir = LOCAL_DIR / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"party_{datetime.now():%Y%m%d_%H%M%S}.log"
    log = open(log_path, "a", buffering=1)

    def say(msg: str) -> None:
        line = f"[launcher {_stamp()}] {msg}"
        print(line, flush=True)
        log.write(line + "\n")

    cmd = [sys.executable, "-u", "-m", "partycam.app", *sys.argv[1:]]
    if shutil.which("caffeinate"):
        cmd = ["caffeinate", "-dis", *cmd]  # no sleeping, display or system, all night

    stopping = False
    child: subprocess.Popen = None  # type: ignore[assignment]

    def on_signal(signum, _frame) -> None:
        nonlocal stopping
        stopping = True
        if child is not None and child.poll() is None:
            child.send_signal(signal.SIGTERM)  # party-cam saves the learned floor and exits cleanly

    signal.signal(signal.SIGINT, on_signal)
    signal.signal(signal.SIGTERM, on_signal)

    say(f"logging to {log_path.relative_to(LOCAL_DIR.parent)}")
    failures = 0
    while not stopping:
        started = time.time()
        last_beat = [started]
        # Its own process group, so Ctrl-C here reaches only the launcher, which then stops it cleanly
        child = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, start_new_session=True,
            env={**os.environ, "PYTHONUNBUFFERED": "1"},
        )  # fmt: skip

        def pump(proc: subprocess.Popen) -> None:
            for line in proc.stdout:
                sys.stdout.write(line)
                sys.stdout.flush()
                log.write(line)
                if line.startswith("[perf]"):
                    last_beat[0] = time.time()

        reader = threading.Thread(target=pump, args=(child,), daemon=True)
        reader.start()

        hung = False
        while child.poll() is None:
            time.sleep(1.0)
            limit = STARTUP_GRACE if last_beat[0] == started else HEARTBEAT_TIMEOUT
            if not stopping and time.time() - last_beat[0] > limit:
                say(f"no heartbeat for {limit:.0f}s; the party cam looks frozen, restarting it")
                hung = True
                _stop_group(child)
                break
        child.wait()
        reader.join(timeout=2)
        code = child.returncode

        if stopping or (code == 0 and not hung):
            say("party cam stopped. Goodnight!")
            break
        ran_for = time.time() - started
        failures = 0 if ran_for > 120 else failures + 1
        wait = BACKOFF[min(max(failures - 1, 0), len(BACKOFF) - 1)]
        reason = "froze" if hung else f"exited with code {code}"
        say(f"party cam {reason} after {ran_for / 60:.1f} min; restarting in {wait}s")
        for _ in range(wait):
            if stopping:
                say("party cam stopped. Goodnight!")
                break
            time.sleep(1)
    log.close()


if __name__ == "__main__":
    main()
