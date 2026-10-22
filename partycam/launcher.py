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
from typing import Optional

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


class Supervisor:
    """The run / watch / restart loop. Separate from main() so tests can supervise a stand-in app."""

    def __init__(self, cmd, say, log, heartbeat_timeout=HEARTBEAT_TIMEOUT, startup_grace=STARTUP_GRACE,
                 backoff=BACKOFF, poll=1.0):  # fmt: skip
        self.cmd = cmd
        self.say = say
        self.log = log
        self.heartbeat_timeout = heartbeat_timeout
        self.startup_grace = startup_grace
        self.backoff = backoff
        self.poll = poll
        self.stopping = False
        self.child: Optional[subprocess.Popen] = None
        self.restarts = 0

    def stop(self) -> None:
        """Quit for good (Ctrl-C / SIGTERM to the launcher)."""
        self.stopping = True
        if self.child is not None and self.child.poll() is None:
            self.child.send_signal(signal.SIGTERM)  # party-cam saves the learned floor and exits cleanly

    def run(self) -> None:
        failures = 0
        while not self.stopping:
            started = time.time()
            last_beat = [started]
            # Its own process group, so Ctrl-C here reaches only the launcher, which then stops it cleanly
            child = self.child = subprocess.Popen(
                self.cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
                start_new_session=True, env={**os.environ, "PYTHONUNBUFFERED": "1"},
            )  # fmt: skip

            def pump(proc: subprocess.Popen) -> None:
                for line in proc.stdout:
                    sys.stdout.write(line)
                    sys.stdout.flush()
                    self.log.write(line)
                    if line.startswith("[perf]"):
                        last_beat[0] = time.time()

            reader = threading.Thread(target=pump, args=(child,), daemon=True)
            reader.start()

            hung = False
            while child.poll() is None:
                time.sleep(self.poll)
                limit = self.startup_grace if last_beat[0] == started else self.heartbeat_timeout
                if not self.stopping and time.time() - last_beat[0] > limit:
                    self.say(f"no heartbeat for {limit:.0f}s; the party cam looks frozen, restarting it")
                    hung = True
                    _stop_group(child)
                    break
            child.wait()
            reader.join(timeout=2)
            code = child.returncode

            if self.stopping or (code == 0 and not hung):
                self.say("party cam stopped. Goodnight!")
                return
            ran_for = time.time() - started
            failures = 0 if ran_for > 120 else failures + 1
            wait = self.backoff[min(max(failures - 1, 0), len(self.backoff) - 1)]
            reason = "froze" if hung else f"exited with code {code}"
            self.say(f"party cam {reason} after {ran_for / 60:.1f} min; restarting in {wait}s")
            self.restarts += 1
            for _ in range(wait):
                if self.stopping:
                    self.say("party cam stopped. Goodnight!")
                    return
                time.sleep(1)


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

    supervisor = Supervisor(cmd, say, log)
    signal.signal(signal.SIGINT, lambda *_: supervisor.stop())
    signal.signal(signal.SIGTERM, lambda *_: supervisor.stop())
    say(f"logging to {log_path.relative_to(LOCAL_DIR.parent)}")
    supervisor.run()
    log.close()


if __name__ == "__main__":
    main()
