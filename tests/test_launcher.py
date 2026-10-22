"""The launcher supervising a stand-in app that crashes, freezes, or runs fine on cue."""

import io
import os
import sys
import textwrap

import pytest

from partycam.launcher import Supervisor

FAKE_APP = textwrap.dedent(
    """
    import os, pathlib, sys, time
    counter = pathlib.Path(sys.argv[1])
    run = int(counter.read_text()) if counter.exists() else 0
    counter.write_text(str(run + 1))
    pathlib.Path(sys.argv[1] + f".pid{run}").write_text(str(os.getpid()))
    plan = sys.argv[2].split(",")
    action = plan[min(run, len(plan) - 1)]
    if action == "ok":
        sys.exit(0)
    if action == "crash":
        sys.exit(3)
    if action == "freeze":
        time.sleep(60)
    if action == "beat":  # healthy for a while: prints [perf] heartbeats, then quits normally
        for _ in range(10):
            print("[perf] fine", flush=True)
            time.sleep(0.15)
        sys.exit(0)
    """
)


@pytest.fixture
def run_plan(tmp_path):
    script = tmp_path / "fake_app.py"
    script.write_text(FAKE_APP)
    counter = tmp_path / "runs"

    def run(plan, **kw):
        said = []
        opts = dict(heartbeat_timeout=0.6, startup_grace=0.6, backoff=[0], poll=0.05)
        opts.update(kw)
        sup = Supervisor([sys.executable, str(script), str(counter), plan], said.append, io.StringIO(), **opts)
        sup.run()
        runs = int(counter.read_text())
        pids = [int((tmp_path / f"runs.pid{i}").read_text()) for i in range(runs)]
        return sup, said, runs, pids

    return run


def alive(pid):
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False


def test_clean_quit_is_not_restarted(run_plan):
    sup, said, runs, _ = run_plan("ok")
    assert runs == 1 and sup.restarts == 0
    assert said[-1] == "party cam stopped. Goodnight!"


def test_crash_is_restarted(run_plan):
    sup, said, runs, _ = run_plan("crash,ok")
    assert runs == 2 and sup.restarts == 1
    assert any("exited with code 3" in m for m in said)


def test_freeze_is_detected_and_restarted_once(run_plan):
    sup, said, runs, pids = run_plan("freeze,ok")
    assert runs == 2 and sup.restarts == 1
    assert sum("no heartbeat" in m for m in said) == 1  # once, not repeated while it's being stopped
    assert not alive(pids[0])  # the frozen one was actually stopped


def test_heartbeats_keep_a_healthy_app_running(run_plan):
    # Runs ~1.5 s with a 0.6 s timeout: only heartbeats keep it from being declared frozen
    sup, said, runs, _ = run_plan("beat")
    assert runs == 1 and sup.restarts == 0
    assert not any("no heartbeat" in m for m in said)


def test_first_retry_uses_the_shortest_pause(run_plan):
    sup, said, runs, _ = run_plan("crash,crash,ok", backoff=[0, 1])
    pauses = [m.split("restarting in ")[1] for m in said if "restarting in" in m]
    assert pauses == ["0s", "1s"]
