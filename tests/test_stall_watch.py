"""K198e: pbs/tools/stall_watch.sh alerts ONCE when a running job's progress files stop changing, says so again when
they move, takes no verdict from a failing qstat, and exits with a "finished" alert when the job ends. Alert only.

Driven by a fake qstat whose answer is a file the test rewrites; POLL / STALL_SEC are a fraction of a second / two
seconds so the whole story takes ~10 s.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "pbs" / "tools" / "stall_watch.sh"
sys.path.insert(0, str(ROOT / "pbs"))
from dag import contract  # noqa: E402

FAKE_QSTAT = r"""#!/bin/bash
answer=$(cat "$FAKE_STATE")
case $answer in
  FAIL)    echo "qstat: cannot connect to server" >&2; exit 2 ;;
  UNKNOWN) echo "qstat: Unknown Job Id $3.aurora-pbs" >&2; exit 153 ;;
  *)       echo "Job Id: $3"; echo "    job_state = ${answer%% *}"
           [[ $answer == *" "* ]] && echo "    Exit_status = ${answer#* }"; exit 0 ;;
esac
"""


def _start(tmp_path, files):
    qstat = tmp_path / "qstat"
    qstat.write_text(FAKE_QSTAT)
    qstat.chmod(0o755)
    state = tmp_path / "state"
    state.write_text("Q")
    env = dict(os.environ, QSTAT=str(qstat), FAKE_STATE=str(state), POLL="0.2", STALL_SEC="2",
               NOTIFY_DIR=str(tmp_path / "notifications"))
    proc = subprocess.Popen(["bash", str(SCRIPT), "8900001", *map(str, files)], env=env,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    return proc, state


def _alerts(tmp_path):
    d = tmp_path / "notifications" / "watchdog"
    return [json.loads(p.read_text()) for p in sorted(d.glob("*.json"))] if d.is_dir() else []


def _events(tmp_path):
    return [a["event"] for a in _alerts(tmp_path)]


def _grow(path, seconds):
    end = time.time() + seconds
    while time.time() < end:
        with open(path, "a") as fh:
            fh.write("step\n")
        time.sleep(0.15)


def test_stall_is_alerted_once_then_resumed_then_finished(tmp_path):
    log = tmp_path / "train.err"
    log.write_text("")
    proc, state = _start(tmp_path, [log])
    try:
        time.sleep(0.8)                      # queued: only waited for
        assert _events(tmp_path) == []
        state.write_text("R")
        _grow(log, 1.5)                      # running and progressing
        assert _events(tmp_path) == []
        time.sleep(3.5)                      # no progress for > STALL_SEC
        assert _events(tmp_path) == ["stalled"]
        time.sleep(1.0)                      # still stalled: no second alert
        assert _events(tmp_path) == ["stalled"]
        _grow(log, 0.8)
        assert _events(tmp_path) == ["stalled", "resumed"]
        state.write_text("FAIL")             # qstat down: no verdict, keeps running
        time.sleep(1.0)
        assert proc.poll() is None and _events(tmp_path) == ["stalled", "resumed"]
        state.write_text("F 0")
        assert proc.wait(timeout=5) == 0
    finally:
        if proc.poll() is None:
            proc.kill()
    alerts = _alerts(tmp_path)
    assert [a["event"] for a in alerts] == ["stalled", "resumed", "finished"]
    assert "exit status 0" in alerts[-1]["text"] and all(a["source"] == "watchdog" for a in alerts)
    assert all(a["job_id"] == "8900001" and a["files"] == [str(log)] for a in alerts)
    out = proc.stdout.read()
    assert "qstat failed" in out and out.count("ALERT stalled") == 1
    # the DAG reads job notifications only: watchdog alerts never look like a job end
    assert contract.read_notifications(tmp_path / "notifications") == {}


def test_a_missing_progress_file_counts_as_no_progress(tmp_path):
    proc, state = _start(tmp_path, [tmp_path / "not-written-yet.log"])
    try:
        state.write_text("R")
        time.sleep(3.5)
        assert _events(tmp_path) == ["stalled"]
    finally:
        proc.kill()


def test_an_unknown_job_ends_the_watch(tmp_path):
    proc, state = _start(tmp_path, [tmp_path / "x.log"])
    state.write_text("UNKNOWN")
    assert proc.wait(timeout=5) == 0
    assert _events(tmp_path) == ["finished"]
