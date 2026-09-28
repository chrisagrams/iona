"""The job-DAG scheduler (pbs/dag, notes/PLAN.md I2), against the FAKE PBS backend.

Nothing here touches the real batch system: every scheduler test runs on FakeBackend, the
PBS parsing tests use captured text, and the job_finish.sh tests run bash with a fake
PBS_JOBID and a tmp SCRATCH_ROOT.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "pbs"))

from dag import backend as be  # noqa: E402
from dag import cli, config, contract, history, scheduler, spec  # noqa: E402
from dag.backend import Behavior, FakeBackend  # noqa: E402
from dag.spec import Calibration, Node, Pipeline, Sweep  # noqa: E402

CARD = "T1-I"
EVAL = "pbs/eval_grouped_retrieval.pbs"
SMOKE_ARMS = "sweeps/arms/hp_scale_smoke.txt"     # 2 arms, read-only
SWEEP_ROOT = "configs/sweep-hp-scale"


# ------------------------------------------------------------------------ helpers
@pytest.fixture
def env(tmp_path):
    dec = tmp_path / "DECISIONS.md"
    dec.write_text("| date | ID | decision | by |\n|---|---|---|---|\n"
                   f"| 2026-09-27 | {CARD} | test card | user |\n")

    class Env:
        home = tmp_path / "home"

        def cfg(self, **over):
            base = {"scratch_root": str(tmp_path / "scratch"),
                    "output_root": str(tmp_path / "out"), "decisions_file": str(dec),
                    "log_dir": str(tmp_path / "logs"), "dry_run": False}
            base.update(over)
            return config.load_config(self.home, base)

        def setup(self, nodes, budget=1000, behaviors=None, **over):
            self.config = self.cfg(**over)
            self.graph = spec.graph_from([Pipeline("p", nodes, budget)])
            self.fake = FakeBackend(self.config, behaviors)
            self.state = {"version": 1, "nodes": {}}
            return self

        def tick(self, live=True, write=True):
            self.sch = scheduler.Scheduler(self.graph, self.config, self.fake, self.home,
                                           live=live, state=self.state)
            return self.sch.run(write=write)

        def run(self, ticks=400, every=10):
            reps = []
            for _ in range(ticks):
                rep = self.tick()
                reps.append(rep)
                if all(v["status"] in ("succeeded", "failed", "blocked")
                       for v in rep["nodes"].values()) and not self.fake.live:
                    break
                self.fake.advance(every)
            return reps

        def status(self, key):
            return self.state["nodes"].get(f"p/{key}", {}).get("status", "waiting")

        def subs(self, node=None):
            return [s for s in self.fake.submissions
                    if node is None or s["vars"].get("DAG_NODE") == f"p/{node}"]

    return Env()


def job(id, deps=(), rt=30, **kw):
    kw.setdefault("approved", CARD)
    return Node(id=id, script=EVAL, deps=list(deps), runtime_min=rt, **kw)


def sweep(id, deps=(), rt=300, **kw):
    kw.setdefault("approved", CARD)
    kw.setdefault("queues", ["capacity"])
    return Node(id=id, sweep=Sweep(SWEEP_ROOT, SMOKE_ARMS), deps=list(deps), runtime_min=rt,
                **kw)


def actions(rep, kind):
    return [a for a in rep["actions"] if a["action"] == kind]


# ------------------------------------------------------------------------ spec
def test_topological_order_and_cycles():
    g = spec.graph_from([Pipeline("p", [job("c", ["b"]), job("a"), job("b", ["a"])], 10)])
    order = spec.topo_order(g["nodes"])
    assert order.index("p/a") < order.index("p/b") < order.index("p/c")
    g = spec.graph_from([Pipeline("p", [job("a", ["b"]), job("b", ["a"])], 10)])
    with pytest.raises(ValueError, match="cycle"):
        spec.topo_order(g["nodes"])
    g = spec.graph_from([Pipeline("p", [job("a", ["ghost"])], 10)])
    with pytest.raises(ValueError, match="unknown"):
        spec.topo_order(g["nodes"])


def test_validate_catches_secrets_commas_and_missing_files(env):
    cfg = env.cfg()
    g = spec.graph_from([Pipeline("p", [
        job("a", vars={"HF_TOKEN": "x"}), job("b", vars={"ARMS": "a,b"}),
        Node(id="c", script="pbs/nope.pbs", approved=CARD, runtime_min=5),
        Node(id="d", sweep=Sweep(SWEEP_ROOT, "sweeps/arms/nope.txt"), approved=CARD),
        job("e", approved=None), job("f", approved="Z9-X")], 10)])
    errors, warnings = spec.validate(g, REPO, cfg["queues"], cfg["decisions_file"])
    text = "\n".join(errors)
    assert "secret" in text and "comma" in text and "nope.pbs" in text and "nope.txt" in text
    assert any("not approved" in w for w in warnings)
    assert any("Z9-X" in w for w in warnings)


def test_logged_cards_reads_the_decisions_table():
    cards = spec.logged_cards(REPO / "notes" / "DECISIONS.md")
    assert "K66-C" in cards and "I2" in cards


# ------------------------------------------------------------------------ ordering / approval
def test_children_wait_for_parents_to_succeed(env):
    env.setup([job("a"), job("b", ["a"]), job("c", ["b"])])
    env.run()
    subs = [s["vars"]["DAG_NODE"] for s in env.subs()]
    assert subs == ["p/a", "p/b", "p/c"]
    t = {s["vars"]["DAG_NODE"]: s["t"] for s in env.subs()}
    ends = {k: v["t"] for k, v in env.fake.ended.items()}
    ids = {s["vars"]["DAG_NODE"]: s["id"] for s in env.subs()}
    assert t["p/b"] >= ends[ids["p/a"]] and t["p/c"] >= ends[ids["p/b"]]
    assert all(env.status(k) == "succeeded" for k in "abc")


def test_approval_gate_refuses_unapproved_and_unlogged_cards(env):
    env.setup([job("a", approved=None), job("b", approved="Z9-X"), job("c")])
    rep = env.tick()
    assert [a["node"] for a in actions(rep, "submit")] == ["p/c"]
    assert "NEEDS APPROVAL" in rep["nodes"]["p/a"]["view"]
    assert "not in DECISIONS.md" in rep["nodes"]["p/b"]["view"]
    env.run()
    assert env.subs("a") == [] and env.subs("b") == []


# ------------------------------------------------------------------------ queues
def test_queue_choice_and_walltime(env):
    env.setup([job("short", rt=20), job("long", rt=100, queues=["debug", "capacity"]),
               job("wide", rt=20, nodes=4), job("toolong", rt=40, queues=["debug"])],
              max_submissions_per_tick=10, generic_q_limit=10)
    rep = env.tick(live=False)
    got = {a["node"]: (a["queue"], a["walltime_min"]) for a in actions(rep, "would-submit")}
    # toolong (debug only, longer critical path) takes debug's free run slot first, so
    # short goes where it can start now: debug-scaling. 20 * 1.3 = 26 -> 30 (5-min steps)
    assert got["p/short"] == ("debug-scaling", 30)
    assert got["p/long"] == ("capacity", 135)       # 100 * 1.3 = 130 -> 135 (15-min steps)
    assert got["p/wide"] == ("capacity", 30)        # 4 nodes do not fit debug (max 2)
    assert got["p/toolong"] == ("debug", 55)        # 40 min fits (< 50), 52 -> 55
    env.setup([job("x", rt=55, queues=["debug"])])
    rep = env.tick(live=False)
    assert "NO QUEUE FITS" in rep["nodes"]["p/x"]["view"]


def test_multi_queue_packing_respects_every_limit(env):
    # 4 short (debug-able) + 2 long capacity jobs; limits: debug & debug-scaling 1R+1Q,
    # capacity 2R, generic Q 3 across queues.
    nodes = [job(f"s{i}", rt=20, queues=["debug", "debug-scaling"]) for i in range(4)]
    nodes += [job(f"L{i}", rt=100, queues=["capacity"]) for i in range(2)]
    env.setup(nodes, generic_q_limit=3, max_submissions_per_tick=10)
    rep = env.tick()
    subs = actions(rep, "submit")
    assert len(subs) == 3                            # the generic Q limit
    # longest critical path first: both capacity jobs, then one short job to debug
    assert [a["queue"] for a in subs] == ["capacity", "capacity", "debug"]
    for _ in range(80):
        env.fake.advance(5)
        env.tick()
        live = env.fake.live.values()
        for q, lim in env.config["queues"].items():
            assert sum(j.queue == q and j.state == "R" for j in live) <= lim["max_running"]
        assert sum(j.state in "QH" for j in live) <= 3
    assert all(env.status(n.id) == "succeeded" for n in nodes)
    assert {s["queue"] for s in env.subs() if s["vars"]["DAG_NODE"].startswith("p/s")} \
        == {"debug", "debug-scaling"}


def test_generic_q_limit_counts_jobs_outside_the_dag(env):
    env.setup([job("a"), job("b")])
    env.fake.submit(EVAL, "capacity", 1, 60, "someone-elses", {})
    env.fake.submit(EVAL, "capacity", 1, 60, "another", {})     # 2 in Q = generic limit
    rep = env.tick()
    assert actions(rep, "submit") == []
    assert "no free slot" in rep["nodes"]["p/a"]["view"]


def test_qsub_limit_refusal_is_not_a_failure(env):
    env.setup([job("a"), job("b", queues=["capacity"])], generic_q_limit=5)
    env.fake.cfg = dict(env.config, generic_q_limit=0)          # PBS is stricter than we think
    rep = env.tick()
    assert actions(rep, "limit-hit")
    assert env.status("a") == "waiting" and env.status("b") == "waiting"
    env.fake.cfg = env.config
    env.run()
    assert env.status("a") == env.status("b") == "succeeded"


# ------------------------------------------------------------------------ success contract
def _note(tmp_path, man=None, **kw):
    n = {"job_id": "7", "status": "ok"}
    n.update(kw)
    if man is not None:
        n["manifest"] = str(contract.write_json_atomic(tmp_path / "m" / "7" / "_SUCCESS.json",
                                                       man))
    return n


def test_success_requires_a_valid_manifest(tmp_path):
    good = {"job_id": "7", "checks": [{"name": "x", "ok": True}],
            "outputs": [{"path": "/a", "exists": True}], "arms_ok": 2, "arms_total": 2,
            "commit": "abc"}
    ok, why, _ = contract.verify_success(_note(tmp_path, good), tmp_path / "m", 2, [])
    assert ok, why
    assert not contract.verify_success(_note(tmp_path), tmp_path / "none", None, [])[0]
    assert not contract.verify_success(_note(tmp_path, dict(good, job_id="8")),
                                       tmp_path / "m", 2, [])[0]
    bad = dict(good, checks=[{"name": "x", "ok": False}])
    assert "checks failed" in contract.verify_success(_note(tmp_path, bad), tmp_path / "m", 2, [])[1]
    bad = dict(good, arms_ok=1)
    assert "arms 1/2" in contract.verify_success(_note(tmp_path, bad), tmp_path / "m", 2, [])[1]
    bad = dict(good, outputs=[{"path": "/a", "exists": False}])
    assert "missing" in contract.verify_success(_note(tmp_path, bad), tmp_path / "m", 2, [])[1]
    assert "spec outputs" in contract.verify_success(_note(tmp_path, good), tmp_path / "m", 2,
                                                     [str(tmp_path / "absent*")])[1]
    assert not contract.verify_success(_note(tmp_path, good, status="partial"),
                                       tmp_path / "m", 2, [])[0]


def test_a_missing_manifest_fails_the_node(env):
    env.setup([job("a"), job("b", ["a"])])
    env.tick()
    env.fake.advance(40)
    for p in Path(env.config["manifests_dir"]).rglob("_SUCCESS.json"):
        p.unlink()
    rep = env.tick()
    assert env.status("a") == "failed" and "_SUCCESS.json" in rep["nodes"]["p/a"]["view"]
    assert env.status("b") == "blocked"


def test_snapshot_parsing():
    snap = ("source:   /x\nbranch:   dev\ncommit:   abc123\ntaken: now\njob: 1\n"
            "uncommitted changes in the checkout at snapshot time (code dirs):\n     M a.py\n")
    assert contract.parse_snapshot(snap) == {"commit": "abc123", "branch": "dev", "dirty": True}


# ------------------------------------------------------------------------ job_finish.sh
def _bash_job(tmp_path, body, **envvars):
    snap = tmp_path / "snap"
    snap.mkdir(exist_ok=True)
    (snap / "SNAPSHOT.txt").write_text(
        "branch:   b\ncommit:   deadbeef\nuncommitted changes in the checkout at snapshot "
        "time (code dirs):\n")
    script = tmp_path / "job.sh"
    script.write_text(f"set -uo pipefail\nSCRATCH_ROOT={tmp_path}/scr\n"
                      f"source {REPO}/pbs/lib/job_finish.sh\n"
                      f"export MSDELTA_CODE_DIR={snap}\n{body}\n")
    e = {**os.environ, "PBS_JOBID": "4242.aurora-pbs", "DAG_NODE": "p/n"}
    e.pop("DRY_RUN", None)
    e.update(envvars)
    return subprocess.run(["bash", str(script)], capture_output=True, text=True, env=e,
                          timeout=30)


def _notes(tmp_path):
    d = tmp_path / "scr" / "notifications"
    return sorted(d.glob("*.json")) if d.exists() else []


def test_job_finish_success_writes_manifest_then_notification(tmp_path):
    out = tmp_path / "o"
    r = _bash_job(tmp_path, f"mkdir -p {out}; jf_output {out}; jf_check made test -d {out}\n"
                            "JF_ARMS_OK=3 JF_ARMS_TOTAL=3\nexit 0")
    assert r.returncode == 0, r.stderr
    [n] = _notes(tmp_path)
    assert n.name.endswith("_4242_p-n_ok.json")
    note = json.loads(n.read_text())
    man = json.loads(Path(note["manifest"]).read_text())
    assert man["job_id"] == "4242" and man["commit"] == "deadbeef" and man["arms_ok"] == 3
    assert contract.verify_success(note, tmp_path / "scr" / "manifests", 3, [])[0]
    assert not list((tmp_path / "scr").rglob(".*tmp*"))       # atomic: no leftovers


@pytest.mark.parametrize("body,status,code", [
    ("JF_ARMS_OK=1 JF_ARMS_TOTAL=3\nexit 1", "partial", 1),
    ("exit 2", "failed", 2),
    ("jf_output /nonexistent/x\nexit 0", "failed", 0),        # exit 0 but an output missing
    ("jf_check nope false\nexit 0", "failed", 0),
])
def test_job_finish_failures_write_only_a_notification(tmp_path, body, status, code):
    r = _bash_job(tmp_path, body)
    assert r.returncode == code                               # exit status is never changed
    [n] = _notes(tmp_path)
    note = json.loads(n.read_text())
    assert note["status"] == status and note["manifest"] is None
    assert not (tmp_path / "scr" / "manifests").exists()


def test_job_finish_records_walltime_on_sigterm(tmp_path):
    script_body = "sleep 20 & wait"
    snap = tmp_path / "job.sh"
    p = subprocess.Popen(["bash", "-c", f"SCRATCH_ROOT={tmp_path}/scr; source "
                          f"{REPO}/pbs/lib/job_finish.sh; {script_body}"],
                         env={**os.environ, "PBS_JOBID": "77.x", "DAG_WALLTIME_SEC": "1"})
    time.sleep(1.5)
    p.terminate()
    assert p.wait(timeout=10) == 143
    [n] = _notes(tmp_path)
    assert json.loads(n.read_text())["status"] == "walltime"
    del snap


def test_job_finish_is_a_noop_outside_pbs_and_in_dry_run(tmp_path):
    _bash_job(tmp_path, "exit 0", DRY_RUN="1")
    e = {k: v for k, v in os.environ.items() if k != "PBS_JOBID"}
    subprocess.run(["bash", "-c", f"SCRATCH_ROOT={tmp_path}/scr; source "
                    f"{REPO}/pbs/lib/job_finish.sh; exit 0"], env=e, check=True)
    assert _notes(tmp_path) == []


def test_sweep_script_reports_an_early_exit(tmp_path):
    """The wiring in aurora-finetune-sweep.pbs: a refused grid still leaves a notification."""
    root = tmp_path / "grid"
    (root / "armA").mkdir(parents=True)
    (root / "armA" / "training.args").write_text("--x 1\n")
    e = {**os.environ, "PBS_JOBID": "5151.x", "SWEEP_ROOT": str(root),
         "SCRATCH_ROOT": str(tmp_path / "scr"), "DAG_NODE": "p/sw"}
    e.pop("DRY_RUN", None)
    r = subprocess.run(["bash", "pbs/aurora-finetune-sweep.pbs"], cwd=REPO, env=e,
                       capture_output=True, text=True, timeout=60)
    assert r.returncode == 2 and "no staleness check registered" in r.stderr
    [n] = _notes(tmp_path)
    note = json.loads(n.read_text())
    assert note["status"] == "failed" and note["exit_code"] == 2 and note["node"] == "p/sw"


def test_pbs_scripts_source_job_finish_after_scratch_root():
    for name in ("aurora-finetune-sweep.pbs", "eval_grouped_retrieval.pbs"):
        text = (REPO / "pbs" / name).read_text()
        i = text.index('source "$REPO_DIR/pbs/lib/job_finish.sh"')
        assert text.index("SCRATCH_ROOT=${SCRATCH_ROOT:-") < i
        assert "jf_output" in text


# ------------------------------------------------------------------------ notifications
def test_notification_reader(tmp_path):
    d = tmp_path / "n"
    contract.write_json_atomic(d / contract.notification_name(100, "11", "p/a", "failed"),
                               {"job_id": "11", "status": "failed", "source": "job"})
    contract.write_json_atomic(d / contract.notification_name(200, "11", "p/a", "ok"),
                               {"job_id": "11", "status": "ok", "source": "job"})
    (d / ".20260101T000000Z_12_x_ok.json.tmp.1").write_text("{half")
    (d / "20260101T000000Z_13_x_ok.json").write_text("{broken")
    contract.write_alert(d, "p/a", "hung", "text", 100)
    notes = contract.read_notifications(d)
    assert list(notes) == ["11"] and notes["11"]["status"] == "ok"
    assert len(list((d / "scheduler").glob("*.json"))) == 1


def test_job_gone_without_notification_is_failed_after_grace(env):
    env.setup([job("a"), job("b", ["a"])], behaviors={"-a": Behavior(outcome="vanish")})
    env.tick()
    env.fake.advance(40)
    env.tick()
    assert env.status("a") == "submitted"          # still within the grace period
    env.fake.advance(10)
    rep = env.tick()
    assert env.status("a") == "failed" and "no notification" in rep["nodes"]["p/a"]["view"]
    assert env.status("b") == "blocked"
    alerts = list((Path(env.config["notifications_dir"]) / "scheduler").glob("*"))
    assert any("failed" in p.name for p in alerts) and any("blocked" in p.name for p in alerts)


# ------------------------------------------------------------------------ partial / resume
def test_partial_sweep_is_resumed_with_resume_job(env):
    env.setup([sweep("sw"), job("after", ["sw"])],
              behaviors={"-sw": Behavior(outcome="partial", arms_ok=1, on_resume="ok")})
    env.run()
    subs = env.subs("sw")
    assert len(subs) == 2
    assert "RESUME_JOB" not in subs[0]["vars"]
    assert subs[1]["vars"]["RESUME_JOB"] == subs[0]["id"]      # only the missing arms rerun
    assert subs[1]["name"].endswith("-r1")
    assert env.status("sw") == "succeeded" and env.status("after") == "succeeded"


def test_walltime_kill_resumes_and_the_resume_cap_holds(env):
    env.setup([sweep("sw", rt=60, max_resumes=2), job("after", ["sw"])],
              behaviors={"-sw": Behavior(runtime_min=500, on_resume="partial", arms_ok=1)})
    env.run()
    subs = env.subs("sw")
    assert len(subs) == 3                                      # 1 run + 2 resumes, then stop
    assert {s["vars"].get("RESUME_JOB") for s in subs[1:]} == {subs[0]["id"]}
    assert env.status("sw") == "failed" and env.status("after") == "blocked"


def test_failed_sweep_with_no_arm_done_is_not_resumed(env):
    env.setup([sweep("sw")], behaviors={"-sw": Behavior(outcome="fail")})
    env.run()
    assert len(env.subs("sw")) == 1 and env.status("sw") == "failed"


def test_rerun_on_walltime_for_resumable_scripts(env):
    env.setup([job("ev", rt=30, rerun_on=("walltime",), max_retries=1)],
              behaviors={"-ev": Behavior(runtime_min=90)})
    env.run()
    assert len(env.subs("ev")) == 2 and env.status("ev") == "succeeded"


# ------------------------------------------------------------------------ orphans
def test_failed_parent_blocks_descendants(env):
    env.setup([job("a"), job("b", ["a"]), job("c", ["b"]), job("d")],
              behaviors={"-a": Behavior(outcome="fail")})
    env.run()
    assert env.status("a") == "failed"
    assert env.status("b") == env.status("c") == "blocked"
    assert env.status("d") == "succeeded"
    assert env.subs("b") == [] and env.subs("c") == []


def test_paired_smoke_run_uses_afterok_and_is_dropped_if_the_smoke_fails(env):
    env.setup([job("smoke", rt=20, queues=["debug"]),
               job("run", ["smoke"], rt=300, queues=["capacity"], submit_with_parent=True)],
              behaviors={"-smoke": Behavior(outcome="fail")}, generic_q_limit=3)
    rep = env.tick()
    subs = actions(rep, "submit")
    assert [a["node"] for a in subs] == ["p/smoke", "p/run"]
    assert subs[1]["depend"] == f"afterok:{subs[0]['job']}"
    env.run()
    assert env.status("smoke") == "failed" and env.status("run") == "blocked"
    assert len(env.subs("run")) == 1


def test_live_orphan_is_cancelled_only_when_enabled(env):
    for allow in (False, True):
        env.setup([job("a"), job("b", ["a"])], allow_cancel_orphans=allow)
        env.state["nodes"]["p/a"] = {"status": "failed", "jobs": [], "resumes": 0,
                                     "retries": 0, "alerts": []}
        jid = env.fake.submit(EVAL, "debug", 1, 30, "dag-p-b", {})
        env.state["nodes"]["p/b"] = {"status": "submitted", "job": jid, "output_job": jid,
                                     "jobs": [{"job": jid, "role": "run", "nodes": 1,
                                               "walltime_min": 30}],
                                     "resumes": 0, "retries": 0, "alerts": []}
        rep = env.tick()
        if allow:
            assert actions(rep, "cancel-orphan") and (jid, False) in env.fake.deletions
            assert env.status("b") == "blocked"
        else:
            assert actions(rep, "orphan") and env.fake.deletions == []
            assert env.status("b") == "submitted"


# ------------------------------------------------------------------------ hung jobs
def test_hung_job_is_flagged_but_not_deleted_by_default(env):
    env.setup([job("a")], behaviors={"-a": Behavior(outcome="hung")})
    env.tick()
    env.fake.advance(5)
    assert not actions(env.tick(), "flag-hung")                # < hung_minutes
    env.fake.advance(10)
    rep = env.tick()
    assert actions(rep, "flag-hung") and actions(rep, "hung-notify-only")
    assert env.fake.deletions == [] and env.status("a") == "submitted"
    env.fake.advance(10)
    assert not actions(env.tick(), "flag-hung")                # alerted once


def test_hung_job_force_deleted_and_resubmitted_when_enabled(env):
    env.setup([job("a", max_retries=1)], allow_force_delete=True,
              behaviors={"-a": Behavior(outcome="hung", on_resume="ok")})
    env.run()
    assert env.fake.deletions and env.fake.deletions[0][1] is True    # qdel -W force (FT27)
    assert len(env.subs("a")) == 2 and env.status("a") == "succeeded"


def test_hung_retry_cap(env):
    env.setup([job("a", max_retries=1)], allow_force_delete=True,
              behaviors={"-a": Behavior(outcome="hung", on_resume="hung")})
    for _ in range(20):
        env.tick()
        env.fake.advance(10)
    assert len(env.subs("a")) == 2                             # never a third copy
    assert len(env.fake.deletions) == 1


def test_stalled_progress_is_flagged(env):
    env.setup([sweep("sw", rt=300, stall_minutes=30,
                     progress=["{scratch}/runs/sweep-*-{job}/logs/train.log"])],
              behaviors={"-sw": Behavior(outcome="stall")})
    env.tick()
    env.fake.advance(20)
    assert not actions(env.tick(), "flag-hung")
    env.fake.advance(20)
    rep = env.tick()
    [a] = actions(rep, "flag-hung")
    assert "no progress" in a["detail"]


# ------------------------------------------------------------------------ runaway protection
def test_idempotent_names_adopt_a_live_job_instead_of_submitting(env):
    env.setup([job("a")])
    jid = env.fake.submit(EVAL, "debug", 1, 30, scheduler.job_name(env.graph["nodes"]["p/a"]),
                          {})
    rep = env.tick()
    assert actions(rep, "submit") == [] and actions(rep, "adopt")[0]["job"] == jid
    assert env.state["nodes"]["p/a"]["job"] == jid
    assert len(env.fake.submissions) == 1


def test_repeated_ticks_never_duplicate(env):
    env.setup([job("a"), job("b")])
    for _ in range(5):
        env.tick()
    assert len(env.fake.submissions) == 2


def test_tick_lock_prevents_overlap(env, capsys):
    env.home.mkdir(parents=True)
    with scheduler.TickLock(env.home):
        with pytest.raises(scheduler.LockHeld):
            with scheduler.TickLock(env.home):
                pass
        rc = cli.main(["--home", str(env.home), "--pipelines-dir", str(env.home),
                       "tick", "--backend", "fake"])
        assert rc == 2 and "tick skipped" in capsys.readouterr().out
    with scheduler.TickLock(env.home):
        pass                                                   # released


def test_dry_run_is_the_default_and_touches_nothing(env):
    cfg = config.load_config(env.home, {})
    assert cfg["dry_run"] is True and cfg["allow_force_delete"] is False
    env.setup([job("a")], dry_run=True)
    rep = env.tick(live=True)                                  # --live without the config switch
    assert not rep["live"] and env.fake.submissions == []
    assert actions(rep, "would-submit")
    pbs = be.PBSBackend(REPO, REPO / "pbs" / "logs")
    with pytest.raises(be.SideEffectRefused):
        pbs.submit(EVAL, "debug", 1, 30, "x", {})
    with pytest.raises(be.SideEffectRefused):
        pbs.delete("1")


def test_dry_run_does_not_run_prepare_steps(env):
    ran = []
    env.setup([job("a", prepare=spec.Prepare("write x", lambda ctx: ran.append(1) or []))],
              dry_run=True)
    rep = env.tick()
    assert ran == [] and actions(rep, "would-submit")[0]["prepare"] == "write x"
    env.setup([job("a", prepare=spec.Prepare("write x", lambda ctx: ran.append(1) or []))])
    env.tick()
    assert ran == [1]


def test_budget_refuses_submissions_beyond_it(env):
    env.setup([job("a", rt=600, queues=["capacity"]), job("b", rt=600, queues=["capacity"])],
              budget=15)                                       # one 13 h job fits, not two
    rep = env.tick()
    assert len(actions(rep, "submit")) == 1
    assert "OVER BUDGET" in " ".join(v["view"] for v in rep["nodes"].values())
    assert any(p.name.endswith("_budget.json")
               for p in (Path(env.config["notifications_dir"]) / "scheduler").glob("*"))


def test_max_submissions_per_tick(env):
    env.setup([job(f"a{i}", rt=600, queues=["capacity"]) for i in range(4)],
              max_submissions_per_tick=1, generic_q_limit=10,
              queues={"capacity": {"max_queued": 10}})
    assert len(actions(env.tick(), "submit")) == 1


def test_heartbeat_written_and_stale_warning(env, capsys, tmp_path):
    pdir = tmp_path / "pipes"
    pdir.mkdir()
    (pdir / "t.py").write_text(
        "from dag.spec import Node, Pipeline\n"
        f"PIPELINE = Pipeline('t', [Node(id='a', script='{EVAL}', runtime_min=10, "
        "approved='K66-C')], 5)\n")
    args = ["--home", str(env.home), "--pipelines-dir", str(pdir)]
    assert cli.main(args + ["status", "--backend", "fake"]) == 0
    assert "no tick has run" in capsys.readouterr().out
    assert cli.main(args + ["tick", "--backend", "fake"]) == 0
    hb = json.loads((env.home / "heartbeat.json").read_text())
    assert hb["live"] is False
    cli.main(args + ["status", "--backend", "fake"])
    assert "STALE" not in capsys.readouterr().out
    hb["time"] -= 3600
    (env.home / "heartbeat.json").write_text(json.dumps(hb))
    cli.main(args + ["status", "--backend", "fake"])
    out = capsys.readouterr().out
    assert "heartbeat STALE" in out and "t/a" in out


# ------------------------------------------------------------------------ estimates
def test_calibration_node_needs_approval_and_sets_the_estimate(env):
    cal = Calibration(vars={"MAX_STEPS": "50"}, steps=50, full_steps=5000, overhead_min=5)
    env.setup([Node(id="big", sweep=Sweep(SWEEP_ROOT, SMOKE_ARMS), approved=CARD,
                    calibration=cal, queues=["debug", "capacity"])])
    rep = env.tick()
    assert "NEEDS APPROVAL" in rep["nodes"]["p/big~calib"]["view"]
    assert rep["nodes"]["p/big"]["view"] == "awaiting calibration run"
    assert env.fake.submissions == []
    cal.approved = CARD
    env.setup([Node(id="big", sweep=Sweep(SWEEP_ROOT, SMOKE_ARMS), approved=CARD,
                    calibration=cal, queues=["debug", "capacity"])],
              behaviors={"-calib": Behavior(runtime_min=8)})
    env.run()
    first, second = env.fake.submissions
    assert first["queue"] == "debug" and first["vars"]["MAX_STEPS"] == "50"
    # 5 + (8 - 5) * 100 = 305 min -> capacity, 305 * 1.3 = 396.5 -> 405
    assert second["queue"] == "capacity" and second["walltime_min"] == 405
    assert "MAX_STEPS" not in second["vars"]


def test_history_from_logs_and_notifications(tmp_path):
    logs = tmp_path / "logs"
    logs.mkdir()
    (logs / "100.aurora.OU").write_text(
        "  [ok   armA] 3600s on x1\n  [ok   armB] 1800s on x1\n  [FAIL armC] exit 1\n"
        "=== done: 2/3 arms ok\n")
    (logs / "101.aurora.OU").write_text("  [ok   armA] 4200s on x2\n")
    scanned = history.scan_logs(logs, tmp_path / "cache.json")
    assert scanned["100"]["done"] == [2, 3]
    assert history.scan_logs(logs, tmp_path / "cache.json") == scanned      # cached
    times = history.arm_times(scanned)
    assert times["armA"] == [3600, 4200] or times["armA"] == [4200, 3600]
    # 2 arms on 1 slot: 65 + 30 min + overhead
    assert history.estimate_sweep_from_logs(["armA", "armB"], times, 1) == pytest.approx(
        65 + 30 + history.SWEEP_OVERHEAD_MIN)
    assert history.estimate_sweep_from_logs(["armA", "armZ"], times, 1) is None
    rt = history.notification_runtimes({"1": {"status": "ok", "kind": "k", "runtime_sec": 600},
                                        "2": {"status": "ok", "kind": "k", "runtime_sec": 1800},
                                        "3": {"status": "failed", "kind": "k",
                                              "runtime_sec": 5}})
    n = Node(id="x", script=EVAL, kind="k")
    assert history.estimate(n, REPO, rt, {}) == (20.0, "history (2 runs)")
    assert history.estimate(Node(id="y", script=EVAL), REPO, rt, {})[0] is None


# ------------------------------------------------------------------------ commits
def test_mixed_commits_are_flagged(env):
    env.setup([job("a"), job("b"), job("cmp", ["a", "b"])],
              behaviors={"-a": Behavior(commit="1" * 40), "-b": Behavior(commit="2" * 40)})
    reps = env.run()
    assert any("MIXED COMMITS" in w for r in reps for w in r["warnings"])
    assert contract.mixed_commits({"x": "abc", "y": "abc"}) is None


# ------------------------------------------------------------------------ PBS parsing
QSTAT_U = """
aurora-pbs-0001.hostmgmt.cm.aurora.alcf.anl.gov:
                                                                 Req'd  Req'd   Elap
Job ID               Username Queue    Jobname    SessID NDS TSK Memory Time  S Time
-------------------- -------- -------- ---------- ------ --- --- ------ ----- - -----
8875194.aurora-pbs-* khuss    debug    hpscalesm* 189568   1 208    --  01:00 R 00:46
8875200.aurora-pbs-* khuss    capacity hps400m       --    1 208    --  14:00 Q   --
"""
QSTAT_JSON = json.dumps({"Jobs": {
    "8875194.aurora-pbs-0001.hostmgmt.cm.aurora.alcf.anl.gov": {
        "Job_Name": "hpscalesmoke", "job_state": "R", "queue": "debug",
        "resources_used": {"walltime": "00:46:21", "cput": "00:00:04"},
        "Resource_List": {"nodect": 1, "walltime": "01:00:00", "select": "1"},
        "stime": "Sun Sep 27 22:44:36 2026"},
    "8875200.aurora-pbs-0001.hostmgmt.cm.aurora.alcf.anl.gov": {
        "Job_Name": "hps400m", "job_state": "R", "queue": "capacity",
        "Resource_List": {"nodect": 2, "walltime": "14:00:00"},
        "stime": "Sun Sep 27 22:44:36 2026"},
    "8870000.x": {"Job_Name": "old", "job_state": "F", "queue": "debug"}}})


def test_qstat_parsing():
    assert be.parse_qstat_u(QSTAT_U) == ["8875194", "8875200"]
    jobs = {j.id: j for j in be.parse_qstat_json(QSTAT_JSON)}
    assert set(jobs) == {"8875194", "8875200"}
    a, b = jobs["8875194"], jobs["8875200"]
    assert a.name == "hpscalesmoke" and a.resources_used and a.walltime_min == 60
    assert b.nodes == 2 and not b.resources_used and b.walltime_min == 840   # FT27 signature
    assert a.stime == time.mktime(time.strptime("Sun Sep 27 22:44:36 2026",
                                                "%a %b %d %H:%M:%S %Y"))


def test_qsub_argv_has_no_secrets_and_the_right_shape():
    argv = be.qsub_argv("pbs/x.pbs", "capacity", 1, 840, "dag-p-a", {"A": "1", "B": "x"},
                        "afterok:123")
    assert argv == ["qsub", "-q", "capacity", "-l", "select=1", "-l", "walltime=14:00:00",
                    "-N", "dag-p-a", "-W", "depend=afterok:123", "-v", "A=1,B=x", "pbs/x.pbs"]


def test_legacy_adopted_job_uses_the_done_line(env, tmp_path):
    logs = tmp_path / "logs"
    logs.mkdir()
    env.setup([sweep("sw", adopt_job="123", legacy_log_success=True), job("after", ["sw"])])
    env.fake.log_path = lambda j: logs / f"{j}.x.OU"
    (logs / "123.x.OU").write_text("=== done: 2/2 arms ok (0 failed) ===\n")
    rep = env.tick()
    assert actions(rep, "adopt") and env.status("sw") == "succeeded"
    assert "legacy" in env.state["nodes"]["p/sw"]["reason"]


# ------------------------------------------------------------------------ K66-C example
def _k66(tmp_path):
    home = tmp_path / "home"
    args = ["--home", str(home), "--pipelines-dir", str(REPO / "pbs" / "dag" / "examples")]
    return home, args


def test_k66c_example_validates_and_plans_the_smoke_first(tmp_path, capsys):
    home, args = _k66(tmp_path)
    assert cli.main(args + ["validate"]) == 0
    assert cli.main(args + ["plan", "--backend", "fake"]) == 0
    out = capsys.readouterr().out
    assert "would-submit       k66c/smoke" in out
    assert "qsub -q debug -l select=1 -l walltime=01:00:00 -N dag-k66c-smoke" in out
    assert out.count("would-submit") == 1                   # nothing else before the smoke
    assert "waiting on smoke" in out and "waiting on train_400m" in out


def test_k66c_example_runs_to_the_end_on_the_fake_backend(tmp_path):
    home, _ = _k66(tmp_path)
    cfg = config.load_config(home, {
        "scratch_root": str(tmp_path / "s"), "output_root": str(tmp_path / "o"),
        "log_dir": str(tmp_path / "logs"), "dry_run": False})
    graph = spec.load(REPO / "pbs" / "dag" / "examples")
    fake = FakeBackend(cfg)
    state = {"version": 1, "nodes": {}}
    for _ in range(400):
        rep = scheduler.Scheduler(graph, cfg, fake, home, live=True, state=state).run(False)
        if all(v["status"] == "succeeded" for v in rep["nodes"].values()):
            break
        fake.advance(10)
    assert all(v["status"] == "succeeded" for v in rep["nodes"].values())
    walls = {s["vars"]["DAG_NODE"]: (s["queue"], s["walltime_min"]) for s in fake.submissions}
    assert walls["k66c/train_400m"] == ("capacity", 840)
    assert walls["k66c/train_025m"] == ("capacity", 600)
    assert {walls[f"k66c/score_400m_{d}"][0] for d in
            ("validation", "oodval", "test", "mouse", "human")} <= {"debug", "debug-scaling"}
    # training started longest first; the models file was written by the prepare step
    order = [s["vars"]["DAG_NODE"] for s in fake.submissions]
    assert order[1] == "k66c/train_400m"
    models = (tmp_path / "o" / "sweeps/arms/score_hp_scale_400m.txt").read_text()
    assert models.count("/final") == 12
    assert not (REPO / "sweeps/arms/score_hp_scale_400m.txt").exists() or \
        "(job 9" not in (REPO / "sweeps/arms/score_hp_scale_400m.txt").read_text()
