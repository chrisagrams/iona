"""Batch-system backends: real PBS (qsub / qstat / qdel) and a FAKE one for tests.

Both expose the same small interface:
    now()                         -> epoch seconds
    jobs()                        -> [Job] every LIVE job of this user (DAG or not)
    submit(script, queue, nodes, walltime_min, name, vars, depend=None) -> job id
                                     raises QueueLimit on "would exceed ... limit"
    delete(job_id, force=False)
    log_path(job_id)              -> Path | None

PBSBackend refuses every side effect unless it was built with allow_side_effects=True,
which the CLI does only for `tick --live` with dry_run turned off in config.json.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from . import contract


class QueueLimit(Exception):
    """qsub refused because a per-user queue limit would be exceeded (not a failure)."""


class SubmitError(Exception):
    pass


class SideEffectRefused(Exception):
    pass


@dataclass
class Job:
    id: str
    name: str
    queue: str
    state: str                    # Q, H, R, E (live states only)
    nodes: int = 1
    walltime_min: float = 0
    stime: float | None = None    # start time (epoch), None while queued
    resources_used: bool = False  # FT27: a healthy running job has them
    depend: str = ""


def _short(job_id: str) -> str:
    return str(job_id).split(".")[0]


def _parse_hms(s: str) -> float:
    parts = [int(p) for p in str(s).split(":")]
    while len(parts) < 3:
        parts.insert(0, 0)
    h, m, sec = parts[-3:]
    return h * 60 + m + sec / 60


def _parse_pbs_time(s: str | None) -> float | None:
    if not s:
        return None
    try:
        return time.mktime(time.strptime(s, "%a %b %d %H:%M:%S %Y"))
    except ValueError:
        return None


def parse_qstat_json(text: str) -> list[Job]:
    """`qstat -f -F json <ids>` -> [Job]. Tolerates PBS's occasionally invalid escapes."""
    try:
        data = json.loads(text, strict=False)
    except ValueError:
        data = json.loads(re.sub(r'\\(?!["\\/bfnrtu])', r"\\\\", text), strict=False)
    jobs = []
    for jid, j in (data.get("Jobs") or {}).items():
        state = j.get("job_state", "?")
        if state in ("F", "X"):
            continue
        rl = j.get("Resource_List", {}) or {}
        ru = j.get("resources_used") or {}
        nodes = rl.get("nodect") or rl.get("select") or 1
        try:
            nodes = int(str(nodes).split(":")[0])
        except ValueError:
            nodes = 1
        jobs.append(Job(
            id=_short(jid), name=j.get("Job_Name", ""), queue=j.get("queue", ""),
            state=state, nodes=nodes,
            walltime_min=_parse_hms(rl.get("walltime", "0")) if rl.get("walltime") else 0,
            stime=_parse_pbs_time(j.get("stime")),
            resources_used=bool(ru.get("walltime") or ru.get("cput")),
            depend=str(rl.get("depend", j.get("depend", "")))))
    return jobs


def parse_qstat_u(text: str) -> list[str]:
    """Short job ids from `qstat -u <user>` (ids are truncated there, digits are enough)."""
    return re.findall(r"^(\d+)\.", text, flags=re.M)


def qsub_argv(script, queue, nodes, walltime_min, name, vars, depend=None) -> list[str]:
    wt = int(round(walltime_min))
    argv = ["qsub", "-q", queue, "-l", f"select={nodes}",
            "-l", f"walltime={wt // 60:02d}:{wt % 60:02d}:00", "-N", name]
    if depend:
        argv += ["-W", f"depend={depend}"]
    if vars:
        argv += ["-v", ",".join(f"{k}={v}" for k, v in vars.items())]
    argv.append(script)
    return argv


class PBSBackend:
    def __init__(self, repo, log_dir, user=None, allow_side_effects=False):
        self.repo = Path(repo)
        self.log_dir = Path(log_dir)
        self.user = user or os.environ.get("USER", "")
        self.allow = allow_side_effects

    def now(self):
        return time.time()

    def _run(self, argv, timeout=60):
        return subprocess.run(argv, capture_output=True, text=True, timeout=timeout,
                              cwd=self.repo)

    def jobs(self):
        out = self._run(["qstat", "-u", self.user])
        ids = parse_qstat_u(out.stdout)
        if not ids:
            return []
        full = self._run(["qstat", "-f", "-F", "json", *ids])
        return parse_qstat_json(full.stdout or "{}")

    def submit(self, script, queue, nodes, walltime_min, name, vars, depend=None):
        if not self.allow:
            raise SideEffectRefused("PBSBackend built without allow_side_effects")
        argv = qsub_argv(script, queue, nodes, walltime_min, name, vars, depend)
        res = self._run(argv)
        text = (res.stdout + res.stderr).strip()
        if "would exceed" in text:
            raise QueueLimit(text)
        m = re.search(r"^(\d+)\.", res.stdout.strip())
        if res.returncode != 0 or not m:
            raise SubmitError(text or f"qsub exit {res.returncode}")
        return m.group(1)

    def delete(self, job_id, force=False):
        if not self.allow:
            raise SideEffectRefused("PBSBackend built without allow_side_effects")
        argv = ["qdel"] + (["-W", "force"] if force else []) + [str(job_id)]
        res = self._run(argv)
        return res.returncode == 0

    def log_path(self, job_id):
        hits = sorted(self.log_dir.glob(f"{job_id}.*.OU"))
        return hits[0] if hits else None


# ------------------------------------------------------------------------------------
# Fake backend
# ------------------------------------------------------------------------------------

@dataclass
class Behavior:
    """What a fake job does. Matched against the job NAME by substring (first match wins).

    outcome: ok | fail | partial | hung | vanish | stall
      ok       exits 0, writes manifest + notification
      fail     exits 1 (no arms ok)
      partial  arms_ok of arms_total finish, exit 1
      hung     R with no resources_used, never ends; plain qdel does not remove it (FT27)
      stall    R with resources_used, but progress files stop growing; ends at walltime
      vanish   disappears at runtime without a notification
    A runtime longer than the walltime is a walltime kill. on_resume applies to jobs
    submitted with RESUME_JOB or as a rerun (their name carries -r<N>).
    """
    runtime_min: float | None = None
    outcome: str = "ok"
    arms_total: int | None = None
    arms_ok: int | None = None
    on_resume: str = "ok"
    commit: str = "c0ffee0000000000000000000000000000000000"


@dataclass
class FakeJob:
    id: str
    name: str
    queue: str
    nodes: int
    walltime_min: float
    script: str
    vars: dict
    depend: str = ""
    state: str = "Q"
    submitted: float = 0
    stime: float | None = None
    deleted_soft: bool = False


class FakeBackend:
    """Simulates Aurora's queues and per-user limits in fake minutes.

    Limits are taken from the scheduler config (queues + generic_q_limit): a submission
    that would exceed a queue's queued limit or the generic Q limit is REFUSED the way qsub
    refuses it. Jobs start FIFO when a running slot is free, finish per their Behavior and
    write notification / manifest files in the job_finish.sh schema.
    """

    def __init__(self, cfg, behaviors=None, start=None, first_id=9000000, state_file=None):
        self.cfg = cfg
        self.behaviors = list((behaviors or {}).items())
        self.t = float(int(time.time())) if start is None else start
        self.t0 = self.t
        self.next_id = first_id
        self.live: dict[str, FakeJob] = {}
        self.ended: dict[str, dict] = {}
        self.submissions: list[dict] = []
        self.deletions: list[tuple] = []
        self.state_file = Path(state_file) if state_file else None
        if self.state_file and self.state_file.exists():
            self._load()

    # -- persistence (only for CLI use with --backend fake) --------------------------
    def _load(self):
        d = json.loads(self.state_file.read_text())
        self.t, self.next_id = d["t"], d["next_id"]
        self.t0 = d.get("t0", self.t)
        self.live = {k: FakeJob(**v) for k, v in d["live"].items()}
        self.ended, self.submissions = d["ended"], d["submissions"]

    def save(self):
        if self.state_file:
            contract.write_json_atomic(self.state_file, {
                "t": self.t, "t0": self.t0, "next_id": self.next_id, "ended": self.ended,
                "submissions": self.submissions,
                "live": {k: asdict(v) for k, v in self.live.items()}})

    # -- interface ------------------------------------------------------------------
    def now(self):
        return self.t

    def jobs(self):
        out = []
        for j in self.live.values():
            b = self._behavior(j)
            outcome = b.on_resume if self._is_resume(j) else b.outcome
            out.append(Job(id=j.id, name=j.name, queue=j.queue, state=j.state,
                           nodes=j.nodes, walltime_min=j.walltime_min, stime=j.stime,
                           resources_used=j.state == "R" and outcome != "hung",
                           depend=j.depend))
        return out

    def _q_counts(self):
        per_q, total = {}, 0
        for j in self.live.values():
            if j.state in ("Q", "H"):
                per_q[j.queue] = per_q.get(j.queue, 0) + 1
                total += 1
        return per_q, total

    def submit(self, script, queue, nodes, walltime_min, name, vars, depend=None):
        lim = self.cfg["queues"][queue]
        per_q, total = self._q_counts()
        if total >= self.cfg["generic_q_limit"]:
            raise QueueLimit(f"qsub: would exceed queue generic's per-user limit of jobs "
                             f"in 'Q' state")
        if lim.get("max_queued") is not None and per_q.get(queue, 0) >= lim["max_queued"]:
            raise QueueLimit(f"qsub: would exceed queue {queue}'s per-user limit of jobs "
                             f"in 'Q' state")
        if nodes > lim["max_nodes"] or walltime_min > lim["max_walltime_min"]:
            raise SubmitError(f"qsub: job violates queue {queue} resource limits")
        jid = str(self.next_id)
        self.next_id += 1
        self.live[jid] = FakeJob(id=jid, name=name, queue=queue, nodes=nodes,
                                 walltime_min=walltime_min, script=script, vars=dict(vars),
                                 depend=depend or "", state="H" if depend else "Q",
                                 submitted=self.t)
        self.submissions.append({"id": jid, "name": name, "queue": queue, "nodes": nodes,
                                 "walltime_min": walltime_min, "vars": dict(vars),
                                 "depend": depend or "", "t": self.t})
        return jid

    def delete(self, job_id, force=False):
        j = self.live.get(str(job_id))
        self.deletions.append((str(job_id), force))
        if j is None:
            return False
        if self._behavior(j).outcome == "hung" and j.state == "R" and not force:
            j.deleted_soft = True        # FT27: a plain qdel leaves it in R
            return True
        del self.live[j.id]
        self.ended[j.id] = {"status": "deleted", "t": self.t}
        return True

    def log_path(self, job_id):
        return None

    # -- simulation -----------------------------------------------------------------
    def _behavior(self, j) -> Behavior:
        for pat, b in self.behaviors:
            if pat in j.name:
                return b
        return Behavior()

    @staticmethod
    def _is_resume(j) -> bool:
        return bool(j.vars.get("RESUME_JOB")) or bool(re.search(r"-r\d+$", j.name))

    def _runtime(self, j) -> float:
        b = self._behavior(j)
        if b.runtime_min is not None:
            rt = b.runtime_min
        else:
            rt = j.walltime_min / self.cfg["walltime_factor"]
        if self._is_resume(j):
            rt = min(rt, j.walltime_min * 0.5)
        return rt

    def advance(self, minutes: float, step: float = 1.0):
        end = self.t + minutes * 60
        while self.t < end:
            self.t = min(end, self.t + step * 60)
            self._step()
        self.save()

    def _step(self):
        # finish / kill running jobs
        for j in list(self.live.values()):
            if j.state != "R":
                continue
            b = self._behavior(j)
            outcome = b.on_resume if self._is_resume(j) else b.outcome
            elapsed = (self.t - j.stime) / 60
            if outcome == "hung":
                continue
            if outcome == "stall":
                if elapsed >= j.walltime_min:
                    self._finish(j, "walltime", b)
                continue
            self._touch_progress(j)
            rt = self._runtime(j)
            if rt > j.walltime_min and elapsed >= j.walltime_min:
                self._finish(j, "walltime", b)
            elif elapsed >= rt and rt <= j.walltime_min:
                self._finish(j, outcome, b)
        # release / drop afterok dependents
        for j in list(self.live.values()):
            if j.state == "H" and j.depend.startswith("afterok:"):
                parent = j.depend.split(":", 1)[1]
                if parent in self.live:
                    continue
                if self.ended.get(parent, {}).get("status") == "ok":
                    j.state = "Q"
                else:                          # PBS deletes an unsatisfiable afterok job
                    del self.live[j.id]
                    self.ended[j.id] = {"status": "dependency-deleted", "t": self.t}
        # start queued jobs, FIFO per queue
        for j in sorted(self.live.values(), key=lambda x: (x.submitted, int(x.id))):
            if j.state != "Q":
                continue
            lim = self.cfg["queues"][j.queue]
            running = sum(1 for x in self.live.values() if x.queue == j.queue and x.state == "R")
            if running < lim["max_running"]:
                j.state, j.stime = "R", self.t

    def _write_outputs(self, j, status, arms_ok):
        """What the real scripts leave behind: sweep run dirs (finished arms get final/),
        an eval OUT_DIR. Only ever under the fake scratch / fake output root."""
        out_job = j.vars.get("RESUME_JOB") or j.id
        if j.vars.get("ARMS_FILE"):
            path = Path(self.cfg["repo"]) / j.vars["ARMS_FILE"]
            arms = path.read_text().split() if path.exists() else []
            for i, arm in enumerate(arms):
                d = Path(self.cfg["scratch_root"]) / "runs" / f"sweep-{arm}-{out_job}"
                (d / "logs").mkdir(parents=True, exist_ok=True)
                if status == "ok" or i < (arms_ok or 0):
                    (d / "final").mkdir(exist_ok=True)
        if j.vars.get("OUT_DIR") and status == "ok":
            (Path(self.cfg["output_root"]) / j.vars["OUT_DIR"]).mkdir(parents=True, exist_ok=True)

    def _progress_files(self, j) -> list:
        """Files a healthy running job keeps writing: each arm's train.log for a sweep."""
        runs = Path(self.cfg["scratch_root"]) / "runs"
        if j.vars.get("ARMS_FILE"):
            path = Path(self.cfg["repo"]) / j.vars["ARMS_FILE"]
            out_job = j.vars.get("RESUME_JOB") or j.id
            arms = path.read_text().split() if path.exists() else []
            return [runs / f"sweep-{a}-{out_job}" / "logs" / "train.log" for a in arms]
        return [runs / f"fake-progress-{j.id}.log"]

    def _touch_progress(self, j):
        for p in self._progress_files(j):
            p.parent.mkdir(parents=True, exist_ok=True)
            p.touch()
            os.utime(p, (self.t, self.t))

    def _finish(self, j, outcome, b: Behavior):
        del self.live[j.id]
        status = {"ok": "ok", "fail": "failed", "partial": "partial",
                  "walltime": "walltime", "vanish": "vanish"}.get(outcome, "failed")
        runtime = int(self.t - j.stime)
        self.ended[j.id] = {"status": status, "t": self.t}
        if status == "vanish":
            return
        total = b.arms_total
        if total is None and j.vars.get("ARMS_FILE"):
            path = Path(self.cfg["repo"]) / j.vars["ARMS_FILE"]
            total = len(path.read_text().split()) if path.exists() else None
        arms_ok = None
        if total is not None:
            arms_ok = {"ok": total, "partial": b.arms_ok if b.arms_ok is not None else total // 2,
                       "walltime": b.arms_ok if b.arms_ok is not None else total // 2,
                       "failed": 0}[status]
        self._write_outputs(j, status, arms_ok)
        node = j.vars.get("DAG_NODE", j.name)
        common = {"schema": 1, "job_id": j.id, "job_name": j.name, "node": node,
                  "pipeline": j.vars.get("DAG_PIPELINE", ""), "kind": j.vars.get("DAG_KIND", ""),
                  "commit": b.commit, "branch": "fake", "dirty": False, "snapshot": "",
                  "outputs": [], "checks": [{"name": "fake", "ok": True}],
                  "arms_ok": arms_ok, "arms_total": total,
                  "resume_job": j.vars.get("RESUME_JOB", ""), "runtime_sec": runtime}
        manifest = None
        if status == "ok":
            manifest = contract.manifest_path(self.cfg["manifests_dir"], j.id)
            contract.write_json_atomic(manifest, dict(common, written=contract.iso(self.t)))
        note = dict(common, source="job", status=status,
                    exit_code=0 if status == "ok" else (143 if status == "walltime" else 1),
                    signal="TERM" if status == "walltime" else "",
                    start=contract.iso(j.stime), end=contract.iso(self.t), queue=j.queue,
                    nodes=j.nodes, host="fake", workdir=self.cfg["repo"],
                    manifest=str(manifest) if manifest else None,
                    summary=f"fake {status}" + (f", arms {arms_ok}/{total}" if total else ""))
        contract.write_json_atomic(
            Path(self.cfg["notifications_dir"]) /
            contract.notification_name(self.t, j.id, node, status), note)
