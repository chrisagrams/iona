"""One scheduling round (a "tick") over the whole DAG.

A tick: (1) observe every submitted node -- notification, manifest, qstat, hung checks;
(2) block the descendants of failed nodes; (3) find ready nodes (deps succeeded, approved,
runtime known); (4) pack them onto free queue slots, longest remaining critical path first,
within every per-user limit, the pipeline budget and max_submissions_per_tick;
(5) write state, events and the heartbeat.

In DRY-RUN nothing touches the batch system and no prepare step runs: submissions, deletes
and cancels are reported as "would ...". Observations (a job finished, a manifest is valid)
are still recorded, since reading is harmless.
"""

from __future__ import annotations

import fcntl
import glob
import math
import os
import re
from pathlib import Path

from . import contract, history, spec
from .backend import QueueLimit, SubmitError

ACTIVE = "submitted"


class LockHeld(Exception):
    pass


class TickLock:
    """flock on <home>/tick.lock: two ticks never overlap (released if a tick dies)."""

    def __init__(self, home):
        self.path = Path(home) / "tick.lock"
        self.fh = None

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.fh = open(self.path, "a+")
        try:
            fcntl.flock(self.fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            self.fh.close()
            raise LockHeld(f"another tick holds {self.path}")
        self.fh.seek(0)
        self.fh.truncate()
        self.fh.write(f"{os.getpid()}\n")
        self.fh.flush()
        return self

    def __exit__(self, *exc):
        fcntl.flock(self.fh, fcntl.LOCK_UN)
        self.fh.close()


def load_state(home) -> dict:
    path = Path(home) / "state.json"
    if path.exists():
        return contract.read_manifest(path) or {"version": 1, "nodes": {}}
    return {"version": 1, "nodes": {}}


def job_name(node, attempt: int = 0) -> str:
    base = f"dag-{node.pipeline}-{node.id}".replace("~", "-")
    base = re.sub(r"[^A-Za-z0-9_.-]", "-", base)[:200]
    return base + (f"-r{attempt}" if attempt else "")


class Scheduler:
    def __init__(self, graph, cfg, backend, home, live=False, state=None):
        self.graph = graph
        self.nodes = graph["nodes"]
        self.cfg = cfg
        self.backend = backend
        self.home = Path(home)
        self.live = bool(live) and not cfg.get("dry_run", True)
        self.state = state if state is not None else load_state(home)
        self.repo = Path(cfg["repo"])
        self.actions: list[dict] = []
        self.views: dict[str, str] = {}
        self.estimates: dict[str, tuple] = {}
        self.cp: dict[str, float] = {}
        self.warnings: list[str] = []
        self.now = backend.now()

    # ---------------------------------------------------------------- helpers
    def rec(self, key) -> dict:
        r = self.state["nodes"].setdefault(key, {})
        r.setdefault("status", "waiting")
        r.setdefault("jobs", [])
        r.setdefault("resumes", 0)
        r.setdefault("retries", 0)
        r.setdefault("alerts", [])
        return r

    def act(self, kind, key, **kw):
        a = {"action": kind if self.live or kind in _OBSERVE else f"would-{kind}", "node": key}
        a.update(kw)
        self.actions.append(a)
        return a

    def alert(self, key, event, text, once=True):
        r = self.rec(key)
        tag = f"{event}:{r.get('job') or ''}:{len(r['jobs'])}"
        if once and tag in r["alerts"]:
            return
        r["alerts"].append(tag)
        self._alerts.append((key, event, text))

    def template(self, text: str, node, rec) -> str:
        def sub(m):
            ref = m.group(1)
            if ref == "job":
                return rec.get("output_job") or "<job>"
            if ref.startswith("job:"):
                dep = ref[4:]
                dep = dep if "/" in dep else f"{node.pipeline}/{dep}"
                return self.rec(dep).get("output_job") or f"<job:{ref[4:]}>"
            return {"scratch": self.cfg["scratch_root"], "repo": self.cfg["output_root"],
                    "pipeline": node.pipeline, "node": node.id}.get(ref, m.group(0))
        return re.sub(r"\{([a-z]+(?::[^}]+)?)\}", sub, str(text))

    def expected_arms(self, node):
        return len(node.sweep.arms(self.repo)) if node.sweep else None

    def is_calib(self, key):
        return key.endswith("~calib")

    # ---------------------------------------------------------------- estimates
    def _history(self):
        self.notif_rt = history.notification_runtimes(self.notifs)
        logs = history.scan_logs(self.cfg["log_dir"], self.home / "history_cache.json"
                                 if self.home.exists() else None)
        self.arm_hist = history.arm_times(logs)

    def estimate(self, key):
        if key not in self.estimates:
            node = self.nodes[key]
            calib = self.state["nodes"].get(key + "~calib", {})
            crt = calib.get("runtime_min") if calib.get("status") == "succeeded" else None
            self.estimates[key] = history.estimate(
                node, self.repo, self.notif_rt, self.arm_hist, crt, self.cfg["slots_per_node"])
        return self.estimates[key]

    def critical_paths(self):
        order = spec.topo_order(self.nodes)
        ch = spec.children(self.nodes)
        default = self.cfg["default_runtime_for_priority_min"]
        for k in reversed(order):
            if self.rec(k)["status"] == "succeeded":
                self.cp[k] = 0.0
                continue
            est = self.estimate(k)[0]
            own = default if est is None else est
            self.cp[k] = own + max([self.cp[c] for c in ch[k]] or [0.0])

    def queue_options(self, node, est):
        opts = []
        f = self.cfg["walltime_factor"]
        for q in sorted(node.queues, key=lambda q: self.cfg["queues"][q]["rank"]):
            lim = self.cfg["queues"][q]
            if node.nodes > lim["max_nodes"]:
                continue
            fit = lim.get("fit_runtime_min")
            if fit is not None and est > fit:
                continue
            step = self.cfg["walltime_round_min"].get(q, 15)
            w = math.ceil(est * f / step - 1e-6) * step
            w = max(w, step)
            if fit is not None:
                w = min(w, lim["max_walltime_min"])
            if w > lim["max_walltime_min"]:
                continue
            opts.append((q, w))
        return opts

    # ---------------------------------------------------------------- budget
    def pipeline_used(self, pipe) -> float:
        used = 0.0
        for k, n in self.nodes.items():
            if n.pipeline != pipe:
                continue
            for j in self.rec(k)["jobs"]:
                if j.get("role") == "adopted":
                    continue
                if j.get("runtime_sec") is not None:
                    used += j["nodes"] * j["runtime_sec"] / 3600
                elif not j.get("dry"):
                    used += j["nodes"] * j["walltime_min"] / 60
        return used

    # ---------------------------------------------------------------- observe
    def observe(self, jobs, now):
        for k in spec.topo_order(self.nodes):
            n, r = self.nodes[k], self.rec(k)
            if n.adopt_job and r["status"] == "waiting" and not r["jobs"]:
                r.update(status=ACTIVE, job=n.adopt_job, output_job=n.adopt_job)
                r["jobs"].append({"job": n.adopt_job, "role": "adopted", "queue": "?",
                                  "nodes": n.nodes, "walltime_min": 0, "t": now})
                self.act("adopt", k, job=n.adopt_job)
            if r["status"] != ACTIVE:
                continue
            jid = r.get("job")
            note = self.notifs.get(str(jid))
            if note is not None:
                self.on_end(k, note)
            elif jid in jobs:
                r.pop("missing_since", None)
                r["queue_state"] = jobs[jid].state
                self.check_hung(k, jobs[jid], now)
            else:
                self.on_gone(k, now)

    def on_gone(self, key, now):
        n, r = self.nodes[key], self.rec(key)
        jid = r["job"]
        if n.legacy_log_success:
            log = self.backend.log_path(jid)
            text = log.read_text(errors="replace") if log else ""
            done = history._DONE.search(text)
            exp = self.expected_arms(n)
            if done and int(done.group(1)) == int(done.group(2)) == (exp or int(done.group(2))):
                r.update(status="succeeded", job=None, legacy=True,
                         reason=f"legacy: log says {done.group(0)} (no manifest)")
                self.act("succeeded", key, job=jid, detail="legacy log contract")
                return
            if log or now - r.setdefault("missing_since", now) > 60 * self.cfg["vanish_grace_min"]:
                self.handle_failure(key, "failed", {"job_id": jid,
                                    "summary": "legacy job: no '=== done: N/N arms ok'"})
            return
        parent = n.dep_keys()[0] if n.submit_with_parent else None
        if parent and self.rec(parent)["status"] != "succeeded":
            # PBS drops an afterok child whose parent did not succeed; re-pair it later.
            r.update(status="waiting", job=None)
            self.act("unpaired", key, job=jid, detail=f"afterok parent {parent} not ok")
            return
        since = r.setdefault("missing_since", now)
        if now - since >= 60 * self.cfg["vanish_grace_min"]:
            self.handle_failure(key, "vanished", {
                "job_id": jid, "summary": "gone from qstat with no notification"})

    def on_end(self, key, note):
        n, r = self.nodes[key], self.rec(key)
        status = note.get("status")
        entry = r["jobs"][-1] if r["jobs"] else {}
        entry.update(end_status=status, runtime_sec=note.get("runtime_sec"),
                     summary=note.get("summary", ""), commit=note.get("commit", ""))
        r.pop("missing_since", None)
        r.pop("hung", None)
        if status == "ok":
            outs = [self.template(o, n, r) for o in n.outputs]
            ok, why, man = contract.verify_success(note, self.cfg["manifests_dir"],
                                                    self.expected_arms(n), outs)
            if ok:
                r.update(status="succeeded", job=None, commit=man.get("commit", ""),
                         dirty=man.get("dirty", False), reason=why,
                         runtime_min=(note.get("runtime_sec") or 0) / 60)
                self.act("succeeded", key, job=note.get("job_id"), detail=why)
                return
            status, note = "contract", dict(note, summary=why)
        self.handle_failure(key, status, note)

    def retry_role(self, node, status, arms_ok):
        """'resume' / 'rerun' / None for an ended, unsuccessful job."""
        if self.is_calib(node.key):
            return None
        if node.sweep is not None:
            if status in ("partial", "walltime", "killed", "hung") or \
                    (status == "failed" and (arms_ok or 0) > 0):
                return "resume"
            return None
        if status == "hung" or status in node.rerun_on:
            return "rerun"
        return None

    def can_retry(self, node, r, role):
        if role == "resume":
            return r["resumes"] < node.max_resumes
        if role == "rerun":
            return r["retries"] < node.max_retries
        return False

    def handle_failure(self, key, status, note):
        n, r = self.nodes[key], self.rec(key)
        jid = note.get("job_id")
        r["job"] = None
        role = self.retry_role(n, status, note.get("arms_ok"))
        if role and self.can_retry(n, r, role):
            r.update(status="waiting", next_role=role,
                     reason=f"{status} ({note.get('summary', '')}); {role} scheduled")
            self.act("retry-scheduled", key, job=jid, detail=r["reason"])
            self.alert(key, status, f"{key}: job {jid} ended {status}: "
                       f"{note.get('summary', '')}; {role} scheduled", once=False)
            return
        r.update(status="failed", reason=f"{status}: {note.get('summary', '')}"
                 + (f" (retry cap reached)" if role else ""))
        self.act("failed", key, job=jid, detail=r["reason"])
        self.alert(key, "failed", f"{key}: job {jid} {r['reason']}", once=False)

    def check_hung(self, key, job, now):
        n, r = self.nodes[key], self.rec(key)
        why = None
        if job.state == "R" and job.stime and not job.resources_used \
                and now - job.stime > 60 * self.cfg["hung_minutes"]:
            why = (f"running {int((now - job.stime) / 60)} min with no resources_used "
                   f"(FT27: script never started)")
        elif n.stall_minutes and job.state == "R" and job.stime \
                and now - job.stime > 60 * n.stall_minutes:
            paths = []
            for g in n.progress:
                paths += glob.glob(self.template(g, n, dict(r, output_job=r.get("output_job")))
                                   .replace("{current}", str(job.id)))
            newest = max((os.path.getmtime(p) for p in paths), default=None)
            if newest is None or now - newest > 60 * n.stall_minutes:
                ago = "never" if newest is None else f"{int((now - newest) / 60)} min ago"
                why = f"no progress for {n.stall_minutes} min (last write {ago})"
        if not why:
            return
        if not r.get("hung"):
            r["hung"] = why
            self.act("flag-hung", key, job=job.id, detail=why)
            self.alert(key, "hung", f"{key}: job {job.id} looks hung: {why}")
        role = "resume" if n.sweep else "rerun"
        if self.cfg["allow_force_delete"] and self.can_retry(n, r, role):
            self.act("force-delete", key, job=job.id, detail=why)
            if self.live:
                self.backend.delete(job.id, force=True)
                self.handle_failure(key, "hung", {"job_id": job.id, "summary": why})
        elif not self.cfg["allow_force_delete"]:
            self.act("hung-notify-only", key, job=job.id,
                     detail="force-delete disabled (allow_force_delete=false)")

    # ---------------------------------------------------------------- orphans
    def propagate_failures(self):
        for k in spec.topo_order(self.nodes):
            r = self.rec(k)
            failed_calib = self.is_calib(k) and r["status"] == "failed"
            if r["status"] != "failed" and not failed_calib:
                continue
            victims = spec.descendants(self.nodes, k)
            if failed_calib:
                parent = k[: -len("~calib")]
                victims = [parent] + spec.descendants(self.nodes, parent)
            for d in victims:
                rd = self.rec(d)
                if rd["status"] == "waiting":
                    rd.update(status="blocked", reason=f"upstream {k} failed")
                    self.act("blocked", d, detail=rd["reason"])
                    self.alert(d, "blocked", f"{d} blocked: upstream {k} failed")
                elif rd["status"] == ACTIVE:
                    jid = rd.get("job")
                    if self.cfg["allow_cancel_orphans"]:
                        self.act("cancel-orphan", d, job=jid, detail=f"upstream {k} failed")
                        if self.live:
                            self.backend.delete(jid)
                            rd.update(status="blocked", job=None,
                                      reason=f"cancelled: upstream {k} failed")
                    else:
                        self.act("orphan", d, job=jid,
                                 detail=f"upstream {k} failed; cancel by hand "
                                        "(allow_cancel_orphans=false)")
                        self.alert(d, "orphan", f"{d}: job {jid} is an orphan "
                                   f"(upstream {k} failed); cancel it by hand")

    # ---------------------------------------------------------------- readiness
    def gate(self, k):
        """None if node k may be submitted once it has a slot, else the reason it may not."""
        n = self.nodes[k]
        if not n.approved:
            return "NEEDS APPROVAL (no card ID) -- refused"
        if self.cards is not None and n.approved not in self.cards:
            return f"card {n.approved} not in DECISIONS.md log -- refused"
        if self.estimate(k)[0] is None:
            return ("awaiting calibration run" if n.calibration else
                    "NEEDS AN ESTIMATE (no history, no calibration)")
        return None

    def ready(self):
        self.cards = spec.logged_cards(self.cfg["decisions_file"]) \
            if self.cfg.get("require_logged_card") else None
        out = []
        for k in spec.topo_order(self.nodes):
            n, r = self.nodes[k], self.rec(k)
            if r["status"] != "waiting":
                self.views[k] = r["status"] + (f": {r['reason']}" if r.get("reason") else "")
                continue
            deps = n.dep_keys()
            pending = [d for d in deps if self.rec(d)["status"] != "succeeded"]
            if pending:
                self.views[k] = f"waiting on {', '.join(p.split('/')[-1] for p in pending)}"
                continue
            if self.is_calib(k):
                parent = k[: -len("~calib")]
                if self.estimate(parent)[0] is not None or \
                        self.rec(parent)["status"] != "waiting":
                    self.views[k] = "not needed (runtime known)"
                    continue
            why = self.gate(k)
            if why:
                self.views[k] = why
                continue
            out.append(k)
        return out

    # ---------------------------------------------------------------- allocate
    def usage(self, jobs):
        run, queued = {}, {}
        for j in jobs.values():
            if j.state in ("R", "E"):
                run[j.queue] = run.get(j.queue, 0) + 1
            elif j.state in ("Q", "H", "W", "T"):
                queued[j.queue] = queued.get(j.queue, 0) + 1
        return run, queued

    def allocate(self, cands, jobs):
        run, queued = self.usage(jobs)
        self.total_q = sum(queued.values())
        self.full = set()
        self.submitted = 0
        self.run_c, self.queued_c = run, queued
        live_names = {j.name: j.id for j in jobs.values()}
        for k in sorted(cands, key=lambda k: -self.cp.get(k, 0)):
            jid = self.try_submit(k, live_names)
            if jid is None:
                continue
            for c in spec.children(self.nodes)[k]:
                cn, cr = self.nodes[c], self.rec(c)
                if cn.submit_with_parent and cr["status"] == "waiting":
                    why = self.gate(c)
                    if why:
                        self.views[c] = f"paired with {k}, not submitted: {why}"
                        continue
                    self.try_submit(c, live_names, depend=f"afterok:{jid}")

    def slot_reason(self):
        return (f"no free slot (running {self.run_c}, queued {self.queued_c}, "
                f"Q total {self.total_q}/{self.cfg['generic_q_limit']})")

    def try_submit(self, key, live_names, depend=None):
        n, r = self.nodes[key], self.rec(key)
        est, src = self.estimate(key)
        if self.submitted >= self.cfg["max_submissions_per_tick"]:
            self.views[key] = "ready; deferred (max_submissions_per_tick)"
            return None
        opts = self.queue_options(n, est)
        if not opts:
            self.views[key] = f"ready; NO QUEUE FITS ({n.nodes} nodes, ~{est:.0f} min)"
            return None
        lims = self.cfg["queues"]
        free = [(q, w) for q, w in opts
                if q not in self.full and self.total_q < self.cfg["generic_q_limit"]
                and (lims[q].get("max_queued") is None
                     or self.queued_c.get(q, 0) < lims[q]["max_queued"])]
        if not free:
            self.views[key] = "ready; " + self.slot_reason()
            return None
        # start-now slots first, then debug before capacity (rank)
        free.sort(key=lambda o: (0 if self.run_c.get(o[0], 0) + self.queued_c.get(o[0], 0)
                                 < lims[o[0]]["max_running"] else 1, lims[o[0]]["rank"]))
        q, wall = free[0]
        pipe = self.graph["pipelines"][n.pipeline]
        cost = n.nodes * wall / 60
        used = self.pipeline_used(n.pipeline)
        if used + cost > pipe.budget_node_hours + 1e-9:
            self.views[key] = (f"ready; OVER BUDGET ({used:.1f} + {cost:.1f} > "
                               f"{pipe.budget_node_hours} node-h)")
            self.alert(key, "budget", f"{key}: submission refused, pipeline {n.pipeline} "
                       f"budget {pipe.budget_node_hours} node-h (used {used:.1f})")
            return None
        role = r.get("next_role") or ("calibration" if self.is_calib(key) else "run")
        attempt = r["resumes"] + r["retries"] + (1 if role in ("resume", "rerun") else 0)
        name = job_name(n, attempt)
        if name in live_names:                       # idempotency: never a second copy
            jid = live_names[name]
            r.update(status=ACTIVE, job=jid, output_job=r.get("output_job") or jid)
            r["jobs"].append({"job": jid, "role": "adopted-by-name", "queue": q,
                              "nodes": n.nodes, "walltime_min": wall, "t": self.now})
            self.act("adopt", key, job=jid, detail=f"live job already named {name}")
            return jid
        vars = {k2: self.template(v, n, r) for k2, v in n.all_vars().items()}
        vars.update(DAG_NODE=key, DAG_PIPELINE=n.pipeline, DAG_KIND=n.kind_key,
                    DAG_WALLTIME_SEC=str(int(wall * 60)))
        if role == "resume":
            vars["RESUME_JOB"] = r["output_job"]
        unresolved = [v for v in vars.values() if "<job" in v]
        why = (f"{'/'.join(o[0] for o in opts)} fit; cp {self.cp.get(key, 0):.0f} min; "
               f"est {est:.0f} min ({src})")
        detail = {"queue": q, "walltime_min": wall, "nodes": n.nodes, "name": name,
                  "role": role, "why": why, "script": n.script_path, "vars": vars,
                  "card": n.approved}
        if depend:
            detail["depend"] = depend
        if n.prepare:
            detail["prepare"] = n.prepare.describe
        if not self.live:
            self.act("submit", key, **detail)
            jid = f"<{key}>"
        else:
            if unresolved:
                self.views[key] = f"ready; unresolved template {unresolved}"
                return None
            if n.prepare:
                try:
                    files = n.prepare.run({"repo": self.cfg["output_root"], "node": n, "rec": r,
                                           "template": lambda s: self.template(s, n, r),
                                           "scratch": self.cfg["scratch_root"]})
                    detail["prepared"] = [str(f) for f in files or []]
                except Exception as e:                       # noqa: BLE001
                    self.views[key] = f"ready; prepare failed: {e}"
                    self.alert(key, "prepare-failed", f"{key}: prepare failed: {e}")
                    return None
            try:
                jid = self.backend.submit(n.script_path, q, n.nodes, wall, name, vars, depend)
            except QueueLimit as e:
                self.full.add(q)
                self.views[key] = f"ready; queue limit hit on {q}: {e}"
                self.act("limit-hit", key, queue=q, detail=str(e))
                return None
            except SubmitError as e:
                self.views[key] = f"ready; qsub failed: {e}"
                self.alert(key, "submit-error", f"{key}: qsub failed: {e}")
                return None
            if role == "resume":
                r["resumes"] += 1
            elif role == "rerun":
                r["retries"] += 1
            r.update(status=ACTIVE, job=jid, output_job=r.get("output_job") or jid,
                     next_role=None, reason="")
            r["jobs"].append({"job": jid, "role": role, "queue": q, "nodes": n.nodes,
                              "walltime_min": wall, "t": self.now, "name": name})
            self.act("submit", key, job=jid, **detail)
        self.queued_c[q] = self.queued_c.get(q, 0) + 1
        self.total_q += 1
        self.submitted += 1
        self.views[key] = f"{'submitted' if self.live else 'would submit'} -> {q} " \
                          f"({wall:.0f} min walltime)"
        return jid

    # ---------------------------------------------------------------- main
    def run(self, write=True):
        self._alerts = []
        jobs = {j.id: j for j in self.backend.jobs()}
        self.notifs = contract.read_notifications(self.cfg["notifications_dir"])
        self._history()
        self.observe(jobs, self.now)
        self.propagate_failures()
        self.critical_paths()
        cands = self.ready()
        self.allocate(cands, jobs)
        self.check_commits()
        if write:
            for key, event, text in self._alerts:
                contract.write_alert(self.cfg["notifications_dir"], key, event, text, self.now)
            self.save()
        return self.report()

    def check_commits(self):
        for k, n in self.nodes.items():
            commits = {d: self.rec(d).get("commit") for d in n.dep_keys()
                       if self.rec(d)["status"] == "succeeded"}
            w = contract.mixed_commits(commits)
            if w:
                self.warnings.append(f"{k} consumes results from different commits -- {w}")
            if self.rec(k).get("dirty"):
                self.warnings.append(f"{k}: snapshot had uncommitted changes")

    def save(self):
        self.home.mkdir(parents=True, exist_ok=True)
        self.state["updated"] = contract.iso(self.now)
        contract.write_json_atomic(self.home / "state.json", self.state)
        with open(self.home / "events.log", "a") as fh:
            for a in self.actions:
                fh.write(f"{contract.iso(self.now)} {a['action']} {a['node']} "
                         f"{a.get('job', '') or ''} {a.get('queue', '')} "
                         f"{a.get('detail', a.get('why', ''))}\n")
        contract.write_json_atomic(self.home / "heartbeat.json", {
            "time": self.now, "iso": contract.iso(self.now), "pid": os.getpid(),
            "live": self.live, "actions": len(self.actions),
            "submitted": sum(1 for a in self.actions if a["action"] == "submit")})

    def report(self) -> dict:
        nodes = {}
        for k in spec.topo_order(self.nodes):
            r = self.rec(k)
            est, src = self.estimate(k)
            nodes[k] = {"status": r["status"], "view": self.views.get(k, r["status"]),
                        "job": r.get("job"), "queue_state": r.get("queue_state"),
                        "est_min": est, "est_source": src, "cp_min": self.cp.get(k),
                        "commit": r.get("commit", ""), "approved": self.nodes[k].approved,
                        "attempts": len(r["jobs"])}
        budgets = {p: {"used": self.pipeline_used(p), "budget": pipe.budget_node_hours}
                   for p, pipe in self.graph["pipelines"].items()}
        return {"live": self.live, "time": contract.iso(self.now), "actions": self.actions,
                "nodes": nodes, "budgets": budgets, "warnings": self.warnings,
                "alerts": [f"{e}: {t}" for _, e, t in self._alerts]}


_OBSERVE = {"succeeded", "failed", "retry-scheduled", "blocked", "flag-hung", "orphan",
            "unpaired", "hung-notify-only", "limit-hit", "adopt"}


def heartbeat_age_min(home, now) -> float | None:
    hb = contract.read_manifest(Path(home) / "heartbeat.json")
    if not hb:
        return None
    return (now - hb["time"]) / 60
