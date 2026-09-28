"""dagctl: validate / plan / tick / status / commits / simulate.

    pbs/dagctl validate
    pbs/dagctl plan   [--backend fake]      # what would be submitted where and why
    pbs/dagctl tick                         # one round; DRY-RUN unless --live AND
                                            #   config.json has "dry_run": false
    pbs/dagctl status                       # DAG, node states, next actions, budget, heartbeat
    pbs/dagctl commits k66c/score_400m_validation k66c/score_200m_validation
    pbs/dagctl simulate --backend fake      # run the DAG to the end on the fake backend

--home defaults to $DAG_HOME or <scratch>/dag. The fake backend keeps its own state and
writes its notifications under <home>/fake-scratch, never into the real scratch.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import sys
from pathlib import Path

from . import backend as be
from . import config, contract, scheduler, spec


def _fmt_min(m):
    if m is None:
        return "?"
    return f"{m / 60:.1f}h" if m >= 90 else f"{m:.0f}m"


def build(args):
    home = Path(args.home) if args.home else config.default_home()
    over = {}
    if args.pipelines_dir:
        over["pipelines_dir"] = args.pipelines_dir
    if getattr(args, "backend", "pbs") == "fake":
        fs = home / "fake-scratch"
        over.update(scratch_root=str(fs), notifications_dir=str(fs / "notifications"),
                    manifests_dir=str(fs / "manifests"), output_root=str(fs / "repo"))
    cfg = config.load_config(home, over)
    graph = spec.load(Path(cfg["pipelines_dir"]), args.pipeline or None)
    return home, cfg, graph


def make_backend(args, cfg, home, live=False):
    if getattr(args, "backend", "pbs") == "fake":
        return be.FakeBackend(cfg, state_file=home / "fake_backend.json")
    return be.PBSBackend(cfg["repo"], cfg["log_dir"], allow_side_effects=live)


def print_actions(rep, verbose=True):
    acts = rep["actions"]
    if not acts:
        print("  (no actions)")
    for a in acts:
        line = f"  {a['action']:<18} {a['node']:<34}"
        if "queue" in a:
            line += f" -> {a['queue']:<13} {_fmt_min(a['walltime_min']):>6} x{a['nodes']}"
            line += f"  [{a['role']}, card {a['card']}]"
        if a.get("job"):
            line += f" job {a['job']}"
        print(line)
        if a.get("why"):
            print(f"      why: {a['why']}")
        if a.get("detail"):
            print(f"      {a['detail']}")
        if verbose and "vars" in a:
            argv = be.qsub_argv(a["script"], a["queue"], a["nodes"], a["walltime_min"],
                                a["name"], a["vars"], a.get("depend"))
            print("      " + " ".join(shlex.quote(x) for x in argv))
        if a.get("prepare"):
            print(f"      prepare (login node, before qsub): {a['prepare']}")


def print_nodes(rep):
    print(f"  {'node':<34} {'status':<10} {'est':>6} {'cp':>6} {'job':>9}  view")
    for k, v in rep["nodes"].items():
        job = v["job"] or ""
        if v.get("queue_state"):
            job = f"{job}:{v['queue_state']}" if job else ""
        print(f"  {k:<34} {v['status']:<10} {_fmt_min(v['est_min']):>6} "
              f"{_fmt_min(v['cp_min']):>6} {job:>9}  {v['view']}")


def print_budgets(rep):
    for p, b in rep["budgets"].items():
        print(f"  budget {p}: {b['used']:.1f} / {b['budget']} node-h")


def cmd_validate(args):
    home, cfg, graph = build(args)
    errors, warnings = spec.validate(graph, Path(cfg["repo"]), cfg["queues"],
                                     cfg["decisions_file"] if cfg["require_logged_card"] else None)
    n = len(graph["nodes"])
    print(f"{n} nodes in {len(graph['pipelines'])} pipeline(s)")
    for w in warnings:
        print(f"  warning: {w}")
    for e in errors:
        print(f"  ERROR: {e}")
    print("valid" if not errors else f"{len(errors)} error(s)")
    return 1 if errors else 0


def _run(args, live, write):
    home, cfg, graph = build(args)
    errors, _ = spec.validate(graph, Path(cfg["repo"]), cfg["queues"])
    if errors:
        print("spec invalid -- run validate:", *errors, sep="\n  ")
        return None, None
    backend = make_backend(args, cfg, home, live=live and not cfg["dry_run"])
    sch = scheduler.Scheduler(graph, cfg, backend, home, live=live)
    return sch, sch.run(write=write)


def cmd_plan(args):
    sch, rep = _run(args, live=False, write=False)
    if rep is None:
        return 1
    print(f"PLAN ({args.backend} backend, dry run, {rep['time']}) -- nothing is submitted")
    print_actions(rep, verbose=not args.brief)
    print()
    print_nodes(rep)
    print_budgets(rep)
    for w in rep["warnings"]:
        print(f"  WARNING: {w}")
    if args.json:
        print(json.dumps(rep, indent=1, default=str))
    return 0


def cmd_tick(args):
    home, cfg, _ = build(args)
    if args.live and cfg["dry_run"]:
        print("--live ignored: config.json has dry_run true (the user turns it off)")
    try:
        with scheduler.TickLock(home):
            sch, rep = _run(args, live=args.live, write=True)
    except scheduler.LockHeld as e:
        print(f"tick skipped: {e}")
        return 2
    if rep is None:
        return 1
    print(f"TICK {'LIVE' if rep['live'] else 'DRY-RUN'} {rep['time']}")
    print_actions(rep, verbose=False)
    for a in rep["alerts"]:
        print(f"  alert: {a}")
    return 0


def cmd_status(args):
    home, cfg, graph = build(args)
    sch, rep = _run(args, live=False, write=False)
    if rep is None:
        return 1
    age = scheduler.heartbeat_age_min(home, sch.now)
    if age is None:
        print("heartbeat: none (no tick has run)")
    elif age > cfg["heartbeat_stale_min"]:
        print(f"WARNING: heartbeat STALE -- last tick {age:.0f} min ago "
              f"(> {cfg['heartbeat_stale_min']} min); is the tick loop running?")
    else:
        print(f"heartbeat: last tick {age:.0f} min ago")
    print(f"mode: {'dry-run' if cfg['dry_run'] else 'LIVE allowed'}; "
          f"force-delete {'on' if cfg['allow_force_delete'] else 'off'}; "
          f"cancel-orphans {'on' if cfg['allow_cancel_orphans'] else 'off'}")
    print_nodes(rep)
    print_budgets(rep)
    print("next actions:")
    print_actions(rep, verbose=False)
    for w in rep["warnings"]:
        print(f"  WARNING: {w}")
    return 0


def cmd_commits(args):
    home, cfg, graph = build(args)
    state = scheduler.load_state(home)
    keys = args.nodes or sorted(graph["nodes"])
    commits = {}
    for k in keys:
        r = state["nodes"].get(k, {})
        commits[k] = r.get("commit", "")
        print(f"  {k:<40} {commits[k][:12] or '-':<12} {'(dirty)' if r.get('dirty') else ''}")
    w = contract.mixed_commits(commits)
    print(w or "all compared results come from one commit")
    return 1 if w else 0


def cmd_simulate(args):
    if args.backend != "fake":
        print("simulate runs on the fake backend only")
        return 2
    home, cfg, graph = build(args)
    # a fresh fake scratch per simulation, so notifications of an earlier run never leak in
    import time as _time
    fs = home / "fake-scratch" / f"sim-{_time.strftime('%Y%m%dT%H%M%S')}-{os.getpid()}"
    cfg.update(dry_run=False, scratch_root=str(fs), notifications_dir=str(fs / "notifications"),
               manifests_dir=str(fs / "manifests"), output_root=str(fs / "repo"))
    fake = be.FakeBackend(cfg)
    state = {"version": 1, "nodes": {}}
    for step in range(args.max_ticks):
        sch = scheduler.Scheduler(graph, cfg, fake, home, live=True, state=state)
        rep = sch.run(write=False)
        for a in rep["actions"]:
            if a["action"] in ("submit", "succeeded", "failed", "retry-scheduled", "blocked",
                               "flag-hung", "force-delete"):
                where = f" -> {a['queue']} {_fmt_min(a['walltime_min'])}" if "queue" in a else ""
                if a["action"] in ("failed", "retry-scheduled", "flag-hung"):
                    where += f"  ({a.get('detail', '')})"
                print(f"  t+{_fmt_min((fake.now() - fake.t0) / 60):>6} {a['action']:<16} "
                      f"{a['node']}{where}")
        done = all(v["status"] in ("succeeded", "failed", "blocked")
                   or v["view"].startswith(("NEEDS", "card", "not needed"))
                   for v in rep["nodes"].values())
        if done and not fake.live:
            print(f"finished at t+{_fmt_min((fake.now() - fake.t0) / 60)} "
                  f"(fake time); {len(fake.submissions)} submissions")
            print_budgets(rep)
            return 0
        fake.advance(args.every)
    print("did not finish within --max-ticks")
    return 1


def main(argv=None):
    p = argparse.ArgumentParser(prog="dagctl", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--home", help="DAG state dir (default $DAG_HOME or <scratch>/dag)")
    p.add_argument("--pipelines-dir")
    p.add_argument("--pipeline", action="append", help="restrict to these pipelines")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("validate")
    for name in ("plan", "tick", "status", "simulate"):
        s = sub.add_parser(name)
        s.add_argument("--backend", choices=["pbs", "fake"], default="pbs")
        if name == "plan":
            s.add_argument("--json", action="store_true")
            s.add_argument("--brief", action="store_true")
        if name == "tick":
            s.add_argument("--live", action="store_true")
        if name == "simulate":
            s.add_argument("--every", type=float, default=10, help="fake minutes per tick")
            s.add_argument("--max-ticks", type=int, default=2000)
    c = sub.add_parser("commits")
    c.add_argument("nodes", nargs="*")
    args = p.parse_args(argv)
    return {"validate": cmd_validate, "plan": cmd_plan, "tick": cmd_tick,
            "status": cmd_status, "commits": cmd_commits,
            "simulate": cmd_simulate}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
