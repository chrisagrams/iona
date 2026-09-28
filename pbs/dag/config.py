"""Scheduler configuration: queue limits, safety switches, paths.

Defaults live here; ``<home>/config.json`` overrides any top-level key (the user edits it,
e.g. to turn dry-run off). Queue facts are the user's (Aurora, 2026-09-27).
"""

from __future__ import annotations

import copy
import json
import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SCRATCH_ROOT = os.environ.get(
    "SCRATCH_ROOT",
    f"/lus/flare/projects/{os.environ.get('PROJECT', 'UIC-HPC')}/{os.environ.get('USER', 'khuss')}/msdelta")

DEFAULTS = {
    # --- safety -----------------------------------------------------------------------
    "dry_run": True,                 # decision 7: ON until the user turns it off
    "allow_force_delete": False,     # decision 6: hung jobs are only flagged by default
    "allow_cancel_orphans": False,   # decision 8: cancelling afterok children of a failed parent
    "max_submissions_per_tick": 4,
    "require_logged_card": True,     # the card ID must appear in notes/DECISIONS.md's log
    "decisions_file": str(REPO / "notes" / "DECISIONS.md"),
    # --- queues -----------------------------------------------------------------------
    # max_running / max_queued are PER USER; fit_runtime_min is the longest expected
    # runtime we send there (debug limits are 1 h; 50 min leaves margin).
    "queues": {
        "debug":         {"max_running": 1, "max_queued": 1, "max_nodes": 2,
                          "max_walltime_min": 60, "fit_runtime_min": 50, "rank": 0},
        "debug-scaling": {"max_running": 1, "max_queued": 1, "max_nodes": 2,
                          "max_walltime_min": 60, "fit_runtime_min": 50, "rank": 1},
        "capacity":      {"max_running": 2, "max_queued": 2, "max_nodes": 16,
                          "max_walltime_min": 24 * 60, "fit_runtime_min": None, "rank": 2},
    },
    # Aurora's per-user limit on jobs in Q state (held jobs count), across queues. qsub says
    # "would exceed queue generic's per-user limit of jobs in 'Q' state" -- hit at ~2-3.
    "generic_q_limit": 2,
    "walltime_factor": 1.3,          # capacity walltime = 1.3 x estimate
    "walltime_round_min": {"debug": 5, "debug-scaling": 5, "capacity": 15},
    "default_runtime_for_priority_min": 60,
    # --- health -----------------------------------------------------------------------
    "hung_minutes": 10,              # FT27: R with no resources_used after this long
    "vanish_grace_min": 5,           # gone from qstat without a notification -> failed
    "heartbeat_stale_min": 30,
    # --- paths ------------------------------------------------------------------------
    "scratch_root": SCRATCH_ROOT,
    "notifications_dir": None,       # default <scratch_root>/notifications
    "manifests_dir": None,           # default <scratch_root>/manifests
    "log_dir": str(REPO / "pbs" / "logs"),
    "repo": str(REPO),
    # where {repo} in declared outputs and prepare steps point; the fake backend moves it
    # under <home>/fake-scratch so a simulation never writes into the checkout
    "output_root": None,
    "pipelines_dir": str(Path(__file__).resolve().parent / "pipelines"),
    "slots_per_node": 12,
}


def default_home() -> Path:
    return Path(os.environ.get("DAG_HOME", f"{SCRATCH_ROOT}/dag"))


def load_config(home: Path, overrides: dict | None = None) -> dict:
    cfg = copy.deepcopy(DEFAULTS)
    path = Path(home) / "config.json"
    if path.exists():
        user = json.loads(path.read_text())
        for k, v in user.items():
            if k == "queues":
                for q, lim in v.items():
                    cfg["queues"].setdefault(q, {}).update(lim)
            else:
                cfg[k] = v
    for k, v in (overrides or {}).items():
        if k == "queues":
            for q, lim in v.items():
                cfg["queues"].setdefault(q, {}).update(lim)
        else:
            cfg[k] = v
    cfg["notifications_dir"] = cfg["notifications_dir"] or f"{cfg['scratch_root']}/notifications"
    cfg["manifests_dir"] = cfg["manifests_dir"] or f"{cfg['scratch_root']}/manifests"
    cfg["output_root"] = cfg["output_root"] or cfg["repo"]
    return cfg
