"""Runtime estimates from history: notification files and pbs/logs/*.OU.

Order of precedence for a node's expected runtime (minutes):
  1. the spec's runtime_min;
  2. a finished calibration run of the node (extrapolated, see spec.Calibration);
  3. notifications of earlier jobs of the same kind (median runtime of the ok ones);
  4. for a sweep, per-arm times from old sweep logs ("[ok   <arm>] <secs>s on <host>"),
     packed onto the node's slots longest-first like the sweep script does, + overhead;
  5. unknown -> the node needs a calibration run (which needs approval) or an estimate.

Log parsing is cached by (path, size, mtime) in <home>/history_cache.json, so a tick costs
milliseconds after the first scan (~600 logs, ~14 MB today).
"""

from __future__ import annotations

import json
import re
import statistics
from pathlib import Path

from . import contract

_ARM_OK = re.compile(r"^\s*\[ok\s+(\S+)\]\s+(\d+)s on ", re.M)
_DONE = re.compile(r"=== done: (\d+)/(\d+) arms ok")
_STAMP = re.compile(r"^=== (\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})", re.M)
SWEEP_OVERHEAD_MIN = 10.0


def parse_log(text: str) -> dict:
    arms = {}
    for name, secs in _ARM_OK.findall(text):
        arms[name] = int(secs)
    done = _DONE.search(text)
    return {"arms": arms, "done": [int(done.group(1)), int(done.group(2))] if done else None}


def scan_logs(log_dir, cache_file=None) -> dict:
    """{job_id: parse_log(...)} for pbs/logs/*.OU, cached."""
    cache = {}
    if cache_file and Path(cache_file).exists():
        try:
            cache = json.loads(Path(cache_file).read_text())
        except ValueError:
            cache = {}
    out, changed = {}, False
    d = Path(log_dir)
    if not d.is_dir():
        return out
    for path in d.glob("*.OU"):
        st = path.stat()
        key = f"{st.st_size}:{int(st.st_mtime)}"
        hit = cache.get(path.name)
        if hit and hit.get("key") == key:
            out[path.name.split(".")[0]] = hit["data"]
            continue
        try:
            data = parse_log(path.read_text(errors="replace"))
        except OSError:
            continue
        cache[path.name] = {"key": key, "data": data}
        out[path.name.split(".")[0]] = data
        changed = True
    if cache_file and changed:
        contract.write_json_atomic(cache_file, cache)
    return out


def arm_times(logs: dict) -> dict:
    """{arm: [seconds, ...]} over every successful arm in every sweep log."""
    times = {}
    for data in logs.values():
        for arm, secs in data["arms"].items():
            times.setdefault(arm, []).append(secs)
    return times


def lpt_makespan(costs: list, slots: int) -> float:
    loads = [0.0] * max(1, slots)
    for c in sorted(costs, reverse=True):
        i = loads.index(min(loads))
        loads[i] += c
    return max(loads)


def estimate_sweep_from_logs(arms: list, times: dict, slots: int) -> float | None:
    if not arms or any(a not in times for a in arms):
        return None
    costs = [statistics.median(times[a]) / 60 for a in arms]
    return lpt_makespan(costs, slots) + SWEEP_OVERHEAD_MIN


def notification_runtimes(notifications: dict) -> dict:
    """{kind: [minutes, ...]} over ok notifications that carry a kind."""
    out = {}
    for n in notifications.values():
        if n.get("status") == "ok" and n.get("kind") and n.get("runtime_sec") is not None:
            out.setdefault(n["kind"], []).append(n["runtime_sec"] / 60)
    return out


def calibrated(node, calib_runtime_min: float) -> float:
    c = node.calibration
    work = max(0.0, calib_runtime_min - c.overhead_min)
    return c.overhead_min + work * c.full_steps / max(1, c.steps)


def estimate(node, repo, notif_runtimes: dict, arm_hist: dict, calib_runtime_min=None,
             slots_per_node: int = 12) -> tuple[float | None, str]:
    """(minutes or None, source)."""
    if node.runtime_min is not None:
        return float(node.runtime_min), "spec"
    if calib_runtime_min is not None and node.calibration is not None:
        return calibrated(node, calib_runtime_min), "calibration"
    runs = notif_runtimes.get(node.kind_key)
    if runs:
        return statistics.median(runs), f"history ({len(runs)} runs)"
    if node.sweep is not None:
        tiles = int(node.vars.get("TILES_PER_ARM", 1))
        slots = node.nodes * slots_per_node // max(1, tiles)
        est = estimate_sweep_from_logs(node.sweep.arms(Path(repo)), arm_hist, slots)
        if est is not None:
            return est, "sweep logs (per-arm)"
    return None, "unknown"
