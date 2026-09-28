"""The job side of the contract: notification files and the _SUCCESS.json manifest.

pbs/lib/job_finish.sh writes both from inside a PBS job; FakeBackend writes the same
schema from Python. This module reads them and decides whether a node succeeded.

  notifications:  <notifications_dir>/<UTC %Y%m%dT%H%M%SZ>_<jobid>_<node>_<status>.json
                  one per job end (ok / failed / partial / walltime / killed); written
                  atomically (dot-tmp + rename), so a reader never sees half a file.
  manifest:       <manifests_dir>/<jobid>/_SUCCESS.json, written LAST and only on success
                  (exit 0, every declared output present, every check passed).
Scheduler alerts go to <notifications_dir>/scheduler/ so they sit next to job notices.
"""

from __future__ import annotations

import glob
import json
import os
import re
import time
from pathlib import Path

SCHEMA = 1
END_STATUSES = ("ok", "failed", "partial", "walltime", "killed")
_FNAME = re.compile(r"^(\d{8}T\d{6}Z)_([^_]+)_(.+)_([a-z]+)\.json$")


def utc_stamp(t: float | None = None) -> str:
    return time.strftime("%Y%m%dT%H%M%SZ", time.gmtime(time.time() if t is None else t))


def iso(t: float | None = None) -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(time.time() if t is None else t))


def write_json_atomic(path, obj) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.parent / f".{path.name}.tmp.{os.getpid()}"
    tmp.write_text(json.dumps(obj, indent=1, sort_keys=True) + "\n")
    os.replace(tmp, path)
    return path


def safe_node(node: str) -> str:
    return re.sub(r"[^A-Za-z0-9.~-]+", "-", node or "none")


def notification_name(t: float, job_id: str, node: str, status: str) -> str:
    return f"{utc_stamp(t)}_{job_id}_{safe_node(node)}_{status}.json"


def read_notifications(directory) -> dict:
    """{job_id: notification} -- the newest per job; dotfiles and bad files are skipped."""
    out = {}
    d = Path(directory)
    if not d.is_dir():
        return out
    for path in sorted(d.iterdir()):
        if not path.is_file() or path.name.startswith("."):
            continue
        m = _FNAME.match(path.name)
        if not m:
            continue
        try:
            data = json.loads(path.read_text())
        except (OSError, ValueError):
            continue
        if data.get("source", "job") != "job":
            continue
        data.setdefault("job_id", m.group(2))
        data["_file"] = str(path)
        out[str(data["job_id"])] = data
    return out


def read_manifest(path) -> dict | None:
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError, TypeError):
        return None


def manifest_path(manifests_dir, job_id) -> Path:
    return Path(manifests_dir) / str(job_id) / "_SUCCESS.json"


def verify_success(notif: dict, manifests_dir, expected_arms: int | None,
                   spec_outputs: list) -> tuple[bool, str, dict | None]:
    """A node succeeded only if its manifest exists and says so. Returns (ok, reason, manifest)."""
    job = str(notif.get("job_id"))
    if notif.get("status") != "ok":
        return False, f"job ended {notif.get('status')}", None
    path = notif.get("manifest") or manifest_path(manifests_dir, job)
    man = read_manifest(path)
    if man is None:
        return False, f"no _SUCCESS.json at {path}", None
    if str(man.get("job_id")) != job:
        return False, f"manifest is for job {man.get('job_id')}, not {job}", man
    failed = [c.get("name") for c in man.get("checks", []) if not c.get("ok")]
    if failed:
        return False, f"checks failed: {failed}", man
    missing = [o.get("path") for o in man.get("outputs", []) if not o.get("exists")]
    if missing:
        return False, f"declared outputs missing: {missing[:3]}", man
    if expected_arms is not None:
        ok, tot = man.get("arms_ok"), man.get("arms_total")
        if ok != expected_arms or tot != expected_arms:
            return False, f"arms {ok}/{tot}, expected {expected_arms}/{expected_arms}", man
    absent = [p for p in spec_outputs if not glob.glob(p)]
    if absent:
        return False, f"spec outputs missing: {absent[:3]}", man
    return True, f"manifest ok ({man.get('commit', '?')[:8]})", man


def write_alert(directory, node: str, event: str, text: str, t: float | None = None,
                extra: dict | None = None) -> Path:
    t = time.time() if t is None else t
    obj = {"schema": SCHEMA, "source": "scheduler", "node": node, "event": event,
           "time": iso(t), "text": text}
    obj.update(extra or {})
    return write_json_atomic(Path(directory) / "scheduler" /
                             f"{utc_stamp(t)}_scheduler_{safe_node(node)}_{event}.json", obj)


def mixed_commits(commits: dict) -> str | None:
    """commits: {node: commit}. A warning if results being compared come from >1 commit."""
    known = {k: c for k, c in commits.items() if c}
    distinct = sorted({c[:12] for c in known.values()})
    if len(distinct) > 1:
        groups = {}
        for k, c in known.items():
            groups.setdefault(c[:8], []).append(k)
        return "MIXED COMMITS: " + "; ".join(f"{c}: {', '.join(sorted(v))}"
                                             for c, v in sorted(groups.items()))
    return None


def parse_snapshot(text: str) -> dict:
    """SNAPSHOT.txt (pbs/lib/code_snapshot.sh) -> {commit, branch, dirty}."""
    out = {"commit": "", "branch": "", "dirty": False}
    lines = text.splitlines()
    for i, line in enumerate(lines):
        if line.startswith("commit:"):
            out["commit"] = line.split(":", 1)[1].strip()
        elif line.startswith("branch:"):
            out["branch"] = line.split(":", 1)[1].strip()
        elif line.startswith("uncommitted changes"):
            out["dirty"] = any(x.strip() for x in lines[i + 1:])
    return out
