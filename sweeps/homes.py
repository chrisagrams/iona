"""The data homes (configs/homes.env, notes/AGENT_PLAYBOOK_2.md C7) for Python scripts: stdlib only, no msdelta import.

    from homes import EVAL, DIAG, DERIVED, RESULTS, RUNS, STORAGE     # scripts in sweeps/ (their directory is on sys.path)

Environment variables of the same name (MSDELTA_EVAL, ...) override the file, as in pbs/lib/homes.sh. Inside a code
snapshot (no configs/) the file is read from the checkout in REPO_DIR, or the exported MSDELTA_* values are used.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
_NAMES = ("MSDELTA_STORAGE", "MSDELTA_RUNS", "MSDELTA_EVAL", "MSDELTA_DIAG", "MSDELTA_DERIVED")


def _env_file() -> Path | None:
    """configs/homes.env of this tree, else of the checkout a job names in REPO_DIR (a code snapshot has no configs/)."""
    for root in (REPO, os.environ.get("REPO_DIR")):
        if root and (Path(root) / "configs" / "homes.env").is_file():
            return Path(root) / "configs" / "homes.env"
    return None


def _load() -> tuple[dict[str, str], Path | None]:
    path = _env_file()
    if path is None:
        missing = [n for n in _NAMES + ("MSDELTA_RESULTS",) if not os.environ.get(n)]
        if missing:
            raise RuntimeError(f"homes: configs/homes.env not found (looked in {REPO} and REPO_DIR="
                               f"{os.environ.get('REPO_DIR')}) and {missing} are not set")
        return {n: os.environ[n] for n in _NAMES}, None
    values: dict[str, str] = {}
    for line in path.read_text().splitlines():
        m = re.fullmatch(r"(MSDELTA_[A-Z_]+)=(.*)", line.strip())
        if m:
            values[m[1]] = os.environ.get(m[1]) or re.sub(r"\$(MSDELTA_[A-Z_]+)", lambda v: values[v[1]], m[2])
    return values, path.parents[1]


_V, _CHECKOUT = _load()
STORAGE = Path(_V["MSDELTA_STORAGE"])
RUNS = Path(_V["MSDELTA_RUNS"])          # primary: training runs, <run>-<job>/
EVAL = Path(_V["MSDELTA_EVAL"])          # primary: evaluation outputs, eval/<track>/<eval run>/
DIAG = Path(_V["MSDELTA_DIAG"])          # primary: diagnostic outputs, diag/<name>/
DERIVED = Path(_V["MSDELTA_DERIVED"])    # derived: machine-readable aggregates, _derived/<name>/
RESULTS = Path(os.environ.get("MSDELTA_RESULTS") or _CHECKOUT / "results")   # results: Markdown + plots, results/<report>/
