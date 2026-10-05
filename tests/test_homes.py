"""K198a: the data homes (configs/homes.env, notes/AGENT_PLAYBOOK_2.md C7) are defined once and read the same way.

pbs/lib/homes.sh exports them to jobs and shells, sweeps/homes.py gives them to Python. In both an exported value wins,
and both still work inside a code snapshot (pbs/lib/code_snapshot.sh copies no configs/): from the checkout named by
REPO_DIR, or from the values the job exported before it snapshotted. With neither they fail loudly instead of writing
to a wrong place. Every script that uses a home sources homes.sh before its first use.
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NAMES = ("MSDELTA_STORAGE", "MSDELTA_RUNS", "MSDELTA_EVAL", "MSDELTA_DIAG", "MSDELTA_DERIVED", "MSDELTA_RESULTS")
USE = re.compile(r"\$\{?MSDELTA_(STORAGE|RUNS|EVAL|DIAG|DERIVED|RESULTS)\b")


def _expected() -> list[str]:
    values: dict[str, str] = {}
    for line in (ROOT / "configs" / "homes.env").read_text().splitlines():
        m = re.fullmatch(r"(MSDELTA_[A-Z_]+)=(.*)", line.strip())
        if m:
            values[m[1]] = re.sub(r"\$(MSDELTA_[A-Z_]+)", lambda v: values[v[1]], m[2])
    values["MSDELTA_RESULTS"] = str(ROOT / "results")
    return [values[n] for n in NAMES]


def _env(**extra) -> dict[str, str]:
    env = {k: v for k, v in os.environ.items() if k not in NAMES + ("REPO_DIR",)}
    env.update(extra)
    return env


def _py(tree: Path, env: dict[str, str]) -> subprocess.CompletedProcess:
    code = (f"import sys; sys.path.insert(0, {str(tree / 'sweeps')!r}); import homes; "
            "print(*(getattr(homes, n) for n in ('STORAGE', 'RUNS', 'EVAL', 'DIAG', 'DERIVED', 'RESULTS')))")
    return subprocess.run([sys.executable, "-c", code], cwd=tree, env=env, capture_output=True, text=True)


def _sh(tree: Path, env: dict[str, str]) -> subprocess.CompletedProcess:
    cmd = f'source "{tree}/pbs/lib/homes.sh" && echo ' + " ".join(f'"${n}"' for n in NAMES)
    return subprocess.run(["bash", "-c", cmd], env=env, capture_output=True, text=True)


def _snapshot(tmp_path: Path) -> Path:
    """The two loaders without configs/, as a code snapshot has them."""
    (tmp_path / "sweeps").mkdir()
    (tmp_path / "pbs" / "lib").mkdir(parents=True)
    shutil.copy(ROOT / "sweeps" / "homes.py", tmp_path / "sweeps")
    shutil.copy(ROOT / "pbs" / "lib" / "homes.sh", tmp_path / "pbs" / "lib")
    return tmp_path


def test_shell_and_python_read_the_same_homes():
    for run in (_py, _sh):
        out = run(ROOT, _env())
        assert out.returncode == 0, out.stderr
        assert out.stdout.split() == _expected(), run.__name__


def test_an_exported_value_wins():
    want = _expected()
    want[NAMES.index("MSDELTA_EVAL")] = "/tmp/elsewhere"
    for run in (_py, _sh):
        assert run(ROOT, _env(MSDELTA_EVAL="/tmp/elsewhere")).stdout.split() == want, run.__name__


def test_a_snapshot_reads_the_checkout_named_by_repo_dir(tmp_path):
    tree = _snapshot(tmp_path)
    for run in (_py, _sh):
        out = run(tree, _env(REPO_DIR=str(ROOT)))
        assert out.returncode == 0, out.stderr
        assert out.stdout.split() == _expected(), run.__name__


def test_a_snapshot_uses_the_exported_values(tmp_path):
    tree = _snapshot(tmp_path)
    given = {n: f"/h/{n.lower()}" for n in NAMES}
    for run in (_py, _sh):
        out = run(tree, _env(**given))
        assert out.returncode == 0, out.stderr
        assert out.stdout.split() == list(given.values()), run.__name__


def test_a_snapshot_with_neither_fails_loudly(tmp_path):
    tree = _snapshot(tmp_path)
    for run in (_py, _sh):
        out = run(tree, _env())
        assert out.returncode != 0 and out.stdout.strip() == "", run.__name__
        assert "homes.env not found" in out.stderr, run.__name__


def _scripts():
    for dirpath, dirnames, filenames in os.walk(ROOT):
        dirnames[:] = [d for d in dirnames if not d.startswith(".") and d not in ("results", "notes")]
        for name in filenames:
            p = Path(dirpath) / name
            if p.suffix in (".pbs", ".sh") and p.is_file():
                yield p


def test_every_script_sources_homes_before_using_one():
    offenders = []
    for p in _scripts():
        if p == ROOT / "pbs" / "lib" / "homes.sh":
            continue
        lines = p.read_text(errors="replace").splitlines()
        uses = [i for i, ln in enumerate(lines) if USE.search(ln) and not ln.lstrip().startswith("#")]
        if not uses:
            continue
        src = [i for i, ln in enumerate(lines) if "pbs/lib/homes.sh" in ln and ln.lstrip().startswith("source")]
        if not src or src[0] > uses[0]:
            offenders.append(f"{p.relative_to(ROOT)}:{uses[0] + 1}")
    assert not offenders, f"source pbs/lib/homes.sh before the first use of a home: {offenders}"


def test_the_feeder_exports_homes_for_its_plan_lines():
    # plan files (pbs/tools/feeder_plans/*.txt) name $MSDELTA_EVAL etc.; feeder.sh expands them with eval
    assert re.search(r'^source "\$REPO/pbs/lib/homes.sh"', (ROOT / "pbs" / "tools" / "feeder.sh").read_text(), re.M)
