"""K198a (notes/AGENT_PLAYBOOK_2.md C7): results/ holds only what people read.

One folder per report, with Markdown and plots and a README.md that says how the report was made (command, inputs,
when). Machine-readable data lives on /flare in the homes of configs/homes.env: evaluation and diagnostic outputs in
$MSDELTA_EVAL / $MSDELTA_DIAG, tables computed from them in $MSDELTA_DERIVED. results/raw/ and results/processed/
are the originals of the verified /flare copies; they stay until their removal is approved (K198b) and are ignored here.
"""
from __future__ import annotations

import os
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
LEGACY = ("raw", "processed")
REPORT_SUFFIXES = {".md", ".png", ".svg"}
OLD = re.compile(r"""results/(raw|processed)\b|["']results["']\s*/\s*["'](raw|processed)["']""")
ALLOWED = {ROOT / "pbs" / "tools" / "k198_results_move.py", Path(__file__).resolve()}

pytestmark = pytest.mark.skipif(not RESULTS.is_dir(), reason="no results/ here (a code snapshot)")


def _report_files():
    for p in RESULTS.rglob("*"):
        rel = p.relative_to(RESULTS)
        if p.is_file() and rel.parts[0] not in LEGACY:
            yield rel


def test_results_holds_only_markdown_and_plots():
    bad = [str(r) for r in _report_files() if r.suffix not in REPORT_SUFFIXES]
    assert not bad, f"data belongs in $MSDELTA_EVAL / $MSDELTA_DIAG / $MSDELTA_DERIVED, not results/: {bad}"


def test_every_report_folder_says_how_it_was_made():
    folders = {r.parts[0] for r in _report_files() if len(r.parts) > 1}
    assert folders, "no report folders"
    missing = sorted(f for f in folders if not (RESULTS / f / "README.md").is_file())
    assert not missing, f"results/<report>/README.md missing for {missing}"
    assert (RESULTS / "README.md").is_file(), "results/README.md (the index) is missing"
    index = (RESULTS / "README.md").read_text()
    unlisted = sorted(f for f in folders if f"{f}/" not in index)
    assert not unlisted, f"report folders not listed in results/README.md: {unlisted}"


def test_no_code_reads_or_writes_the_old_layout():
    offenders = []
    for dirpath, dirnames, filenames in os.walk(ROOT):
        dirnames[:] = [d for d in dirnames if not d.startswith(".") and d not in ("results", "notes", "paper")]
        for name in filenames:
            p = Path(dirpath) / name
            if p.suffix in (".py", ".sh", ".pbs", ".txt") and p.resolve() not in ALLOWED:
                for i, line in enumerate(p.read_text(errors="replace").splitlines(), 1):
                    if OLD.search(line):
                        offenders.append(f"{p.relative_to(ROOT)}:{i}")
    assert not offenders, f"use the homes (sweeps/homes.py, pbs/lib/homes.sh) instead of results/raw|processed: {offenders}"
