"""A3: does an order-aware student readout fix the adjacent-swap blindness? PLAN.md A3.

    python sweeps/make_a3.py
    python sweeps/make_a3.py --check

The A1 student pools its residue tokens with mean+max, which is nearly order-blind: in
reranking it ranks the truth above an adjacent-residue swap only 70.5% of the time (vs
98-99% for other decoys). Its tokens DO carry order (positions are embedded and the
transformer mixes them); the pooling discards it. This grid changes only the readout:

    pool   mean+max over the tokens (A1, the control; re-run here on the same launcher)
    cls    a learned token prepended to the sequence; its output is the embedding
    attn   a learned query attending over the tokens

Everything else is configs/a1-align-100k-050m-c7s600 (same teacher, same cache via
--target_cache, same recipe), 3 seeds each, run by pbs/aurora-finetune-sweep.pbs one arm
per tile. Scored with pbs/eval_align_test.pbs and pbs/rescoring.pbs.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
BASE = REPO / "configs" / "a1-align-100k-050m-c7s600" / "training.args"
OUT = REPO / "configs" / "sweep-a3"
CACHE = "/lus/flare/projects/UIC-HPC/khuss/msdelta/align-targets-a1-050m-c7s600"
READOUTS = ("pool", "cls", "attn")
SEEDS = ("0", "1", "2")
RUN_PREFIX = "v2_a3-"


def render(readout, seed):
    name = f"{readout}_seed{seed}"
    over = {"--sequence_readout": readout, "--seed": seed, "--target_cache": CACHE,
            "--run_name": f"{RUN_PREFIX}{name}", "--output_dir": f"./runs/{RUN_PREFIX}{name}"}
    tokens = BASE.read_text().split()
    lines, seen = [], set()
    for f, v in zip(tokens[::2], tokens[1::2]):
        lines.append(f"{f} {over.get(f, v)}"); seen.add(f)
    lines += [f"{f} {v}" for f, v in over.items() if f not in seen]
    return name, "\n".join(lines) + "\n"


def description(readout, seed):
    return (f"A3 STUDENT READOUT ARM: readout={readout}, seed {seed}. Identical to "
            f"configs/a1-align-100k-050m-c7s600 (teacher C7-50m step 600, cache "
            f"{CACHE}) apart from the PeptideEncoder readout. PLAN.md A3.\n")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true")
    cli = ap.parse_args()
    combos = [(r, s) for r in READOUTS for s in SEEDS]
    if cli.check:
        stale = [n for n, t in (render(*c) for c in combos)
                 if not (OUT / n / "training.args").exists()
                 or (OUT / n / "training.args").read_text() != t
                 or not (OUT / n / "DESCRIPTION.md").exists()]
        if stale:
            print(f"  {len(stale)} stale or missing: {', '.join(stale[:6])}")
            return 1
        print(f"  {len(combos)} arms match {BASE.relative_to(REPO)}")
        return 0
    if OUT.exists():
        shutil.rmtree(OUT)
    for c in combos:
        name, text = render(*c)
        (OUT / name).mkdir(parents=True, exist_ok=True)
        (OUT / name / "training.args").write_text(text)
        (OUT / name / "DESCRIPTION.md").write_text(description(*c))
    (OUT / ".template").write_text(f"{BASE.relative_to(REPO)}\nalign\n")
    print(f"arms={len(combos)} under {OUT.relative_to(REPO)}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
