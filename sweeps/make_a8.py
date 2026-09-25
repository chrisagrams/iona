"""A8: mass-aware student training. PLAN.md A8 (user-approved 2026-09-25).

    python sweeps/make_a8.py
    python sweeps/make_a8.py --check

Deployed search filters candidates by precursor mass, so the student's job is separating
peptides of (nearly) the SAME mass. Same student and data as the A-oodsel baseline
(configs/a1-align-100k-400m-oodsel: teacher C7 400m seed 1 step 600, chosen on the
8-other-species OOD validation; cache align-targets-400m-oodsel), LiT + 0.1 MSE, batch 256:

    massb        mass-bucketed batches: each batch = mass neighbours (+-0.5 Da jitter), so
                 in-batch negatives are near-same-mass competitors
    massb_hn4    + 4 hard negatives per row: other TRAINING peptides within +-20 ppm
                 (I/L-equivalents excluded)

3 seeds each. Baseline = the A-oodsel MSE students. Selection on validation; evaluation on
the test split and nine-species yeast (open / +-1.1 Da / 20 ppm) vs yHydra, then R.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
BASE = REPO / "configs" / "a1-align-100k-400m-oodsel" / "training.args"
OUT = REPO / "configs" / "sweep-a8"
CACHE = "/lus/flare/projects/UIC-HPC/khuss/msdelta/align-targets-400m-oodsel"
ARMS = {
    "massb": {"--mass_batches": "true", "--hard_negatives": "0"},
    "massb_hn4": {"--mass_batches": "true", "--hard_negatives": "4", "--neg_source": "mass",
                  "--neg_ppm": "20"},
}
SEEDS = ("0", "1", "2")
RUN_PREFIX = "v2_a8-"


def render(arm, seed):
    name = f"{arm}_seed{seed}"
    over = {"--align_loss": "lit", "--align_temperature": "0.05", "--mse_weight": "0.1",
            "--per_device_train_batch_size": "256", "--seed": seed, "--target_cache": CACHE,
            "--run_name": f"{RUN_PREFIX}{name}", "--output_dir": f"./runs/{RUN_PREFIX}{name}",
            **ARMS[arm]}
    tokens = BASE.read_text().split()
    lines, seen = [], set()
    for f, v in zip(tokens[::2], tokens[1::2]):
        lines.append(f"{f} {over.get(f, v)}"); seen.add(f)
    lines += [f"{f} {v}" for f, v in over.items() if f not in seen]
    return name, "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true")
    cli = ap.parse_args()
    combos = [(a, s) for a in ARMS for s in SEEDS]
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
    for a, s in combos:
        name, text = render(a, s)
        (OUT / name).mkdir(parents=True)
        (OUT / name / "training.args").write_text(text)
        (OUT / name / "DESCRIPTION.md").write_text(
            f"A8 ARM {a}, seed {s}: mass-aware student ({ARMS[a]}), LiT + 0.1 MSE, batch 256, "
            f"teacher = A-oodsel's (cache {CACHE}). PLAN.md A8.\n")
    print(f"arms={len(combos)} under {OUT.relative_to(REPO)}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
