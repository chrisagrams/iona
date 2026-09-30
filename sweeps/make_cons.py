"""K160-C (user 2026-09-30): the with-consensus twin of every contrastive model we already have at the chosen recipe.

    python sweeps/make_cons.py            # write configs/sweep-cons + sweeps/arms/cons_all.txt
    python sweeps/make_cons.py --check

User: "We should pause K155 and start running training runs with consensus at the same checkpoints we
already have and the same HPs we already have, then compare them on what we have."
  Twins of:  the K66-C/K136-C finals (lr 4e-4, P170xK2, checkpoint 540,423) of 25m/50m/100m/200m/400m and the
             K155-C arms finished before the pause (100m checkpoints 220k/330k/430k), 3 seeds each -> 24 arms.
  Change:    ONLY --include_consensus false -> true, with --consensus_weight 1 (the flag's default: the
             consensus is one more group member sampled like the rest; = C27 "w1"). Every other line is copied
             from the source arm's training.args, so the pair differs in consensus alone.
  Scored:    as K66-C (validation, test, oodval, mouse, human, yeast; library search on), next to the source arms.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "configs" / "sweep-cons"
ARMS_DIR = REPO / "sweeps" / "arms"
RUN_PREFIX = "v2_cons-"
ARM = "lr4e-4_p170k2"
SEEDS = ("0", "1", "2")
# (source grid, scale, checkpoint tag) of the no-consensus arms that exist
SOURCES = [("sweep-hp-scale", s, "540k") for s in ("025m", "050m", "100m", "200m", "400m")] + \
          [("sweep-allck", "100m", ck) for ck in ("220k", "330k", "430k")]


def arms() -> dict[str, tuple[str, str, str]]:
    out = {}
    for grid, size, ck in SOURCES:
        for seed in SEEDS:
            src = f"s{size}_ck{ck}_{ARM}_seed{seed}"
            name = f"s{size}_ck{ck}_{ARM}_cons_seed{seed}"
            tokens = (REPO / "configs" / grid / src / "training.args").read_text().split()
            o = dict(zip(tokens[::2], tokens[1::2]))
            assert o.get("--include_consensus") == "false" and "--consensus_weight" not in o, src
            o["--include_consensus"] = "true"
            o["--consensus_weight"] = "1"
            o["--run_name"] = RUN_PREFIX + name
            o["--output_dir"] = f"./runs/{RUN_PREFIX}{name}"
            args = "\n".join(f"{k} {v}" for k, v in o.items()) + "\n"
            desc = (f"CONSENSUS TWIN (K160-C): configs/{grid}/{src} with --include_consensus true "
                    f"--consensus_weight 1; nothing else changed. sweeps/make_cons.py.\n")
            out[name] = (args, desc, size)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    cli = ap.parse_args()
    want = arms()
    if cli.check:
        have = {d.name for d in OUT.iterdir() if d.is_dir()} if OUT.exists() else set()
        bad = sorted(set(want) ^ have) + [n for n in want if n in have and
                                          (OUT / n / "training.args").read_text() != want[n][0]]
        print("stale or missing arms: " + str(bad) if bad else f"{len(want)} arms match")
        return 1 if bad else 0
    if OUT.exists():
        shutil.rmtree(OUT)
    for name, (args, desc, _) in want.items():
        (OUT / name).mkdir(parents=True)
        (OUT / name / "training.args").write_text(args)
        (OUT / name / "DESCRIPTION.md").write_text(desc)
    (ARMS_DIR / "cons_all.txt").write_text("\n".join(sorted(want)) + "\n")
    print(f"wrote {len(want)} arms to {OUT.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
