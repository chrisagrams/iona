"""K155-C (user 2026-09-30): the contrastive recipe on every pretraining checkpoint of 50m, 100m, 200m, 400m.

    python sweeps/make_allck.py            # write configs/sweep-allck + sweeps/arms/allck_<scale>.txt
    python sweeps/make_allck.py --check

User: "Using the optimal HPs we have here (record this as a choice and argue if you have any qualms about
it) fine tune all 4 scales with contrastive as we described."
  Recipe:  the K66-C/K136-C winner on validation experimental MAP@R at every scale: lr 4e-4, 170 groups x 2
           spectra; otherwise exactly the K66-C arms (SupCon, same-mass batches, t 0.002, KL 10,
           ms-contrastive-100k only, NO consensus, 3 epochs, trimmed GradCache chunk 4).
  Checkpoints: every production checkpoint on /flare except 540,423, whose runs already exist (K66-C/K136-C
           finals, same recipe): 50m 7, 100m 6, 200m 6, 400m 7 -> 26 checkpoints x 3 seeds = 78 arms.
  Jobs:    one capacity job per scale, 2 nodes (12 arms per node).
  Scored:  as K66-C (validation, test, oodval, mouse, human, yeast; library search on).
Qualms and choices: notes/DECISIONS.md K155-C.
"""

from __future__ import annotations

import argparse
import shutil
from itertools import product
from pathlib import Path

import make_hp_scale as hp

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "configs" / "sweep-allck"
ARMS_DIR = REPO / "sweeps" / "arms"
RUN_PREFIX = "v2_allck-"
PRETRAINED = "/flare/UIC-HPC/khuss/msdelta/pretrained/msdelta-{}-production-01-checkpoint-{}"
CHECKPOINTS = {
    "50m": (10000, 50000, 120000, 133233, 220000, 330000, 430000),
    "100m": (10000, 120000, 138073, 220000, 330000, 430000),
    "200m": (10000, 120000, 192799, 220000, 330000, 430000),
    "400m": (10000, 120000, 181381, 220000, 255058, 330000, 430000),
}
ARM = "lr4e-4_p170k2"
SEEDS = ("0", "1", "2")


def arms() -> dict[str, tuple[str, str, str]]:
    out = {}
    over = hp.ARMS[ARM]
    for (size, cks), seed in product(CHECKPOINTS.items(), SEEDS):
        for ck in cks:
            name = f"s{size.rjust(4, '0')}_ck{ck // 1000:03d}k_{ARM}_seed{seed}"
            o = {"--pretrained_path": PRETRAINED.format(size, ck),
                 "--save_steps": str(hp.TRAIN_GROUPS // int(over["--groups_per_batch"]) // 2),
                 "--run_name": RUN_PREFIX + name, "--output_dir": f"./runs/{RUN_PREFIX}{name}", **over}
            pairs = [(k, o.pop(k) if k in o else v) for k, v in hp.parse(Path(str(hp.TEMPLATE).format(seed)))]
            pairs += list(o.items())
            args = "\n".join(f"{k} {v}" for k, v in pairs) + "\n"
            desc = (f"ALL-CHECKPOINT ARM (K155-C): {size} checkpoint {ck:,}, {ARM} "
                    f"({' '.join(f'{k} {v}' for k, v in over.items())}), seed {seed}. Otherwise the K66-C arm "
                    f"s{size.rjust(4, '0')}_ck540k_{ARM}_seed{seed}. sweeps/make_allck.py.\n")
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
    for size in CHECKPOINTS:
        names = sorted(n for n, v in want.items() if v[2] == size)
        (ARMS_DIR / f"allck_{size.rjust(4, '0')}.txt").write_text("\n".join(names) + "\n")
    print(f"wrote {len(want)} arms to {OUT.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
