"""K66-C (approved by the user 2026-09-27): per-scale search for the single-dataset contrastive recipe. PLAN.md C23.

    python sweeps/make_hp_scale.py            # write configs/sweep-hp-scale
    python sweeps/make_hp_scale.py --check

APPROVED CARD (notes/DECISIONS.md, K66-C):
  Question: does the new recipe (lr 4e-4 + P128xK2, K68-C) hold at every model size, or does the
            best learning rate / batch shape move with scale?
  Fixed:    SupCon, same-mass batches, t 0.002, KL 10, ms-contrastive-100k only, 3 epochs,
            trimmed GradCache (chunk 4), no gradient checkpointing, final pretraining checkpoint
            (540,423) of each scale; encoder saved every half epoch.
  Varied:   lr 2e-4 / 4e-4 / 8e-4 with 128 groups x 2 spectra; lr 4e-4 with 170 groups x 2.
  Scales:   25m, 100m, 200m, 400m (50m done in K53); 3 seeds each -> 48 arms, one 12-arm node per scale.
  Scored:   finals on validation, 8-species OOD, mouse, human; with/without filtering, filter
            passes/failures; selection on validation, OOD second check.
  Validate: debug smoke, 20 steps of 25m and 400m. Walltime 14 h for 400m (est. ~11 h), 10 h others.
  After:    the winner per scale on every pretraining checkpoint (its own card).

K136-C (approved 2026-09-29): 50m was searched earlier (K53) without the lr 4e-4 + P170xK2 cell; EXTRA adds
that one cell at 50m (3 seeds), otherwise exactly as above.
"""

from __future__ import annotations

import argparse
import shutil
from itertools import product
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "configs" / "sweep-hp-scale"
TEMPLATE = REPO / "configs" / "sweep-c8c19" / "s050m_ck540k_supcon_mass_seed{}" / "training.args"
RUN_PREFIX = "v2_hp-scale-"
PRETRAINED = "/flare/UIC-HPC/khuss/msdelta/pretrained/msdelta-{}-production-01-checkpoint-540423"
SIZES = ("25m", "100m", "200m", "400m")
TRAIN_GROUPS = 90288
SEEDS = ("0", "1", "2")
ARMS = {
    "lr2e-4_p128k2": {"--learning_rate": "2e-4", "--groups_per_batch": "128", "--replicates": "2"},
    "lr4e-4_p128k2": {"--learning_rate": "4e-4", "--groups_per_batch": "128", "--replicates": "2"},
    "lr8e-4_p128k2": {"--learning_rate": "8e-4", "--groups_per_batch": "128", "--replicates": "2"},
    "lr4e-4_p170k2": {"--learning_rate": "4e-4", "--groups_per_batch": "170", "--replicates": "2"},
}
EXTRA = [("50m", "lr4e-4_p170k2")]  # K136-C

def parse(path: Path) -> list[tuple[str, str]]:
    tokens = path.read_text().split()
    return list(zip(tokens[::2], tokens[1::2]))


def arms() -> dict[str, tuple[str, str]]:
    out = {}
    cells = [(s, a, o) for s in SIZES for a, o in ARMS.items()] + [(s, a, ARMS[a]) for s, a in EXTRA]
    for (size, arm, over), seed in product(cells, SEEDS):
        name = f"s{size.rjust(4, '0')}_ck540k_{arm}_seed{seed}"
        o = {"--pretrained_path": PRETRAINED.format(size),
             "--save_steps": str(TRAIN_GROUPS // int(over.get("--groups_per_batch", "85")) // 2),
             "--run_name": RUN_PREFIX + name, "--output_dir": f"./runs/{RUN_PREFIX}{name}", **over}
        pairs = [(k, o.pop(k) if k in o else v) for k, v in parse(Path(str(TEMPLATE).format(seed)))]
        pairs += list(o.items())
        args = "\n".join(f"{k} {v}" for k, v in pairs) + "\n"
        desc = (f"HP-SCALE ARM (C23/K66-C, approved): {size}, {arm} ({' '.join(f'{k} {v}' for k, v in over.items())}), "
                f"seed {seed}. Otherwise sweep-c8c19 s050m_ck540k_supcon_mass_seed{seed}, from the {size} checkpoint 540,423. "
                f"sweeps/make_hp_scale.py.\n")
        out[name] = (args, desc)
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
    for name, (args, desc) in want.items():
        (OUT / name).mkdir(parents=True)
        (OUT / name / "training.args").write_text(args)
        (OUT / name / "DESCRIPTION.md").write_text(desc)
    print(f"wrote {len(want)} arms to {OUT.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
