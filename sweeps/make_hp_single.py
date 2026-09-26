"""Hyperparameter search per model size for the single-dataset contrastive recipe (after C8 x C19).

    python sweeps/make_hp_single.py            # write configs/sweep-hp-single
    python sweeps/make_hp_single.py --check

Recipe fixed by C8 x C19 (sweep-c8c19): SupCon loss + same-mass batches (C19), trained from the
PRETRAINED encoder on ms-contrastive-100k alone, KL anchor to the pretrained intensity head,
length-trimmed GradCache (chunk 4), no gradient checkpointing. Every size (25M, 50M, 100M, 200M,
400M) starts from its FINAL pretraining checkpoint (540,423).

One factor at a time around the base setting, 1 seed each (12 arms per size = one node):

    base         lr 1e-4, temperature 0.002, 85 groups x 3 spectra, KL weight 10
    lr           5e-5, 2e-4, 4e-4
    temperature  0.001, 0.005, 0.01
    kl_weight    0, 1, 30
    batch        64 x 4, 128 x 2            (~255 spectra per step, as the base)

3 epochs, the encoder saved every half epoch, so each run also gives its epoch curve. Selected
on ms-contrastive-100k VALIDATION MAP@R (8-other-species OOD validation as the second check).
The winning setting per size then runs on every pretraining checkpoint (next sweep).
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "configs" / "sweep-hp-single"
TEMPLATE = REPO / "configs" / "sweep-c8c19" / "s050m_ck540k_supcon_mass_seed0" / "training.args"
PRETRAINED = "/flare/UIC-HPC/khuss/msdelta/pretrained/msdelta-{}-production-01-checkpoint-540423"
SIZES = ("25m", "50m", "100m", "200m", "400m")
RUN_PREFIX = "v2_hp-single-"
EPOCH_STEPS = 1062                       # at ~255 spectra per step
ARMS = {
    "base": {},
    "lr5e-5": {"--learning_rate": "5e-5"},
    "lr2e-4": {"--learning_rate": "2e-4"},
    "lr4e-4": {"--learning_rate": "4e-4"},
    "t0.001": {"--temperature": "0.001"},
    "t0.005": {"--temperature": "0.005"},
    "t0.01": {"--temperature": "0.01"},
    "kl0": {"--kl_weight": "0"},
    "kl1": {"--kl_weight": "1"},
    "kl30": {"--kl_weight": "30"},
    "p64k4": {"--groups_per_batch": "64", "--replicates": "4"},
    "p128k2": {"--groups_per_batch": "128", "--replicates": "2"},
}


def parse(path: Path) -> list[tuple[str, str]]:
    t = path.read_text().split()
    return list(zip(t[::2], t[1::2]))


def arms() -> dict[str, tuple[str, str]]:
    base = parse(TEMPLATE)
    out = {}
    for size in SIZES:
        for arm, over in ARMS.items():
            name = f"s{size.rjust(4, '0')}_ck540k_{arm}"
            o = {"--pretrained_path": PRETRAINED.format(size), "--seed": "0",
                 "--num_train_epochs": "3", "--save_steps": str(EPOCH_STEPS // 2),
                 "--run_name": RUN_PREFIX + name, "--output_dir": f"./runs/{RUN_PREFIX}{name}", **over}
            pairs = [(k, o.pop(k) if k in o else v) for k, v in base]
            pairs += list(o.items())
            args = "\n".join(f"{k} {v}" for k, v in pairs) + "\n"
            desc = (f"HP-SINGLE ARM: {size} from pretrained checkpoint 540,423, setting '{arm}' "
                    f"({' '.join(f'{k} {v}' for k, v in over.items()) or 'base'}); SupCon + same-mass "
                    f"batches on ms-contrastive-100k only, 3 epochs, encoder every half epoch. "
                    f"sweeps/make_hp_single.py.\n")
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
