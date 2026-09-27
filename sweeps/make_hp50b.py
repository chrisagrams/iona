"""C23 follow-up at 50m (K53): extend the learning rate past the edge and combine the winners.

    python sweeps/make_hp50b.py            # write configs/sweep-hp50b
    python sweeps/make_hp50b.py --check

sweep-hp-single at 50m@540k (job 8872806, scored 8873673 / 8873676): lr 4e-4 won on both
validation (+0.012 exp MAP@R) and OOD (+0.018) but was the HIGHEST lr tried, so the optimum is
unlocated; P128xK2 (+0.015 OOD) and KL 1 (+0.010 OOD) were the other promising single factors.

    lr8e-4          lr 8e-4
    lr1.6e-3        lr 1.6e-3
    lr4e-4_p128k2   lr 4e-4 + 128 groups x 2 spectra
    lr4e-4_kl1      lr 4e-4 + KL weight 1

3 seeds each (12 arms = one node). Everything else is the sweep-c8c19 supcon_mass arm
(SupCon + same-mass batches, t 0.002, 3 epochs, encoder every half epoch of the arm's own
schedule, trimmed GradCache); its 3 seeds are the lr 1e-4 reference.
"""

from __future__ import annotations

import argparse
import shutil
from itertools import product
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "configs" / "sweep-hp50b"
TEMPLATE = REPO / "configs" / "sweep-c8c19" / "s050m_ck540k_supcon_mass_seed{}" / "training.args"
RUN_PREFIX = "v2_hp50b-"
TRAIN_GROUPS = 90288
SEEDS = ("0", "1", "2")
ARMS = {
    "lr8e-4": {"--learning_rate": "8e-4"},
    "lr1.6e-3": {"--learning_rate": "1.6e-3"},
    "lr4e-4_p128k2": {"--learning_rate": "4e-4", "--groups_per_batch": "128", "--replicates": "2"},
    "lr4e-4_kl1": {"--learning_rate": "4e-4", "--kl_weight": "1"},
}


def parse(path: Path) -> list[tuple[str, str]]:
    tokens = path.read_text().split()
    return list(zip(tokens[::2], tokens[1::2]))


def arms() -> dict[str, tuple[str, str]]:
    out = {}
    for (arm, over), seed in product(ARMS.items(), SEEDS):
        name = f"s050m_ck540k_{arm}_seed{seed}"
        o = {"--save_steps": str(TRAIN_GROUPS // int(over.get("--groups_per_batch", "85")) // 2),
             "--run_name": RUN_PREFIX + name, "--output_dir": f"./runs/{RUN_PREFIX}{name}", **over}
        pairs = [(k, o.pop(k) if k in o else v) for k, v in parse(Path(str(TEMPLATE).format(seed)))]
        pairs += list(o.items())
        args = "\n".join(f"{k} {v}" for k, v in pairs) + "\n"
        desc = (f"HP50B ARM (C23/K53): {arm} ({' '.join(f'{k} {v}' for k, v in over.items())}), "
                f"seed {seed}. Otherwise sweep-c8c19 s050m_ck540k_supcon_mass_seed{seed}. "
                f"sweeps/make_hp50b.py.\n")
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
