"""C27-C: weight the consensus spectrum more heavily in training. notes/C27_consensus_weight_card.md.

    python sweeps/make_c27.py            # write configs/sweep-c27
    python sweeps/make_c27.py --check

--consensus_weight w (with --include_consensus): GroupBatchSampler draws a group's K members
WITHOUT replacement, the consensus with weight w and each experimental spectrum with weight 1.
P(the K = 2 pair contains the consensus): w 1 -> 0.50, w 3 -> 0.80, w inf -> 1.00.

    cons_w1       include_consensus, w 1   (C20 at the new recipe: uniform over the 4)
    cons_w3       include_consensus, w 3   ("as if it were there 3 times")
    cons_always   include_consensus, w inf (every pair = consensus + one experimental)
    cons_w3_kl0   as cons_w3, KL weight 0

3 seeds each -> 12 arms, one capacity node (1 tile per arm). Everything else is the K53 50m arm
sweep-hp50b s050m_ck540k_lr4e-4_p128k2_seed{0,1,2} (lr 4e-4, P128xK2, final 50m checkpoint
540,423) -- the current K66-C recipe; the reference arm ref_exp is those K53 runs, not retrained.
"""

from __future__ import annotations

import argparse
import shutil
from itertools import product
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "configs" / "sweep-c27"
TEMPLATE = REPO / "configs" / "sweep-hp50b" / "s050m_ck540k_lr4e-4_p128k2_seed{}" / "training.args"
RUN_PREFIX = "v2_c27-"
PRETRAINED = "/flare/UIC-HPC/khuss/msdelta/pretrained/msdelta-50m-production-01-checkpoint-540423"
TRAIN_GROUPS = 90288
SEEDS = ("0", "1", "2")
ARMS = {
    "cons_w1": {"--include_consensus": "true", "--consensus_weight": "1"},
    "cons_w3": {"--include_consensus": "true", "--consensus_weight": "3"},
    "cons_always": {"--include_consensus": "true", "--consensus_weight": "inf"},
    "cons_w3_kl0": {"--include_consensus": "true", "--consensus_weight": "3", "--kl_weight": "0"},
}


def parse(path: Path) -> list[tuple[str, str]]:
    tokens = path.read_text().split()
    return list(zip(tokens[::2], tokens[1::2]))


def arms() -> dict[str, tuple[str, str]]:
    out = {}
    for (arm, over), seed in product(ARMS.items(), SEEDS):
        name = f"s050m_ck540k_{arm}_seed{seed}"
        o = {"--pretrained_path": PRETRAINED,
             "--save_steps": str(TRAIN_GROUPS // int(over.get("--groups_per_batch", "128")) // 2),
             "--run_name": RUN_PREFIX + name, "--output_dir": f"./runs/{RUN_PREFIX}{name}", **over}
        pairs = [(k, o.pop(k) if k in o else v) for k, v in parse(Path(str(TEMPLATE).format(seed)))]
        pairs += list(o.items())
        args = "\n".join(f"{k} {v}" for k, v in pairs) + "\n"
        desc = (f"C27 ARM (C27-C, notes/C27_consensus_weight_card.md): {arm} "
                f"({' '.join(f'{k} {v}' for k, v in over.items())}), seed {seed}. Otherwise "
                f"sweep-hp50b s050m_ck540k_lr4e-4_p128k2_seed{seed} (K53 50m, lr 4e-4, P128xK2, "
                f"checkpoint 540,423). sweeps/make_c27.py.\n")
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
