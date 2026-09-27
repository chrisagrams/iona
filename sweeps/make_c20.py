"""C20 (K54): train WITH the consensus spectrum of each group.

    python sweeps/make_c20.py            # write configs/sweep-c20
    python sweeps/make_c20.py --check

Each ms-contrastive-100k group has 3 experimental spectra and 1 consensus spectrum built from
them; every run so far trained on the 3 experimental ones only. The hp-single sweep showed the
KL anchor strongly shapes retrieval WITH consensus spectra in the gallery (all-spectra MAP@R
0.85 at KL 0 vs 0.66 at KL 30), so consensus handling and KL may interact (K54).

    cons_k3       consensus included, 85 groups x 3 spectra sampled from the 4
    cons_k4       consensus included, 64 groups x 4 spectra (every member, ~same batch size)
    cons_k3_kl0   as cons_k3, KL weight 0

3 seeds each (9 arms, one node). Everything else is the sweep-c8c19 supcon_mass arm (lr 1e-4,
kept at the old value so the comparison with its 3 seeds -- the experimental-only reference --
is clean); scored on experimental MAP@R (headline) AND with consensus in the gallery.
"""

from __future__ import annotations

import argparse
import shutil
from itertools import product
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "configs" / "sweep-c20"
TEMPLATE = REPO / "configs" / "sweep-c8c19" / "s050m_ck540k_supcon_mass_seed{}" / "training.args"
RUN_PREFIX = "v2_c20-"
TRAIN_GROUPS = 90288
SEEDS = ("0", "1", "2")
ARMS = {
    "cons_k3": {"--include_consensus": "true"},
    "cons_k4": {"--include_consensus": "true", "--groups_per_batch": "64", "--replicates": "4"},
    "cons_k3_kl0": {"--include_consensus": "true", "--kl_weight": "0"},
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
        desc = (f"C20 ARM (K54): {arm} ({' '.join(f'{k} {v}' for k, v in over.items())}), "
                f"seed {seed}. Otherwise sweep-c8c19 s050m_ck540k_supcon_mass_seed{seed}. "
                f"sweeps/make_c20.py.\n")
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
