"""C21: two mass regions per batch (the user's design, K22). PLAN.md C21.

    python sweeps/make_regions.py            # write configs/sweep-regions
    python sweeps/make_regions.py --check

THE QUESTION. sweep-mix's within75 adds 21 RANDOM groups to a 64-group same-mass block: the
64 anchors see 75% near-mass negatives, but the 21 random ones see almost none. Here the 21
(or 42) are a same-mass block of their OWN from a second, unrelated mass region, so every
anchor has near-mass negatives (its own block) and far-mass ones (the other block):
majority anchors 63 near / 21 far, minority 20 / 64. Sampler only; every negative used.

    regions75  64 + 21 groups (majority 75/25, minority 24/76)
    regions50  43 + 42 groups (every anchor ~50/50: the symmetric case)

3 seeds each (6 arms, one node). Everything else is the sweep-c8c19 supcon_mass arm, so
c8c19 (0% / random) and sweep-mix (within75, within50, between75) are the comparisons.
"""

from __future__ import annotations

import argparse
import shutil
from itertools import product
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "configs" / "sweep-regions"
TEMPLATE = REPO / "configs" / "sweep-c8c19" / "s050m_ck540k_supcon_mass_seed{}" / "training.args"
RUN_PREFIX = "v2_regions-"
SEEDS = ("0", "1", "2")
MIXES = {"regions75": ("0.25", "regions"), "regions50": ("0.5", "regions")}


def parse(path: Path) -> list[tuple[str, str]]:
    tokens = path.read_text().split()
    return list(zip(tokens[::2], tokens[1::2]))


def arms() -> dict[str, tuple[str, str]]:
    out = {}
    for (mix, (frac, mode)), seed in product(MIXES.items(), SEEDS):
        name = f"s050m_ck540k_supcon_{mix}_seed{seed}"
        over = {"--random_group_fraction": frac, "--random_mix": mode,
                "--run_name": RUN_PREFIX + name, "--output_dir": f"./runs/{RUN_PREFIX}{name}"}
        pairs = [(k, over.pop(k) if k in over else v) for k, v in parse(Path(str(TEMPLATE).format(seed)))]
        pairs += list(over.items())
        args = "\n".join(f"{k} {v}" for k, v in pairs) + "\n"
        desc = (f"REGIONS ARM: every batch = two same-mass blocks from two mass regions, the second holding {float(frac):.0%} of the groups, "
                f"seed {seed}. Otherwise sweep-c8c19 s050m_ck540k_supcon_mass_seed{seed}. "
                f"sweeps/make_regions.py.\n")
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
