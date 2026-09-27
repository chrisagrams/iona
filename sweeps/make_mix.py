"""C19 ablation: mixed same-mass / random batches. PLAN.md C19 (mixed batches).

    python sweeps/make_mix.py            # write configs/sweep-mix
    python sweeps/make_mix.py --check

THE QUESTION. Same-mass batches (C19) beat random ones (+0.009 validation, +0.080 OOD MAP@R,
50M, 3 seeds; job 8872141). The filter-width analysis (job 8872964) showed the gain is largest
without a precursor filter and that 98-99% of open-retrieval top-1 errors are FAR in mass, so
C19 improves the embedding globally, not just within a window. Does mixing global negatives
back in (25% random) keep or add to that?

    within75   every batch = 64 consecutive same-mass groups + 21 random groups
    within50   every batch = 42 same-mass + 43 random (dose response)
    between75  75% of batches pure same-mass, 25% pure random

3 seeds each (9 arms, one node). Everything else is the sweep-c8c19 supcon_mass arm exactly
(50M@540k, ms-contrastive-100k only, P85 x K3, t 0.002, KL 10, lr 1e-4, 3 epochs, encoder
every half epoch, trimmed GradCache), so sweep-c8c19 supcon_mass (0% random) and
supcon_random (100%) are the two ends and are not re-run.
"""

from __future__ import annotations

import argparse
import shutil
from itertools import product
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "configs" / "sweep-mix"
TEMPLATE = REPO / "configs" / "sweep-c8c19" / "s050m_ck540k_supcon_mass_seed{}" / "training.args"
RUN_PREFIX = "v2_mix-"
SEEDS = ("0", "1", "2")
MIXES = {"within75": ("0.25", "within"), "within50": ("0.5", "within"),
         "between75": ("0.25", "between")}


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
        desc = (f"MIX ARM: same-mass batches with {float(frac):.0%} of groups random ({mode} batches), "
                f"seed {seed}. Otherwise sweep-c8c19 s050m_ck540k_supcon_mass_seed{seed}. "
                f"sweeps/make_mix.py.\n")
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
