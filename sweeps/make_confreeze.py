"""C2 and C4 at the frozen C1 recipe. PLAN.md C2 (scale) and C4 (pretraining checkpoint).

    python sweeps/make_confreeze.py --clean
    python sweeps/make_confreeze.py --check

THE RECIPE is C1's best cell, copied rather than re-chosen, so every arm here differs from
sweep-conlong's s050m_t0002_pk256_ep24 (MAP@R 0.877 / 0.880 / 0.874) ONLY in the
pretrained checkpoint:

    lr 1e-4, KL 10, temperature 0.002, width 256 (P 64 x K 4), 24 epochs,
    GradCache chunk 4, split_seed 0, final model (eval_strategy no), seeds 0-2.

Every arm is rendered from ONE dict (RECIPE) so the scales and checkpoints cannot drift
apart the way the older C2/C4 runs did (t0.07 vs 0.002, lr 1e-4 vs 2e-5, 3 vs 24 epochs).

C2: 50m / 100m / 200m / 400m at checkpoint 220,000. 50m is re-run rather than borrowed
from sweep-conlong so all C2 and C4 numbers come from one generator and one code version;
it also re-measures seed noise against the conlong arms.

C4: 50m and 100m across the rest of the canonical ladder (10k, 120k, 330k, 430k, 540,423);
their 220k point is the C2 arm.

Measured cost, one tile per arm (conlong 8857593, conscale 8856643): 50m 1.95 h, 100m
~2.8 h, 200m ~4.5 h (interpolated), 400m ~6.9 h at 24 epochs.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "configs" / "sweep-confreeze"
STAMP = OUT / ".template"
TEMPLATE = REPO / "configs" / "finetune-contrastive-50m" / "training.args"
RUN_PREFIX = "v2_confreeze-"
FROZEN = "/flare/UIC-HPC/khuss/msdelta/pretrained"

RECIPE = {
    "--learning_rate": "1e-4",
    "--kl_weight": "10",
    "--temperature": "0.002",
    "--groups_per_batch": "64",
    "--replicates": "4",
    "--num_train_epochs": "24",
    "--gradcache_chunk": "4",
    # Same checkpointing as sweep-conlong (see make_conlong.py): no intermediate saves,
    # so a killed arm restarts from step 0 with the same seed; final/ is written by main().
    "--save_only_model": "true",
    "--save_strategy": "no",
    "--save_steps": "700",
    "--save_total_limit": "1",
    "--split_seed": "0",
}
SEEDS = ("0", "1", "2")
C2 = [(s, "220000") for s in ("50m", "100m", "200m", "400m")]
C4 = [(s, c) for s in ("50m", "100m") for c in ("10000", "120000", "330000", "430000", "540423")]
# C4b (user 2026-09-25): 200m and 400m across the same ladder, ONE seed (their 220k point is C2).
C4B = [(s, c) for s in ("200m", "400m") for c in ("10000", "120000", "330000", "430000", "540423")]
SEEDS_C4B = ("0",)


def checkpoint(scale: str, ck: str) -> str:
    return f"{FROZEN}/msdelta-{scale}-production-01-checkpoint-{ck}"


def arm_name(scale: str, ck: str, seed: str) -> str:
    return f"s{scale:0>4}_ck{int(ck) // 1000:03d}k_seed{seed}"


def render_arm(scale: str, ck: str, seed: str) -> tuple[str, str]:
    name = arm_name(scale, ck, seed)
    overrides = dict(RECIPE)
    overrides.update({
        "--pretrained_path": checkpoint(scale, ck),
        "--seed": seed,
        "--run_name": f"{RUN_PREFIX}{name}",
        "--output_dir": f"./runs/{RUN_PREFIX}{name}",
    })
    lines, seen = [], set()
    tokens = TEMPLATE.read_text().split()
    for flag, value in zip(tokens[::2], tokens[1::2]):
        lines.append(f"{flag} {overrides.get(flag, value)}")
        seen.add(flag)
    for flag, value in overrides.items():
        if flag not in seen:
            lines.append(f"{flag} {value}")
    return name, "\n".join(lines) + "\n"


def description(scale: str, ck: str, seed: str) -> str:
    question = "C2 (scale) and C4 (checkpoint)" if ck == "220000" and scale in ("50m", "100m") \
        else "C2 (scale)" if ck == "220000" else "C4 (checkpoint)"
    n = len(SEEDS_C4B) if (scale, ck) in C4B else len(SEEDS)
    return (f"CONTRASTIVE FROZEN-RECIPE ARM for {question}: {scale} at pretraining "
            f"checkpoint {ck}, seed {seed} of {n}. Recipe is C1's best cell "
            f"(lr 1e-4, KL 10, t 0.002, P64xK4, 24 epochs, GradCache 4), identical across "
            f"every arm of this grid; only the pretrained checkpoint varies.\n")


def combos():
    return ([(s, c, seed) for s, c in C2 + C4 for seed in SEEDS]
            + [(s, c, seed) for s, c in C4B for seed in SEEDS_C4B])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--clean", action="store_true")
    cli = ap.parse_args()
    for s, c in C2 + C4 + C4B:
        if not Path(checkpoint(s, c), "model.safetensors").exists():
            raise SystemExit(f"{s}@{c} not frozen at {checkpoint(s, c)}")

    if cli.check:
        stale = [n for n, text in (render_arm(*c) for c in combos())
                 if not (OUT / n / "training.args").exists()
                 or (OUT / n / "training.args").read_text() != text
                 or not (OUT / n / "DESCRIPTION.md").exists()]
        if stale:
            print(f"  {len(stale)} stale or missing: {', '.join(stale[:6])}")
            return 1
        print(f"  {len(combos())} arms match {TEMPLATE.relative_to(REPO)}")
        return 0

    if cli.clean and OUT.exists():
        shutil.rmtree(OUT)
    for combo in combos():
        name, text = render_arm(*combo)
        (OUT / name).mkdir(parents=True, exist_ok=True)
        (OUT / name / "training.args").write_text(text)
        (OUT / name / "DESCRIPTION.md").write_text(description(*combo))
    STAMP.write_text(f"{TEMPLATE.relative_to(REPO)}\ncontrastive-hp\n")
    arms = REPO / "sweeps" / "arms"
    c2 = [arm_name(*c) for c in combos() if c[1] == "220000"]
    c4 = [arm_name(*c) for c in combos() if c[1] != "220000" and (c[0], c[1]) not in C4B]
    c4b = [arm_name(*c) for c in combos() if (c[0], c[1]) in C4B]
    (arms / "confreeze_c4b.txt").write_text("\n".join(c4b) + "\n")
    (arms / "confreeze_c2.txt").write_text("\n".join(c2) + "\n")
    (arms / "confreeze_c4.txt").write_text("\n".join(c4) + "\n")
    print(f"arms={len(combos())} under {OUT.relative_to(REPO)}/  (C2 {len(c2)}, C4 {len(c4)}, C4b {len(c4b)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
