"""Do more negatives help PK softmax? Re-asked at the hyperparameters that win.

    python sweeps/make_gradcache_v2_grid.py --clean

THE FIRST ANSWER WAS CONFOUNDED THREE WAYS. GradCache at P=16 K=4 -- 64 rows, so each
anchor sees 3 positives and 60 negatives instead of 1 and 2 -- scored 5.70 against PK's
5.86 at P=2 K=2, and that was read as "more negatives do not help". But the GradCache
arms ran at lr 5e-4 and temperature 0.2 where the PK arms ran 2e-5 and 0.07, they
predate the FT14 sampler fix, and they were n=1.

Now that the HP grid has settled the configuration (lr 2e-5 / KL 0 / temperature 0.2,
job 8847624, 4 seeds per cell), the question can be asked properly: same
hyperparameters, same sampler, same seeds, only the batch breadth differing.

WHY IT MATTERS. P=2 K=2 gives every anchor one positive and two negatives, which is the
thinnest contrastive signal the loss admits, and it is forced by memory: DeltaMZBias is
O(batch * peaks^2) and a softmax denominator cannot be split. GradCache splits the
FORWARD pass instead, embedding in chunks under no_grad, taking the loss gradient with
respect to the embeddings, then re-embedding with grad. If breadth is worth anything to
this objective, this is how PK gets it.

The pair loss offers the same breadth more cheaply, which is why both are being asked
at once -- see sweep-pairaccum.
"""

from __future__ import annotations

import argparse
import itertools
import shutil
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "configs" / "sweep-gradcache-v2"
STAMP = OUT / ".template"
TEMPLATE = REPO / "configs" / "finetune-contrastive-50m" / "training.args"
RUN_PREFIX = "v2_gcv2-"
CHECKPOINT = ("/flare/UIC-HPC/khuss/msdelta/pretrained/"
              "msdelta-50m-production-01-checkpoint-133233")

# (label, groups_per_batch, replicates, gradcache_chunk): breadth from 4 rows to 64.
SHAPES = {
    "p02k02": ("2", "2", "0"),     # the baseline: 1 positive, 2 negatives, no GradCache
    "p08k02": ("8", "2", "4"),     # 16 rows
    "p16k04": ("16", "4", "4"),    # 64 rows: 3 positives, 60 negatives
}
SEEDS = ("0", "1", "2", "3")
LEARNING_RATE, KL_WEIGHT, TEMPERATURE = "2e-5", "0", "0.2"
EPOCHS = "3"


def arm_name(shape, seed):
    return f"{shape}_seed{seed}"


def render_arm(shape, seed):
    p, k, chunk = SHAPES[shape]
    name = arm_name(shape, seed)
    overrides = {
        "--pretrained_path": CHECKPOINT,
        "--groups_per_batch": p,
        "--replicates": k,
        "--gradcache_chunk": chunk,
        "--learning_rate": LEARNING_RATE,
        "--kl_weight": KL_WEIGHT,
        "--temperature": TEMPERATURE,
        "--num_train_epochs": EPOCHS,
        "--seed": seed,
        "--split_seed": "0",
        "--run_name": f"{RUN_PREFIX}{name}",
        "--output_dir": f"./runs/{RUN_PREFIX}{name}",
    }
    lines, seen = [], set()
    tokens = TEMPLATE.read_text().split()
    for flag, value in zip(tokens[::2], tokens[1::2]):
        lines.append(f"{flag} {overrides.get(flag, value)}")
        seen.add(flag)
    for flag, value in overrides.items():
        if flag not in seen:
            lines.append(f"{flag} {value}")
    return name, "\n".join(lines) + "\n"


def description(shape, seed):
    p, k, chunk = SHAPES[shape]
    rows = int(p) * int(k)
    gc = "no GradCache" if chunk == "0" else f"GradCache chunk {chunk}"
    return (f"BREADTH ARM: P={p} x K={k} = {rows} rows, {gc}, so each anchor sees "
            f"{int(k)-1} positive(s) and {rows-int(k)} negative(s). Seed {seed}, at the "
            f"configuration the HP grid selected: lr {LEARNING_RATE}, KL {KL_WEIGHT}, "
            f"temperature {TEMPERATURE}, {EPOCHS} epochs. Re-asks whether more negatives "
            f"help, a question whose first answer (5.70 vs 5.86, 'no') was confounded by "
            f"a different learning rate, a different temperature, the pre-FT14 sampler, "
            f"and n=1.\n")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--clean", action="store_true")
    cli = ap.parse_args()
    if not Path(CHECKPOINT, "model.safetensors").exists():
        raise SystemExit("50m not frozen")
    combos = list(itertools.product(SHAPES, SEEDS))
    if cli.check:
        stale = [n for n, t in (render_arm(*c) for c in combos)
                 if not (OUT / n / "training.args").exists()
                 or (OUT / n / "training.args").read_text() != t
                 or not (OUT / n / "DESCRIPTION.md").exists()]
        if stale:
            print(f"  {len(stale)} stale or missing: {', '.join(stale[:6])}")
            return 1
        print(f"  {len(combos)} arms match {TEMPLATE.relative_to(REPO)}")
        return 0
    if cli.clean and OUT.exists():
        shutil.rmtree(OUT)
    for c in combos:
        name, text = render_arm(*c)
        (OUT / name).mkdir(parents=True, exist_ok=True)
        (OUT / name / "training.args").write_text(text)
        (OUT / name / "DESCRIPTION.md").write_text(description(*c))
    STAMP.write_text(f"{TEMPLATE.relative_to(REPO)}\ngradcache-v2\n")
    print(f"arms={len(combos)} under {OUT.relative_to(REPO)}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
