"""C9: does an MLP projection head help the contrastive embedding? PLAN.md C9.

    python sweeps/make_conhead.py
    python sweeps/make_conhead.py --check

Master's MSDeltaForRetrieval pools and then projects through an MLP (2H -> 512 -> 256)
before the loss; ours has no head -- the loss and retrieval both use the normalised
pooled vector. SimCLR/SupCon also train through a head but retrieve on the PRE-head
features. So the head can matter in two ways, and one run answers both:

  head output   (master's readout)   -> retrieval/..., all/...        in the evals
  pre-head      (SimCLR's readout)   -> retrieval_pooled/..., pooled_all/...

Everything else is the frozen C1 recipe, rendered by make_confreeze.render_arm, so the
no-head control is sweep-confreeze's s050m_ck220k_seed{0,1,2} (C2) exactly: same
checkpoint, seeds, data, and code path apart from --projection_dim.

--quick writes sweep-conhead-quick: the same arms at 3 epochs (~15 min each at 50m, fits
a 1 h debug-scaling slot) for an early read. Its control is sweep-conlong's
s050m_t0002_pk256_ep03_seed{0,1,2}, identical apart from the head and checkpointing flags.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "sweeps"))
from make_confreeze import render_arm as confreeze_arm  # noqa: E402

OUT = REPO / "configs" / "sweep-conhead"
HEAD = {"--projection_dim": "256", "--projection_hidden": "512",
        "--projection_dropout": "0.1"}
CELLS = [("50m", "220000")]
SEEDS = ("0", "1", "2")
RUN_PREFIX = "v2_conhead-"


def render(scale, ck, seed, epochs=None):
    base_name, text = confreeze_arm(scale, ck, seed)
    name = f"{base_name}_head256" + (f"_ep{int(epochs):02d}" if epochs else "")
    lines = []
    for line in text.splitlines():
        flag = line.split(" ", 1)[0]
        if flag == "--num_train_epochs" and epochs:
            line = f"--num_train_epochs {epochs}"
        elif flag == "--run_name":
            line = f"--run_name {RUN_PREFIX}{name}"
        elif flag == "--output_dir":
            line = f"--output_dir ./runs/{RUN_PREFIX}{name}"
        lines.append(line)
    lines += [f"{f} {v}" for f, v in HEAD.items()]
    return name, "\n".join(lines) + "\n"


def description(scale, ck, seed):
    return (f"C9 PROJECTION-HEAD ARM: {scale} at checkpoint {ck}, seed {seed}. The frozen "
            f"C1 recipe plus master's head shape (pooled -> 512 -> 256, dropout 0.1); the "
            f"control is sweep-confreeze s{scale:0>4}_ck{int(ck)//1000:03d}k_seed{seed}. "
            f"Scores both the head output and the pre-head features.\n")


def combos(quick=False):
    return [(s, c, seed) + (("3",) if quick else ()) for s, c in CELLS for seed in SEEDS]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--quick", action="store_true")
    cli = ap.parse_args()
    out = OUT.with_name(OUT.name + "-quick") if cli.quick else OUT
    todo = combos(cli.quick)
    if cli.check:
        stale = [n for n, t in (render(*c) for c in todo)
                 if not (out / n / "training.args").exists()
                 or (out / n / "training.args").read_text() != t
                 or not (out / n / "DESCRIPTION.md").exists()]
        if stale:
            print(f"  {len(stale)} stale or missing: {', '.join(stale[:6])}")
            return 1
        print(f"  {len(todo)} arms match the confreeze recipe + head")
        return 0
    if out.exists():
        shutil.rmtree(out)
    for c in todo:
        name, text = render(*c)
        (out / name).mkdir(parents=True, exist_ok=True)
        (out / name / "training.args").write_text(text)
        (out / name / "DESCRIPTION.md").write_text(description(*c[:3]))
    (out / ".template").write_text("configs/finetune-contrastive-50m/training.args\ncontrastive-hp\n")
    print(f"arms={len(todo)} under {out.relative_to(REPO)}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
