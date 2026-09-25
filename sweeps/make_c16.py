"""C16: the C7 recipe from the FINAL 200m pretraining checkpoint (540,423), one seed, then
downstream (A student, R reranking). PLAN.md C16 (user, 2026-09-25).

    python sweeps/make_c16.py --stage 1            # configs/sweep-c16-stage1
    python sweeps/make_c16.py --stage 2 --from RUN # configs/sweep-c16-stage2 (after stage 1)
    python sweeps/make_c16.py --stage 1 --check

Stage 1 = configs/sweep-conlong/s400m_t0002_pk256_ep12_seed0 (the C1 recipe as run at 400m:
t 0.002, KL 10, P64 x K4, lr 1e-4, 12 epochs, replicate corpus) with only the starting
checkpoint (200m @ 540,423) and names changed. Stage 2 = configs/sweep-con100k-best/
cont400m_ep01_seed0 (one epoch of ms-contrastive-100k, encoder every 300 steps) with only
--pretrained_path (the stage-1 run's final/) and names changed.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
BASE200 = "/flare/UIC-HPC/khuss/msdelta/pretrained/msdelta-200m-production-01-checkpoint-540423"
STAGES = {
    1: (REPO / "configs/sweep-conlong/s400m_t0002_pk256_ep12_seed0/training.args",
        REPO / "configs/sweep-c16-stage1", "s200m540k_t0002_pk256_ep12_seed0", "v2_c16-"),
    2: (REPO / "configs/sweep-con100k-best/cont400m_ep01_seed0/training.args",
        REPO / "configs/sweep-c16-stage2", "cont200m540k_ep01_seed0", "v2_c16-"),
}


def render(stage: int, start: str) -> tuple[str, str]:
    src, _, name, prefix = STAGES[stage]
    over = {"--pretrained_path": start, "--run_name": f"{prefix}{name}",
            "--output_dir": f"./runs/{prefix}{name}"}
    tokens = src.read_text().split()
    assert "--pretrained_path" in tokens[::2]
    lines = [f"{f} {over.get(f, v)}" for f, v in zip(tokens[::2], tokens[1::2])]
    return name, "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stage", type=int, required=True, choices=(1, 2))
    ap.add_argument("--from", dest="start", default="", help="stage 2: the stage-1 final/ dir")
    ap.add_argument("--check", action="store_true")
    cli = ap.parse_args()
    if cli.stage == 2 and not cli.start and not cli.check:
        ap.error("--stage 2 needs --from STAGE1_RUN/final")
    out = STAGES[cli.stage][1]
    if cli.check:
        arms = [p for p in out.iterdir() if p.is_dir()] if out.exists() else []
        if len(arms) != 1:
            print(f"  {out}: {len(arms)} arms"); return 1
        text = (arms[0] / "training.args").read_text()
        start = text.split("--pretrained_path ", 1)[1].split()[0]
        name, want = render(cli.stage, start)
        ok = arms[0].name == name and text == want and (arms[0] / "DESCRIPTION.md").exists()
        print(f"  {out.name}: {'1 arm matches' if ok else 'STALE'}")
        return 0 if ok else 1
    start = BASE200 if cli.stage == 1 else cli.start
    name, text = render(cli.stage, start)
    if out.exists():
        shutil.rmtree(out)
    (out / name).mkdir(parents=True)
    (out / name / "training.args").write_text(text)
    (out / name / "DESCRIPTION.md").write_text(
        f"C16 STAGE {cli.stage}: C7 recipe from the FINAL 200m checkpoint (540,423). "
        f"Start: {start}. Otherwise identical to {STAGES[cli.stage][0].parent.name} "
        f"({STAGES[cli.stage][0].parent.parent.name}). PLAN.md C16.\n")
    print(f"wrote {out.relative_to(REPO)}/{name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
