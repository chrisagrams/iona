"""Worked example: TODAY's K66-C pipeline (approved card, notes/DECISIONS.md 2026-09-27)
expressed as a DAG spec. It mirrors the hand-written driver k66_pipeline.sh:

    smoke (debug, 25m + 400m arms, MAX_STEPS=20)
      -> train_400m / train_200m / train_100m / train_025m   (capacity, 1 node, 12 arms each;
                                                             walltime 14 h / 10 h per the card)
           -> score_<scale>_{validation, oodval, test, mouse, human}  (debug / debug-scaling)

It lives in examples/, NOT in pipelines/, on purpose: K66-C is already being run by that
driver, and a live tick over this spec would submit it a second time. Plan it with

    pbs/dagctl --pipelines-dir pbs/dag/examples --home <tmp> plan --backend fake

Runtimes are the card's walltimes divided by the 1.3 walltime factor, so the scheduler
reproduces the approved walltimes exactly (14:00:00 and 10:00:00).
"""

import glob
import os
from pathlib import Path

from dag.spec import Node, Pipeline, Prepare, Sweep

CARD = "K66-C"
SCALES = ["400m", "200m", "100m", "025m"]            # longest first, as in the driver
WALL_H = {"400m": 14, "200m": 10, "100m": 10, "025m": 10}
BASE = "/lus/flare/projects/UIC-HPC/khuss/msdelta/baselines"
# name -> extra vars (k66_datasets.txt; the driver pinned a queue per set, here the
# scheduler picks debug or debug-scaling, whichever has a slot)
SCORING = {
    "validation": {"SPLIT": "validation"},
    "oodval": {"DATA": f"{BASE}/nine_oodval20k/prepared"},
    "test": {"SPLIT": "test"},
    "mouse": {"DATA": f"{BASE}/noble_mouse20k/prepared"},
    "human": {"DATA": f"{BASE}/noble_human20k/prepared"},
}


def models_file(scale: str) -> Prepare:
    """Write sweeps/arms/score_hp_scale_<scale>.txt from the training job's run dirs."""
    rel = f"sweeps/arms/score_hp_scale_{scale}.txt"

    def run(ctx):
        job = ctx["template"](f"{{job:train_{scale}}}")
        runs = sorted(glob.glob(f"{ctx['scratch']}/runs/sweep-s{scale}_ck540k_*-{job}"))
        if not runs:
            raise RuntimeError(f"no run dirs for job {job}")
        lines = [f"# K66-C {scale} finals (job {job})"]
        for d in runs:
            name = os.path.basename(d)[len("sweep-"):].rsplit("-", 1)[0]
            lines.append(f"{name} {d}/final")
        path = Path(ctx["repo"]) / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(f".{path.name}.tmp")
        tmp.write_text("\n".join(lines) + "\n")
        os.replace(tmp, path)
        return [path]

    return Prepare(describe=f"write {rel} from {{scratch}}/runs/sweep-s{scale}_ck540k_*-"
                            f"<job of train_{scale}>", run=run)


nodes = [
    Node(id="smoke", sweep=Sweep("configs/sweep-hp-scale", "sweeps/arms/hp_scale_smoke.txt"),
         vars={"MAX_STEPS": "20"}, nodes=1, runtime_min=45, queues=["debug"],
         approved=CARD, max_resumes=0, note="debug smoke: 25m + 400m arms, 20 steps"),
]
for s in SCALES:
    nodes.append(Node(
        id=f"train_{s}", deps=["smoke"],
        sweep=Sweep("configs/sweep-hp-scale", f"sweeps/arms/hp_scale_{s}.txt"),
        nodes=1, runtime_min=WALL_H[s] * 60 / 1.3, queues=["capacity"],
        approved=CARD, max_resumes=2,
        outputs=[f"{{scratch}}/runs/sweep-s{s}_ck540k_*-{{job}}"],
        progress=[f"{{scratch}}/runs/sweep-s{s}_ck540k_*-{{job}}/logs/train.log"],
        stall_minutes=60))
    for name, extra in SCORING.items():
        nodes.append(Node(
            id=f"score_{s}_{name}", deps=[f"train_{s}"],
            script="pbs/eval_grouped_retrieval.pbs",
            vars=dict(MODELS=f"sweeps/arms/score_hp_scale_{s}.txt", **extra,
                      OUT_DIR=f"{{scratch}}/eval/contrastive/hp-scale-{s}-{name}"),   # $MSDELTA_EVAL (configs/homes.env)
            nodes=1, runtime_min=30, queues=["debug", "debug-scaling"], approved=CARD,
            rerun_on=("walltime", "killed"), max_retries=1,
            outputs=[f"{{scratch}}/eval/contrastive/hp-scale-{s}-{name}"],
            prepare=models_file(s)))

PIPELINE = Pipeline(name="k66c", nodes=nodes, budget_node_hours=80, card=CARD,
                    description="K66-C per-scale HP search: smoke -> 4 sweeps -> scoring")
