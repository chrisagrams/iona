"""Does the hyperparameter optimum move as pretraining progresses? Asked once.

    python sweeps/make_ckpt_hp_grid.py --task denoise --clean
    python sweeps/make_ckpt_hp_grid.py --task contrastive --clean

THE TRAP THIS AVOIDS. Both configurations in use were chosen at checkpoint 1, a quarter
of the way through pretraining. The naive way to check whether they still hold is to
sweep hyperparameters at every checkpoint and every scale, which is 6 rungs x 4 scales
x a grid -- a programme with no end, and most of it spent confirming nothing changed.

TEST THE EXTREME INSTEAD. 50m is the one scale that is FULLY pretrained (540,423), and
its configuration was chosen at 133,233 -- 25%. That 4x gap is the largest available
anywhere in the project. If the optimum does not move between a quarter-trained and a
fully-trained encoder, it will not move between the 41% and 61% rungs the checkpoint
ladder actually uses, and the question can be closed rather than re-asked per cell.

If it DOES move, that is worth knowing early and cheaply, before twenty-one denoise
arms and forty-two contrastive arms are spent at a stale configuration.

WHAT EACH TASK SWEEPS, and why they differ. Denoise costs twelve tiles an arm, so it
probes only the two axes the 216-arm grid showed mattered -- learning rate and
encoder_lr_scale; head width and epochs were inside the noise there. Contrastive costs
one tile and five minutes, so it re-runs the full lr x KL x temperature grid.
"""

from __future__ import annotations

import argparse
import itertools
import shutil
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
FROZEN = "/flare/UIC-HPC/khuss/msdelta/pretrained"
SCALE = "50m"
CKPT = "540423"          # fully pretrained; the configuration in use came from 133233

TASKS = {
    "denoise": dict(
        out="sweep-ckpthp-denoise", prefix="v2_ckhpdn-",
        template=REPO / "configs" / "finetune-denoise-50m-ds" / "training.args",
        seeds=("1", "2"),
        axes={"--learning_rate": {"lr5e5": "5e-5", "lr2e4": "2e-4", "lr5e4": "5e-4"},
              "--encoder_lr_scale": {"es05": "0.5", "es10": "1.0"}},
        fixed={"--num_train_epochs": "4", "--head_hidden_size": "512",
               "--per_device_train_batch_size": "1",
               "--gradient_accumulation_steps": "1"}),
    "contrastive": dict(
        out="sweep-ckpthp-contrastive", prefix="v2_ckhpcon-",
        template=REPO / "configs" / "finetune-contrastive-50m" / "training.args",
        seeds=("0", "1", "2", "3"),
        axes={"--learning_rate": {"lr2e5": "2e-5", "lr1e4": "1e-4", "lr5e4": "5e-4"},
              "--kl_weight": {"kl0": "0", "kl10": "10"},
              "--temperature": {"t007": "0.07", "t02": "0.2"}},
        fixed={"--num_train_epochs": "3", "--groups_per_batch": "2",
               "--replicates": "2", "--split_seed": "0"}),
}


def combos(task):
    spec = TASKS[task]
    keys = list(spec["axes"])
    return [(dict(zip(keys, c)), sd)
            for c in itertools.product(*(spec["axes"][k] for k in keys))
            for sd in spec["seeds"]]


def render(task, choice, seed):
    spec = TASKS[task]
    name = "_".join(choice[k] for k in spec["axes"]) + f"_seed{seed}"
    over = dict(spec["fixed"])
    over |= {k: spec["axes"][k][v] for k, v in choice.items()}
    over |= {
        "--pretrained_path": f"{FROZEN}/msdelta-{SCALE}-production-01-checkpoint-{CKPT}",
        "--seed": seed,
        "--run_name": f"{spec['prefix']}{name}",
        "--output_dir": f"./runs/{spec['prefix']}{name}",
    }
    lines, seen = [], set()
    for flag, value in zip(*[iter(spec["template"].read_text().split())] * 2):
        lines.append(f"{flag} {over.get(flag, value)}")
        seen.add(flag)
    for flag, value in over.items():
        if flag not in seen:
            lines.append(f"{flag} {value}")
    return name, "\n".join(lines) + "\n"


def describe(task, choice, seed):
    settings = ", ".join(f"{k.lstrip('-')} {TASKS[task]['axes'][k][v]}"
                         for k, v in choice.items())
    return (f"CHECKPOINT-HP ARM ({task}): {SCALE} at checkpoint {int(CKPT):,} -- fully "
            f"pretrained -- with {settings}, seed {seed}. Asks whether the "
            f"configuration chosen at checkpoint 133,233 (25% pretrained) is still the "
            f"optimum at 100%. This is the largest pretraining gap available in the "
            f"project; if the answer is no shift, the question closes and the "
            f"checkpoint ladder can keep using the existing configuration rather than "
            f"sweeping at every rung.\n")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--task", required=True, choices=sorted(TASKS))
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--clean", action="store_true")
    cli = ap.parse_args()
    spec = TASKS[cli.task]
    out = REPO / "configs" / spec["out"]
    if not Path(FROZEN, f"msdelta-{SCALE}-production-01-checkpoint-{CKPT}",
                "model.safetensors").exists() and not cli.check:
        raise SystemExit(f"{SCALE} checkpoint {CKPT} is not frozen")
    cs = combos(cli.task)

    if cli.check:
        stale = [n for n, t in (render(cli.task, *c) for c in cs)
                 if not (out / n / "training.args").exists()
                 or (out / n / "training.args").read_text() != t
                 or not (out / n / "DESCRIPTION.md").exists()]
        if stale:
            print(f"  {len(stale)} stale or missing: {', '.join(stale[:6])}")
            return 1
        print(f"  {len(cs)} arms match {spec['template'].name}")
        return 0

    if cli.clean and out.exists():
        shutil.rmtree(out)
    for c in cs:
        name, text = render(cli.task, *c)
        (out / name).mkdir(parents=True, exist_ok=True)
        (out / name / "training.args").write_text(text)
        (out / name / "DESCRIPTION.md").write_text(describe(cli.task, *c))
    (out / ".template").write_text(f"{spec['template'].name}\nckpthp-{cli.task}\n")
    print(f"arms={len(cs)} under configs/{spec['out']}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
