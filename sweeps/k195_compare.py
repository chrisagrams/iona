"""K195a-P: today's masked-intensity loss vs the proposal (KL to intensity^0.5), 25m debug-hour arms vs Chris's 25m.

    .venv/bin/python sweeps/k195_compare.py

Arms (Chris's 25m production-01 recipe, 20 shards, global batch 528, compiled; stopped by the 1 h walltime):
  A 8903620  trains on today's loss          B 8903621  trains on proposal_loss
Both log 'loss' (today's KL to the linear intensity share) and 'proposal_loss' (KL to intensity^0.5), train and eval.
Steps come from trainer_state.json (checkpoint-5000) and, after it, from the logged epoch x steps per epoch.

Reference lines: results/raw/diag/k195/k195a_floors.json (each loss for a model that fits the OTHER target
perfectly, and for a uniform prediction).

Writes results/raw/diag/k195/k195a_<arm>_log.json and results/processed/figures/P_pretrain/k195a_losses.png.
"""

from __future__ import annotations

import ast
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RUNS = Path("/lus/flare/projects/UIC-HPC/khuss/msdelta/runs/k195")
ARMS = {"A": ("8903620", "A: trains on today's loss", "#2563eb"), "B": ("8903621", "B: trains on proposal_loss", "#dc2626")}
OUT = ROOT / "results/raw/diag/k195"
FIGS = ROOT / "results/processed/figures/P_pretrain"


def load(arm, job):
    run = RUNS / f"k195-25m-{arm}-{job}"
    state = json.loads(max(run.glob("checkpoint-*/trainer_state.json"),
                           key=lambda p: int(p.parent.name.split("-")[1])).read_text())
    hist = state["log_history"]
    last = hist[-1]
    per_epoch = last["step"] / last["epoch"]
    seen = {(h["step"], "eval_loss" in h) for h in hist}
    for line in (run / "logs" / f"train-{job}.out").read_text().splitlines():
        for m in re.finditer(r"\{'(?:loss|eval_loss)'[^}]*\}", line):
            d = {k: float(v) for k, v in ast.literal_eval(m.group(0)).items()}
            d["step"] = int(round(d["epoch"] * per_epoch))
            if (d["step"], "eval_loss" in d) not in seen and d["step"] > last["step"]:
                hist.append(d); seen.add((d["step"], "eval_loss" in d))
    train = sorted((h for h in hist if "loss" in h and "eval_loss" not in h), key=lambda h: h["step"])
    ev = sorted((h for h in hist if "eval_loss" in h), key=lambda h: h["step"])
    return train, ev


def main() -> int:
    chris = json.loads((OUT / "chris25m_log.json").read_text())["rows"]
    FIGS.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
    data = {}
    for arm, (job, label, colour) in ARMS.items():
        train, ev = load(arm, job)
        data[arm] = (train, ev)
        (OUT / f"k195a_{arm}_log.json").write_text(json.dumps({"job": job, "train": train, "eval": ev}, indent=0))
        axes[0].plot([h["step"] for h in train], [h["loss"] for h in train], color=colour, lw=1.2, label=f"{label} (train)")
        axes[0].plot([h["step"] for h in ev], [h["eval_loss"] for h in ev], color=colour, lw=2, ls="--", marker="o",
                     ms=3, label=f"{label} (eval)")
        axes[1].plot([h["step"] for h in train], [h["proposal_loss"] for h in train], color=colour, lw=1.2,
                     label=f"{label} (train)")
        axes[1].plot([h["step"] for h in ev], [h["eval_proposal_loss"] for h in ev], color=colour, lw=2, ls="--",
                     marker="o", ms=3, label=f"{label} (eval)")
        axes[2].plot([h["eval_loss"] for h in ev], [h["eval_proposal_loss"] for h in ev], color=colour, marker="o",
                     ms=3, label=label)
    steps_max = max(h["step"] for t, _ in data.values() for h in t)
    cr = [r for r in chris if r["step"] <= steps_max]
    axes[0].plot([r["step"] for r in cr], [r["loss"] for r in cr], color="#6b7280", lw=1.2, label="Chris 25m (train)")
    fl = json.loads((OUT / "k195a_floors.json").read_text())
    axes[0].axhline(fl["todays_loss_of_a_perfect_proposal_model"], color="#dc2626", ls=":", lw=1,
                    label="floor for B: a perfect proposal model scores this")
    axes[1].axhline(fl["proposal_loss_of_a_perfect_todays_loss_model"], color="#2563eb", ls=":", lw=1,
                    label="a perfect today's-loss model scores this")
    axes[1].axhline(fl["uniform_proposal_loss"], color="#6b7280", ls=":", lw=1, label="uniform prediction")
    axes[0].set(title="today's loss (KL to linear intensity share)", xlabel="step", ylabel="loss", yscale="log")
    axes[1].set(title="proposal_loss (KL to intensity^0.5 share)", xlabel="step", ylabel="proposal_loss", yscale="log")
    axes[2].set(title="eval: both losses, one point per eval", xlabel="eval_loss",
                ylabel="eval_proposal_loss", xscale="log", yscale="log")
    for ax in axes:
        ax.grid(alpha=0.3); ax.legend(frameon=False, fontsize=7)
    fig.suptitle("K195a-P, 25m, Chris's recipe on 20 shards (global batch 528), 1 h debug each; train curves are "
                 "per-log-step means", x=0.01, ha="left", fontsize=9, color="#6b7280")
    fig.tight_layout(); out = FIGS / "k195a_losses.png"; fig.savefig(out, dpi=150); plt.close(fig)
    print(out)
    for arm, (train, ev) in data.items():
        print(f"\n{arm}: last train step {train[-1]['step']}")
        print("  step   eval_loss  eval_proposal")
        for h in ev:
            print(f"  {h['step']:5d}  {h['eval_loss']:.4f}     {h['eval_proposal_loss']:.4f}")
    print("\ntrain loss at Chris's steps (mean of the logged values within +-50 steps):")
    for s in (1000, 2000, 3000, 4000, 5000, 6000):
        c = [r["loss"] for r in chris if abs(r["step"] - s) <= 50]
        row = [f"{s:5d}", f"Chris {sum(c) / len(c):.4f}" if c else "Chris --"]
        for arm, (train, _) in data.items():
            v = [h for h in train if abs(h["step"] - s) <= 50]
            row.append(f"{arm} loss {sum(h['loss'] for h in v) / len(v):.4f} prop {sum(h['proposal_loss'] for h in v) / len(v):.4f}"
                       if v else f"{arm} --")
        print("  " + "  ".join(row))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
