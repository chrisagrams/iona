#!/usr/bin/env python3
"""Plot within-peptide (confound-free) target-side findability vs spectrum quality.
Reads runs/tercile_retrieval.json and renders find@1 / find@5 across low/mid/high
within-peptide score bins for both checkpoints. Whitened variant is the primary."""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

J = json.load(open("runs/tercile_retrieval.json"))
SCORE = "max_score"
BINS = ["low", "mid", "high"]
x = [0, 1, 2]

# short, readable checkpoint labels
def short(name):
    n = Path(name).parent.name
    return n.replace("_7235042", "").replace("_7232144", "")

cks = list(J["checkpoints"].keys())
colors = {"consensus_xl_28M": "#1f77b4", "v14_cap_XL_hf": "#d62728"}

fig, axes = plt.subplots(1, 2, figsize=(12, 5.2), sharex=True)
fig.suptitle("Within-peptide target-side findability vs spectrum quality "
             f"(confound-free; {SCORE} terciles, whitened embedding)",
             fontsize=13, fontweight="bold")

for ax, metric, title in [
    (axes[0], "findability_top1", "find@1  (top-1 hit rate over sibling queries)"),
    (axes[1], "findability_top5", "find@5  (top-5 hit rate over sibling queries)"),
]:
    # blue labels well above, red labels well below, with white bbox for legibility
    yoff = {"consensus_xl_28M": 15, "v14_cap_XL_hf": -20}
    for ck in cks:
        lab = short(ck)
        by_bin = J["checkpoints"][ck]["target_findability_within"][SCORE]["variants"]["w"]["by_bin"]
        d = {r["bin"]: r for r in by_bin}
        y = [d[b][metric] for b in BINS]
        sp = J["checkpoints"][ck]["target_findability_within"][SCORE]["variants"]["w"]["mean_spearman"]
        col = colors.get(lab, None)
        ax.plot(x, y, "o-", lw=2.2, ms=9, color=col,
                label=f"{lab}  (ρ={sp:+.3f})")
        for xi, yi in zip(x, y):
            ax.annotate(f"{yi:.4f}", (xi, yi), textcoords="offset points",
                        xytext=(0, yoff.get(lab, 9)), ha="center", fontsize=8.5,
                        color=col,
                        bbox=dict(boxstyle="round,pad=0.15", fc="white",
                                  ec="none", alpha=0.7))
    ax.set_xticks(x)
    ax.set_xticklabels([f"{b}\n(n=22,264)" for b in BINS])
    ax.set_xlabel("within-peptide score bin")
    ax.set_ylabel(metric.replace("findability_", "find@").replace("top", ""))
    ax.set_title(title, fontsize=11)
    ax.grid(alpha=0.3)
    ax.legend(title="checkpoint (Spearman score,earliness)", fontsize=9)
    # headroom for annotations both above and below
    lo, hi = ax.get_ylim(); pad = (hi - lo) * 0.18; ax.set_ylim(lo - pad, hi + pad)

fig.text(0.5, 0.005,
         "Full mixed gallery (66,792 spectra, 2,024 peptides). Every peptide "
         "contributes equally to all 3 bins, so a low→high rise isolates spectrum "
         "quality from peptide identity. Effect is real, monotonic, and small.",
         ha="center", fontsize=8.5, style="italic")
fig.tight_layout(rect=[0, 0.03, 1, 0.96])
out = "runs/findability_within_peptide.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
print("wrote", out)
