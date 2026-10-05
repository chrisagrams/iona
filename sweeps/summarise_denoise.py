"""Turn finished denoise sweep runs into the grid tables in $MSDELTA_DERIVED/denoise/.

    python sweeps/summarise_denoise.py --runs /lus/flare/projects/UIC-HPC/$USER/msdelta/runs

Each grid writes one file, rebuilt from whatever run directories it finds: check that every
run a table lists is still on /flare before rerunning it over an existing table. K198a moved the
tables out of git into $MSDELTA_DERIVED/denoise/ (derived data; notes/AGENT_PLAYBOOK_2.md C7).

WHY EVERY TABLE PRINTS `spectra` AND `peaks`. The first 216-arm grid (job 8840345) scored
1,440 test spectra where every later job scored 8,584 of the same 9,893-row split. Same
dataset (one cached revision, arrow files untouched between runs), byte-identical configs,
no change to the split code, twelve tiles in both. The cause is UNIDENTIFIED. Its numbers
therefore sit in their own file and must never be compared with the rest, and printing the
scored count in every header is what makes such a mismatch visible next time instead of
silently halving an AUROC.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import homes  # noqa: E402  (data homes, configs/homes.env)

REPO = Path(__file__).resolve().parent.parent
OUT = homes.DERIVED / "denoise"

# job id -> (filename, title, how to split an arm name into columns)
GRIDS = {
    "8840408": ("grid_denoise_50m.txt", "50m denoise hyperparameter grid, 216 arms"),
    "8841345": ("grid_denoise_by_scale.txt", "100m"),
    "8841992": ("grid_denoise_by_scale.txt", "200m"),
    "8842147": ("grid_denoise_by_scale.txt", "400m"),
    "8845252": ("grid_denoise_400m_probe.txt",
                "400m: encoder_lr_scale x epochs, plus repeats of the grid winner"),
    "8841984": ("grid_denoise_scratch.txt",
                "50m with a RANDOM encoder -- the pretraining ablation"),
    "8840345": ("grid_denoise_50m_earlytest.txt",
                "50m grid on the REDUCED evaluation -- NOT COMPARABLE"),
}
AXES = re.compile(r"lr(?P<lr>[0-9a-z]+)(?:_es(?P<es>[0-9]+))?(?:_ep(?P<ep>[0-9]+))?"
                  r"(?:_h(?P<h>[0-9]+))?(?:_b(?P<b>[0-9]+))?$")
PRETTY = {"1e6": "1e-6", "1e5": "1e-5", "5e5": "5e-5", "1e4": "1e-4", "2e4": "2e-4",
          "5e4": "5e-4", "01": "0.1", "025": "0.25", "05": "0.5", "10": "1.0", "0": "0"}


def load(runs: Path, job: str) -> dict[str, dict]:
    out = {}
    for d in sorted(runs.glob(f"sweep-*-{job}")):
        arm = re.sub(rf"^sweep-|-{job}$", "", d.name)
        f = d / "test_results.json"
        if f.exists():
            try:
                out[arm] = json.loads(f.read_text())
            except Exception:
                pass
    return out


def columns(arm: str) -> dict[str, str]:
    m = AXES.match(arm)
    if not m:
        return {}
    return {k: PRETTY.get(v, v) for k, v in m.groupdict().items() if v}


def spectra(rec: dict) -> float:
    return rec.get("test_runtime", 0) * rec.get("test_samples_per_second", 0)


def table(rows: list[tuple[str, dict]], cols: list[str]) -> list[str]:
    head = f"  {'arm':<28}" + "".join(f"{c:>8}" for c in cols) + \
           f"{'AUROC':>9}{'F1':>8}{'AUPRC':>8}"
    lines = [head, "  " + "-" * (len(head) - 2)]
    for arm, rec in rows:
        c = columns(arm)
        lines.append(f"  {arm:<28}" + "".join(f"{c.get(k,''):>8}" for k in cols) +
                     f"{rec['test_auroc']:>9.4f}{rec['test_f1']:>8.4f}"
                     f"{rec['test_auprc']:>8.4f}")
    return lines


def header(title: str, job: str, recs: dict) -> list[str]:
    sp = {round(spectra(r)) for r in recs.values()}
    pk = {int(r["test_n_peaks"]) for r in recs.values()}
    return [f"# {title}",
            f"# job {job}, {len(recs)} arms with results",
            f"# scored {min(sp):,}-{max(sp):,} test spectra / {min(pk):,}-{max(pk):,} peaks"
            f"  (test split is 9,893 spectra)",
            f"# metric: test_auroc on per-peak noise classification, sorted best first",
            ""]


def write_50m(runs: Path) -> None:
    recs = load(runs, "8840408")
    rows = sorted(recs.items(), key=lambda kv: -kv[1]["test_auroc"])
    v = [r["test_auroc"] for _, r in rows]
    top8 = v[:8]
    body = header(GRIDS["8840408"][1], "8840408", recs) + [
        "THE TOP CLUSTER IS NOT RESOLVED. The best eight arms span "
        f"{max(top8)-min(top8):.4f} across three head",
        "widths and two encoder scales, against a seed noise of 0.0005 measured at a fixed",
        "configuration (denoise_scale_seeds.txt). Most of the ranking inside that",
        "cluster is not a measurement. Do not quote a winner without the error bar.",
        "",
        "encoder_lr_scale 0 means a FROZEN encoder: those arms collapse, which is the one",
        "unambiguous result in this grid.",
        "",
    ] + table(rows, ["lr", "es", "ep", "h", "b"])
    (OUT / "grid_denoise_50m.txt").write_text("\n".join(body) + "\n")
    return len(rows)


def write_by_scale(runs: Path) -> int:
    out, n = [], 0
    for job, scale in (("8841345", "100m"), ("8841992", "200m"), ("8842147", "400m")):
        recs = load(runs, job)
        n += len(recs)
        rows = sorted(recs.items(), key=lambda kv: -kv[1]["test_auroc"])
        out += [f"############ {scale}"] + header(
            f"{scale} denoise hyperparameter grid", job, recs)[1:] + \
            table(rows, ["lr", "es", "b"]) + [""]
    head = ["# Per-scale denoise hyperparameter grids, 12 arms each",
            "# Identical axes at every scale, so the winners are directly comparable.",
            "#",
            "# THE SAME POINT WINS AT EVERY SCALE: lr2e4_es05_b12 (lr 2e-4, encoder at 0.5x",
            "# the head's rate, effective batch 12) is first at 100m, 200m and 400m, and the",
            "# same settings won the 216-arm 50m grid. Four scales, one answer -- which is",
            "# what licenses repeating a single configuration across scales in FT5.",
            "#",
            "# The ranking WITHIN each top cluster is not resolved: the best three arms span",
            "# 0.0003-0.0008 against a seed noise of 0.0005.",
            ""]
    (OUT / "grid_denoise_by_scale.txt").write_text("\n".join(head + out) + "\n")
    return n


def write_probe(runs: Path) -> int:
    recs = load(runs, "8845252")
    rows = sorted(recs.items(), key=lambda kv: -kv[1]["test_auroc"])
    rep = [r["test_auroc"] for a, r in recs.items() if a.startswith("repeat") or a == "es05_ep4"]
    extra = []
    if len(rep) > 1:
        extra = [f"FIXED-SEED REPRODUCIBILITY: {len(rep)} byte-identical configs, same --seed, "
                 f"give sd {st.stdev(rep):.5f}",
                 "(mean {:.4f}). That is XPU reduction nondeterminism alone. Across SEEDS it is"
                 .format(sum(rep)/len(rep)),
                 "~0.0005; use that one for anything compared across configurations.", ""]
    body = header(GRIDS["8845252"][1], "8845252", recs) + [
        "WHY THIS GRID EXISTS: 400m scores BELOW 200m, and this asks whether that is the",
        "encoder being over-written by fine-tuning, or 400m simply being read too early.",
        "",
        "THE ANSWER IS NEITHER. With the 8-epoch column in, BOTH axes turn over. At the",
        "winning encoder_lr_scale of 0.5, eight epochs scores 0.0021 BELOW four -- five",
        "times the fixed-seed noise. The lower the encoder rate the later the peak (0.1",
        "is still climbing at eight epochs) but the throttled arms never catch up: 0.1 at",
        "eight epochs reaches 0.9399 against 0.5 at four reaching 0.9439.",
        "",
        "The best 400m result anywhere in this grid is 0.9439, still below 200m's 0.9447",
        "at four epochs. More fine-tuning does not rescue 400m.",
        "",
        "AN EARLIER READING OF THIS GRID, BEFORE THE EIGHT-EPOCH COLUMN LANDED, said 400m",
        "was under-trained rather than over-written, reasoning from ep2 -> ep4 being",
        "+0.0021. That extrapolated a trend past the last sampled point, and the next",
        "point reversed it.",
        "",
        "encoder_lr_scale is an INVERTED U. This grid samples 0.1/0.25/0.5 and rises",
        "monotonically; 1.0 was run in the 400m grid (grid_denoise_by_scale.txt) and scores",
        "0.9396 against 0.5's 0.9436. The peak is at 0.5 and the rise does not continue.",
        "",
    ] + extra + table(rows, ["es", "ep"])
    (OUT / "grid_denoise_400m_probe.txt").write_text("\n".join(body) + "\n")
    return len(rows)


def write_scratch(runs: Path) -> int:
    recs = load(runs, "8841984")
    rows = sorted(recs.items(), key=lambda kv: -kv[1]["test_auroc"])
    pre = load(runs, "8840408")
    def best(rs, ep):
        v = [r["test_auroc"] for a, r in rs.items() if re.search(rf"_ep{ep}(?:_|$)", a)]
        return max(v) if v else None
    p4, s4, s8 = best(pre, 4), best(recs, 4), best(recs, 8)
    # FT13, job 8847610. NOT 8846027: that one ran without TILES_PER_ARM=12 and so
    # trained at effective batch 1 instead of 12 -- a different experiment, discarded.
    ft13 = load(runs, "8847610")
    p8 = max((r["test_auroc"] for r in ft13.values()), default=float("nan"))
    body = header(GRIDS["8841984"][1], "8841984", recs) + [
        "--random_init true: the same architecture with the encoder reinitialised, so",
        "the only thing removed is pretraining. Everything else -- data, head, schedule,",
        "test split -- is held.",
        "",
        "THE GRIDS DID NOT SWEEP THE SAME EPOCH COUNTS: pretrained ran 2 and 4, this",
        "one ran 4 and 8, which left two comparisons answering different questions and",
        "one hole. FT13 filled it.",
        "",
        f"    matched at 4 epochs ......... {p4:.4f} vs {s4:.4f}   pretraining worth "
        f"+{p4-s4:.4f}",
        f"    scratch at double budget .... {p4:.4f} vs {s8:.4f}   pretraining worth "
        f"+{p4-s8:.4f}",
        f"    matched at 8 epochs ......... {p8:.4f} vs {s8:.4f}   pretraining worth "
        f"+{p8-s8:.4f}",
        "",
        "FT13 (job 8847610, six seeds) filled the hole, and the ambiguity turned out not",
        "to matter much: the pretrained model gains only +0.0014 from 4 to 8 epochs, so",
        "the matched-at-8 figure and the old cross-budget one nearly coincide.",
        "",
        "THE GAP NARROWS WITH BUDGET, +0.046 matched at 4 against +0.033 matched at 8,",
        "because scratch gains ten times as much from the extra epochs (+0.0145 against",
        "+0.0014). Whether it narrows to nothing is open -- sweep-denoise-scratch-scale",
        "runs 50m at 16 epochs for that.",
        "",
        "What is unambiguous either way: the pretrained model at TWO epochs already",
        f"scores {best(pre,2):.4f}, above this ablation's best at eight.",
        "",
    ] + table(rows, ["lr", "ep", "b"])
    (OUT / "grid_denoise_scratch.txt").write_text("\n".join(body) + "\n")
    return len(rows)


def write_earlytest(runs: Path) -> int:
    recs = load(runs, "8840345")
    rows = sorted(recs.items(), key=lambda kv: -kv[1]["test_auroc"])
    body = header(GRIDS["8840345"][1], "8840345", recs) + [
        "#" * 78,
        "DO NOT COMPARE THESE NUMBERS WITH ANY OTHER FILE IN results/.",
        "",
        "This grid scored 1,440 test spectra. Every later job scored 8,584 of the same",
        "9,893-row split. The cause is UNIDENTIFIED, and the usual explanations are all",
        "ruled out:",
        "    dataset changed .... no. One cached revision, arrow files written 2026-09-11",
        "                         and untouched between the runs; HF_HUB_OFFLINE=1.",
        "    config differed .... no. Byte-identical to job 8840408 apart from the W&B",
        "                         project name.",
        "    split code changed . no commits to data.py or the split path between the two.",
        "    fewer tiles ........ no. Twelve per arm in both.",
        "    peaks truncated .... no. 193 peaks/spectrum here, 196 later.",
        "",
        "It scored almost exactly one sixth of what the others did (8,584/6 = 1,431). That",
        "shape suggests a gather over a subset of ranks, but twelve tiles does not divide",
        "into six by anything identified in the code. Resolving it means bisecting the",
        "runner at that commit.",
        "",
        "Kept because deleting a measurement is worse than labelling it. The grid's SHAPE",
        "-- which axes matter -- may still be informative; its LEVELS are not comparable.",
        "#" * 78,
        "",
    ] + table(rows, ["lr", "es", "ep", "h", "b"])
    (OUT / "grid_denoise_50m_earlytest.txt").write_text("\n".join(body) + "\n")
    return len(rows)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs", default="/lus/flare/projects/UIC-HPC/khuss/msdelta/runs")
    cli = ap.parse_args()
    runs = Path(cli.runs)
    if not runs.is_dir():
        raise SystemExit(f"no run directory at {runs}")
    OUT.mkdir(exist_ok=True)
    print(f"  grid_denoise_50m.txt            {write_50m(runs)} arms")
    print(f"  grid_denoise_by_scale.txt       {write_by_scale(runs)} arms")
    print(f"  grid_denoise_400m_probe.txt     {write_probe(runs)} arms")
    print(f"  grid_denoise_scratch.txt        {write_scratch(runs)} arms")
    print(f"  grid_denoise_50m_earlytest.txt  {write_earlytest(runs)} arms")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
