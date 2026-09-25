"""C11 (PLAN.md): a spectrum-retrieval benchmark neither GLEAMS nor our encoders trained on.

    PYTHONPATH=. .venv/bin/python baselines_wip/c11_build.py OUT_DIR RUN [RUN ...]

Source: Gaolaboratory/psm-rerank-hek-hct116 (pinned revision, spectra as stage 1 read
them). Labels: MSFragger's rank-1 candidate of each spectrum, kept when it is a TARGET at
q <= 1% (TDC on -log10 e-value, per run) -- the same confident-PSM definition as the
reranking baseline. Groups = peptide + charge (our notation, as every retrieval metric).
Kept: spectra with 1..512 peaks (our encoders' limit, so every method sees the SAME set)
in groups of >= 2 spectra (a query needs another relevant spectrum).

Writes, from the SAME rows in the SAME order:
  OUT/prepared/   the eval_grouped_retrieval `score` format (processed by MSDeltaProcessor)
  OUT/export/     meta.parquet + experimental.mgf (+ empty consensus.mgf): the format the
                  GLEAMS / yHydra embedders and score_retrieval.py already read
"""
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

S = "/lus/flare/projects/UIC-HPC/khuss/msdelta"
ROWS = f"{S}/rerank-psm/a1-050m-c7s600/rows"
PROCESSOR = "/flare/UIC-HPC/khuss/msdelta/pretrained/msdelta-50m-production-01-checkpoint-220000"
MAX_PEAKS = 512


def confident_psms(run: str) -> dict:
    """spectrum_id -> (peptide, charge) for rank-1 targets at q <= 1% within the run."""
    from msdelta.rerank_psm_fdr import qvalues
    t = pq.read_table(f"{ROWS}/{run}.parquet", columns=[
        "spectrum_id", "peptide", "charge", "is_decoy", "search_rank",
        "search_neglog10_evalue"]).to_pandas()
    top = t[t["search_rank"] == 1].drop_duplicates("spectrum_id")
    s = top["search_neglog10_evalue"].fillna(-1e9).to_numpy(float)
    d = top["is_decoy"].to_numpy(bool)
    q = qvalues(s, d)
    keep = top[(q <= 0.01) & ~d]
    return dict(zip(keep["spectrum_id"], zip(keep["peptide"], keep["charge"].astype(int))))


def main(out_dir, *runs):
    from huggingface_hub import hf_hub_download
    from datasets import Dataset
    from msdelta.grouped_retrieval import _process
    from msdelta.processing_msdelta import MSDeltaProcessor
    from msdelta.rerank_psm_embed import REPO_ID, REVISION

    out = Path(out_dir)
    if out.exists():
        raise SystemExit(f"{out} exists; refusing to overwrite")
    processor = MSDeltaProcessor.from_pretrained(PROCESSOR, max_peaks=MAX_PEAKS)
    recs, stats = [], {}
    for run in runs:
        labels = confident_psms(run)
        ds = "HCT116" if "HCT116" in run else "HEK293"
        path = hf_hub_download(REPO_ID, f"{ds}/{run}.parquet", repo_type="dataset",
                               revision=REVISION)
        tab = pq.read_table(path, columns=["spectrum_id", "precursor_mz", "charge", "mz",
                                           "intensity"]).to_pylist()
        n_conf = n_kept = 0
        for r in tab:
            lab = labels.get(r["spectrum_id"])
            if lab is None:
                continue
            n_conf += 1
            mz = np.asarray(r["mz"], np.float64); it = np.asarray(r["intensity"], np.float64)
            ok = (it > 0) & np.isfinite(mz)
            mz, it = mz[ok], it[ok]
            if not (0 < len(mz) <= MAX_PEAKS):
                continue
            n_kept += 1
            recs.append({"spectrum_id": r["spectrum_id"], "run": run, "dataset": ds,
                         "peptide": lab[0], "charge": int(lab[1]),
                         "precursor": float(r["precursor_mz"]), "mz": mz, "intensity": it})
        stats[run] = {"confident": n_conf, "kept_le512": n_kept}
        print(f"[c11] {run}: {n_conf:,} confident PSMs, {n_kept:,} with 1..{MAX_PEAKS} peaks",
              flush=True)
    size = Counter((r["peptide"], r["charge"]) for r in recs)
    recs = [r for r in recs if size[(r["peptide"], r["charge"])] >= 2]
    groups = len({(r["peptide"], r["charge"]) for r in recs})
    print(f"[c11] {len(recs):,} spectra in {groups:,} peptide+charge groups (>= 2 each)",
          flush=True)

    # our prepared format
    prep = {k: [] for k in ("mz", "log_intensity", "peptide", "charge", "precursor",
                            "source", "analyte_id")}
    for r in recs:
        m, li = _process(processor, r["mz"], r["intensity"])
        prep["mz"].append(m); prep["log_intensity"].append(li)
        prep["peptide"].append(r["peptide"]); prep["charge"].append(r["charge"])
        prep["precursor"].append(r["precursor"]); prep["source"].append("experimental")
        prep["analyte_id"].append(f"{r['peptide']}_{r['charge']}")
    Dataset.from_dict(prep).save_to_disk(str(out / "prepared"))
    info = {"source": f"{REPO_ID}@{REVISION}", "runs": list(runs), "per_run": stats,
            "spectra": len(recs), "groups": groups, "max_peaks": MAX_PEAKS,
            "labels": "MSFragger rank-1 targets at q<=1% (per-run TDC on -log10 e-value)"}
    (out / "prepared" / "PREPARED.json").write_text(json.dumps(info, indent=1))

    # the external-baseline export (same rows, same order)
    exp = out / "export"; exp.mkdir(parents=True)
    with open(exp / "experimental.mgf", "w") as f:
        for i, r in enumerate(recs):
            f.write("BEGIN IONS\n")
            f.write(f"TITLE=row={i};analyte_id={r['peptide']}_{r['charge']};"
                    f"source=experimental;spectrum_id={r['spectrum_id']}\n")
            f.write(f"PEPMASS={r['precursor']:.6f}\nCHARGE={r['charge']}+\nRTINSECONDS=0\n")
            for m, it in zip(r["mz"], r["intensity"]):
                f.write(f"{m:.5f} {it:.4f}\n")
            f.write("END IONS\n")
    (exp / "consensus.mgf").write_text("")
    pq.write_table(pa.table({
        "row": np.arange(len(recs)), "analyte_id": [f"{r['peptide']}_{r['charge']}" for r in recs],
        "peptide": [r["peptide"] for r in recs], "charge": [r["charge"] for r in recs],
        "precursor": [r["precursor"] for r in recs], "source": ["experimental"] * len(recs),
        "spectrum_id": [r["spectrum_id"] for r in recs],
        "n_peaks": [len(r["mz"]) for r in recs], "in_mp512": [True] * len(recs)}),
        exp / "meta.parquet")
    (exp / "EXPORT.json").write_text(json.dumps(info, indent=1))
    print(f"[c11] wrote {out}/prepared and {out}/export", flush=True)


if __name__ == "__main__":
    main(*sys.argv[1:])
