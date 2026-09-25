"""Nine-species (yeast test split): an unseen HIGH-RES HCD retrieval benchmark.

    PYTHONPATH=. .venv/bin/python baselines_wip/nine_build.py OUT_DIR

Source: InstaDeepAI/ms_ninespecies_benchmark (DeepNovo's nine-species, MassIVE
MSV000081382; InstaNovo's cleaned copy), `test` split = S. cerevisiae, 111,312 labelled
spectra. Every spectrum has <= 452 peaks, so nothing is trimmed or dropped for our 512-peak
encoders: spectra are used exactly as published.

Groups = modified peptide + charge. Kept: groups of >= 2 spectra, each capped at
MAX_PER_GROUP drawn at random (seed 0), as in C11. Modifications are converted to our
notation: C(+57.02) -> C[57.0215], M(+15.99) -> M[15.9949], N/Q(+.98) -> [0.9840]; any other
modification string stops the build (never silently mapped).

Writes the same two formats as c11_build.py, from the SAME rows in the SAME order:
  OUT/prepared/   eval_grouped_retrieval `score` format
  OUT/export/     meta.parquet + experimental.mgf (+ empty consensus.mgf)
"""
import json
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

REPO = "InstaDeepAI/ms_ninespecies_benchmark"
REVISION = "95d03bc005e512bd20c6abb3f8650e327272973a"
FILE = "data/test-00000-of-00001.parquet"
PROCESSOR = "/flare/UIC-HPC/khuss/msdelta/pretrained/msdelta-50m-production-01-checkpoint-220000"
MAX_PEAKS = 512
MAX_PER_GROUP = 20
MODS = {"C(+57.02)": "C[57.0215]", "M(+15.99)": "M[15.9949]",
        "N(+.98)": "N[0.9840]", "Q(+.98)": "Q[0.9840]"}


def to_notation(modified: str) -> str:
    out = modified
    for k, v in MODS.items():
        out = out.replace(k, v)
    if "(" in out or ")" in out:
        raise SystemExit(f"unmapped modification in {modified!r}")
    return out


def main(out_dir, *_ignored):
    from huggingface_hub import hf_hub_download
    from datasets import Dataset
    from msdelta.grouped_retrieval import _process
    from msdelta.processing_msdelta import MSDeltaProcessor

    out = Path(out_dir)
    if out.exists():
        raise SystemExit(f"{out} exists; refusing to overwrite")
    processor = MSDeltaProcessor.from_pretrained(PROCESSOR, max_peaks=MAX_PEAKS)
    path = hf_hub_download(REPO, FILE, repo_type="dataset", revision=REVISION)
    rows = pq.read_table(path).to_pylist()
    recs = []
    for i, r in enumerate(rows):
        mz = np.asarray(r["mz_array"], np.float64); it = np.asarray(r["intensity_array"], np.float64)
        ok = (it > 0) & np.isfinite(mz)
        mz, it = mz[ok], it[ok]
        if not (0 < len(mz) <= MAX_PEAKS):
            raise SystemExit(f"row {i}: {len(mz)} peaks -- outside 1..{MAX_PEAKS}, refusing to "
                             f"trim or drop silently")
        recs.append({"spectrum_id": f"yeast:{i}", "peptide": to_notation(r["modified_sequence"]),
                     "charge": int(r["precursor_charge"]), "precursor": float(r["precursor_mz"]),
                     "mz": mz, "intensity": it})
    n_all = len(recs)
    size = Counter((r["peptide"], r["charge"]) for r in recs)
    recs = [r for r in recs if size[(r["peptide"], r["charge"])] >= 2]
    rng = np.random.default_rng(0)
    seen: Counter = Counter(); keep = np.zeros(len(recs), bool)
    for i in rng.permutation(len(recs)):
        k = (recs[i]["peptide"], recs[i]["charge"])
        if seen[k] < MAX_PER_GROUP:
            seen[k] += 1; keep[i] = True
    recs = [r for r, k in zip(recs, keep) if k]
    groups = len({(r["peptide"], r["charge"]) for r in recs})
    print(f"[nine] {n_all:,} yeast test spectra -> {len(recs):,} in {groups:,} groups "
          f"(>= 2 each, capped at {MAX_PER_GROUP})", flush=True)

    prep = {k: [] for k in ("mz", "log_intensity", "peptide", "charge", "precursor",
                            "source", "analyte_id")}
    for r in recs:
        m, li = _process(processor, r["mz"], r["intensity"])
        prep["mz"].append(m); prep["log_intensity"].append(li)
        prep["peptide"].append(r["peptide"]); prep["charge"].append(r["charge"])
        prep["precursor"].append(r["precursor"]); prep["source"].append("experimental")
        prep["analyte_id"].append(f"{r['peptide']}_{r['charge']}")
    Dataset.from_dict(prep).save_to_disk(str(out / "prepared"))
    info = {"source": f"{REPO}@{REVISION}/{FILE}", "species": "S. cerevisiae (test split)",
            "spectra_total": n_all, "spectra": len(recs), "groups": groups,
            "max_peaks_seen": int(max(len(r["mz"]) for r in recs)),
            "max_per_group": MAX_PER_GROUP, "trimmed": False,
            "labels": "published modified_sequence (DB search) + precursor charge"}
    (out / "prepared" / "PREPARED.json").write_text(json.dumps(info, indent=1))

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
    print(f"[nine] wrote {out}/prepared and {out}/export", flush=True)


if __name__ == "__main__":
    main(*sys.argv[1:])
