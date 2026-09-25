"""C-diagnostic inputs (PLAN.md C13 follow-up).

    PYTHONPATH=. .venv/bin/python baselines_wip/nine_diag_build.py SRC_DIR OUT_DIR FIT_DIR

1. OUT_DIR: a ~20k-spectrum subset of the nine-species yeast set built by nine_build.py
   (SRC_DIR/{prepared,export}): whole peptide+charge groups drawn at random (seed 0) until
   N_TARGET spectra, both formats rewritten with rows renumbered 0..n-1 in the same order.
2. FIT_DIR: the ABTT fit set -- N_FIT spectra drawn at random (seed 0) from the TRAIN split
   of InstaDeepAI/ms_ninespecies_benchmark (the 8 OTHER species; peptides largely disjoint
   from yeast), in the prepared format. Nothing from the yeast test set is used to fit.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

N_TARGET = 20000
N_FIT = 25000
REPO = "InstaDeepAI/ms_ninespecies_benchmark"
REVISION = "95d03bc005e512bd20c6abb3f8650e327272973a"
PROCESSOR = "/flare/UIC-HPC/khuss/msdelta/pretrained/msdelta-50m-production-01-checkpoint-220000"


def subset(src: Path, out: Path):
    from datasets import load_from_disk
    meta = pq.read_table(src / "export" / "meta.parquet").to_pandas()
    rng = np.random.default_rng(0)
    groups = meta.groupby("analyte_id").indices
    keys = list(groups); chosen = []
    for j in rng.permutation(len(keys)):
        chosen.extend(groups[keys[j]])
        if len(chosen) >= N_TARGET:
            break
    rows = np.sort(np.asarray(chosen))                     # original order kept
    new_of = {int(r): i for i, r in enumerate(rows)}
    prep = load_from_disk(str(src / "prepared")).select(rows.tolist())
    prep.save_to_disk(str(out / "prepared"))
    info = json.loads((src / "prepared" / "PREPARED.json").read_text())
    info |= {"subset_of": str(src), "spectra": len(rows),
             "groups": int(meta.iloc[rows]["analyte_id"].nunique())}
    (out / "prepared" / "PREPARED.json").write_text(json.dumps(info, indent=1))
    exp = out / "export"; exp.mkdir(parents=True)
    keep = False
    with open(src / "export" / "experimental.mgf") as fin, open(exp / "experimental.mgf", "w") as fout:
        buf = []
        for line in fin:
            if line.startswith("BEGIN IONS"):
                buf = [line]; keep = False
            elif line.startswith("TITLE=row="):
                old = int(line.split(";")[0].split("=")[2])
                keep = old in new_of
                buf.append(line.replace(f"row={old};", f"row={new_of.get(old, -1)};", 1))
            else:
                buf.append(line)
                if line.startswith("END IONS") and keep:
                    fout.writelines(buf)
    (exp / "consensus.mgf").write_text("")
    m = meta.iloc[rows].reset_index(drop=True); m["row"] = np.arange(len(m))
    pq.write_table(pa.Table.from_pandas(m, preserve_index=False), exp / "meta.parquet")
    (exp / "EXPORT.json").write_text(json.dumps(info, indent=1))
    print(f"[diag] subset: {len(rows):,} spectra in {info['groups']:,} groups -> {out}", flush=True)


def fit_set(fit: Path):
    from datasets import Dataset
    from huggingface_hub import hf_hub_download
    from msdelta.grouped_retrieval import _process
    from msdelta.processing_msdelta import MSDeltaProcessor
    import sys as _s; _s.path.insert(0, str(Path(__file__).parent))
    from nine_build import to_notation
    processor = MSDeltaProcessor.from_pretrained(PROCESSOR, max_peaks=512)
    tabs = [pq.read_table(hf_hub_download(REPO, f"data/train-0000{i}-of-00002.parquet",
                                          repo_type="dataset", revision=REVISION))
            for i in range(2)]
    t = pa.concat_tables(tabs)
    idx = np.sort(np.random.default_rng(0).choice(t.num_rows, N_FIT, replace=False))
    rows = t.take(pa.array(idx)).to_pylist()
    prep = {k: [] for k in ("mz", "log_intensity", "peptide", "charge", "precursor",
                            "source", "analyte_id")}
    skipped = 0
    for r in rows:
        mz = np.asarray(r["mz_array"], np.float64); it = np.asarray(r["intensity_array"], np.float64)
        ok = (it > 0) & np.isfinite(mz); mz, it = mz[ok], it[ok]
        if not (0 < len(mz) <= 512):
            skipped += 1; continue
        try:
            pep = to_notation(r["modified_sequence"])
        except SystemExit:
            pep = r["sequence"]                  # the fit ignores labels; keep the spectrum
        m, li = _process(processor, mz, it)
        prep["mz"].append(m); prep["log_intensity"].append(li); prep["peptide"].append(pep)
        prep["charge"].append(int(r["precursor_charge"])); prep["precursor"].append(float(r["precursor_mz"]))
        prep["source"].append("experimental"); prep["analyte_id"].append(f"{pep}_{r['precursor_charge']}")
    Dataset.from_dict(prep).save_to_disk(str(fit))
    (fit / "PREPARED.json").write_text(json.dumps({"source": f"{REPO}@{REVISION} train (8 non-yeast species)",
                                                   "spectra": len(prep["mz"]), "skipped_over_512": skipped}, indent=1))
    print(f"[diag] fit set: {len(prep['mz']):,} train spectra (8 other species), {skipped} over 512 skipped -> {fit}", flush=True)


def main(src, out, fit):
    src, out, fit = Path(src), Path(out), Path(fit)
    if not out.exists():
        subset(src, out)
    if not fit.exists():
        fit_set(fit)


if __name__ == "__main__":
    main(*sys.argv[1:4])
