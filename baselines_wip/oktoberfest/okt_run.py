"""Oktoberfest (Prosit) rescoring of one psm-rerank-hek-hct116 run, all top-10 candidates.

    python okt_run.py --parquet RUN.parquet --out DIR [--limit N] [--model Prosit_2020_intensity_CID]

Oktoberfest reads mzML/raw only, so we (1) write its internal search format
(DIR/msms/msms.prosit, all candidates incl. decoys) which it picks up instead of converting,
and (2) monkeypatch oktoberfest.preprocessing.load_spectra to build its spectra table from the
parquet (MASS_ANALYZER=ITMS, FRAGMENTATION=CID/HCD), with an empty placeholder <run>.mzml so
it finds the run. Then oktoberfest.runner.run_rescoring runs as usual: Koina predictions
(Prosit intensity + Prosit_2019_irt), feature calculation, Percolator on 'original' (search
features) and 'rescore' (+Prosit features).
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

UNIMOD = {"Carbamidomethyl": 4, "Oxidation": 35, "Acetyl": 1}


def modseq(seq, mods):
    res = list(seq); nterm = ""
    for m in mods or []:
        p = int(m["position"]); u = UNIMOD[m["name"]]
        if p == 0:
            nterm = f"[UNIMOD:{u}]-"
        elif p <= len(seq):
            res[p - 1] += f"[UNIMOD:{u}]"
        else:
            raise ValueError(m)
    return nterm + "".join(res)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--model", default="Prosit_2020_intensity_CID")
    ap.add_argument("--frag", default="CID")
    ap.add_argument("--tol", type=float, default=0.35)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--instrument", default="LUMOS")
    a = ap.parse_args()
    out = Path(a.out).resolve(); out.mkdir(parents=True, exist_ok=True)

    cols = ["spectrum_id", "run_id", "scan", "charge", "precursor_neutral_mass", "rt",
            "acquisition_mz_min", "acquisition_mz_max", "mz", "intensity", "candidates"]
    pf = pq.ParquetFile(a.parquet)
    spec_rows, psm_rows = [], []
    for batch in pf.iter_batches(batch_size=2000, columns=cols):
        for r in batch.to_pylist():
            run = r["run_id"]
            spec_rows.append({
                "RAW_FILE": run, "SCAN_NUMBER": int(r["scan"]),
                "INTENSITIES": np.asarray(r["intensity"], dtype=np.float32),
                "MZ": np.asarray(r["mz"], dtype=np.float32),
                "MZ_RANGE": f"{r['acquisition_mz_min']:.1f}-{r['acquisition_mz_max']:.1f}",
                "RETENTION_TIME": float(r["rt"]), "MASS_ANALYZER": "ITMS",
                "FRAGMENTATION": a.frag, "COLLISION_ENERGY": 35.0,
                "INSTRUMENT_TYPES": a.instrument})
            for c in r["candidates"]:
                prots = ";".join(p.split()[0] for p in (c["proteins"] or "").split(";") if p.strip())
                psm_rows.append({
                    "RAW_FILE": run, "SCAN_NUMBER": int(r["scan"]),
                    "MODIFIED_SEQUENCE": modseq(c["sequence"], c["modifications"]),
                    "PRECURSOR_CHARGE": int(r["charge"]), "SCAN_EVENT_NUMBER": int(r["scan"]),
                    "MASS": float(r["precursor_neutral_mass"]),
                    "SCORE": float(c["msfragger_hyperscore"]),
                    "Annotated_Ions_MSF": c["num_matched_ions"], "Total_Ions_MSF": c["tot_num_ions"],
                    "MZ_diff_MSF": c["massdiff"],
                    "EXPECT": 10.0 ** (-float(c["search_neglog10_evalue"])),
                    "NEXT_SCORE": float(c["msfragger_hyperscore"]) - float(c["search_delta_score"] or 0),
                    "REVERSE": bool(c["is_decoy"]), "SEQUENCE": c["sequence"],
                    "PEPTIDE_LENGTH": len(c["sequence"]), "PROTEINS": prots or "UNKNOWN",
                    "candidate_id": c["candidate_id"]})
            if a.limit and len(spec_rows) >= a.limit:
                break
        if a.limit and len(spec_rows) >= a.limit:
            break
    spectra = pd.DataFrame(spec_rows)
    psms = pd.DataFrame(psm_rows)
    # Prosit 2020 (intensity CID/HCD) and Prosit_2019_irt reject N-terminal acetyl; such
    # candidates cannot be scored and are dropped (scored as missing downstream).
    nacet = psms["MODIFIED_SEQUENCE"].str.startswith("[UNIMOD:1]-")
    print(f"dropping {int(nacet.sum())} N-term-acetyl candidates of {len(psms)}", flush=True)
    psms = psms[~nacet].reset_index(drop=True)
    psms.to_csv(out / "all_candidates.csv", index=False)  # bookkeeping for scoring
    (out / "msms").mkdir(exist_ok=True)
    psms.drop(columns=["candidate_id"]).to_csv(out / "msms" / "msms.prosit", index=False)
    sdir = out / "spectra_in"; sdir.mkdir(exist_ok=True)
    (sdir / f"{run}.mzml").write_text("placeholder; spectra injected from parquet\n")
    (out / "search_placeholder.pepXML").write_text("")
    print(f"{run}: {len(spectra)} spectra, {len(psms)} candidates", flush=True)

    cfg = {
        "type": "Rescoring", "tag": "",
        "inputs": {"search_results": str(out / "search_placeholder.pepXML"),
                   "search_results_type": "Msfragger", "spectra": str(sdir),
                   "spectra_type": "mzml"},
        "output": str(out),
        "models": {"intensity": a.model, "irt": "Prosit_2019_irt"},
        "prediction_server": "koina.wilhelmlab.org:443", "ssl": True,
        "numThreads": a.threads, "fdr_estimation_method": "percolator",
        "add_feature_cols": "none", "allFeatures": False, "regressionMethod": "spline",
        "massTolerance": a.tol, "unitMassTolerance": "da", "fragmentation_method": a.frag,
    }
    (out / "config.json").write_text(json.dumps(cfg, indent=1))

    import oktoberfest.preprocessing as pp
    from oktoberfest import runner

    def load_spectra(filenames, *args, **kwargs):
        return spectra.copy()
    pp.load_spectra = load_spectra
    runner.pp.load_spectra = load_spectra
    runner.run_rescoring(out / "config.json")


if __name__ == "__main__":
    main()
