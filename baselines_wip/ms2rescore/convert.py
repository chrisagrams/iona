"""Convert one psm-rerank-hek-hct116 parquet run to MS2Rescore inputs.

    python convert.py --parquet RUN.parquet --out DIR [--limit N]

Writes DIR/<run_id>.mgf (TITLE = spectrum_id, RTINSECONDS, PEPMASS, CHARGE) and
DIR/psms.tsv in psm_utils 'tsv' format with ALL top-10 MSFragger candidates per spectrum
(targets and decoys), score = -log10(e-value), plus MSFragger PIN-like rescoring features.
Also writes DIR/candidates.tsv: candidate bookkeeping (spectrum_id, peptidoform, is_decoy,
search_rank, neglog10_evalue) to map results back.

--emb ROWS.parquet --emb-set SET adds OUR embedding features as extra rescoring:* columns,
joined on candidate_id from msdelta.rerank_psm_embed's per-candidate rows (nothing else of
ours is added, so the A/B isolates the embedding):
  cos      emb_cos: student(candidate) . encoder(spectrum)
  cosws    emb_cos + within-spectrum versions over the candidate's own pool:
           emb_cos_delta (minus the best OTHER candidate), emb_cos_rank (1 = best),
           emb_cos_z (standardised within the pool), emb_cos_gap12 (best - second best)
  null     the same five, computed from cosine_null (candidate vs a RANDOM other spectrum):
           the leakage control -- it must add nothing

--labfeat LAB.parquet adds the lab's feature table for this run (features/<ds>/<run>.parquet,
joined on candidate_id) as rescoring:lab_* columns: every feat__ column that is not empty and
not constant in this run; missing values -> 0 for mod_count_* columns, the run median otherwise
(the same rule as msdelta.rerank_psm_fdr.load_lab_features). Combines with --emb.
"""
import argparse
import csv
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq


def proforma(seq, mods, charge):
    res = list(seq)
    nterm, cterm = "", ""
    for m in mods or []:
        p, name = int(m["position"]), m["name"]
        if p == 0:
            nterm += f"[{name}]"
        elif p == len(seq) + 1:
            cterm += f"[{name}]"
        else:
            res[p - 1] += f"[{name}]"
    s = "".join(res)
    if nterm:
        s = nterm + "-" + s
    if cterm:
        s = s + "-" + cterm
    return f"{s}/{charge}"


EMB_FEATS = {"cos": ["emb_cos"],
             "cosws": ["emb_cos", "emb_cos_delta", "emb_cos_rank", "emb_cos_z", "emb_cos_gap12"]}
EMB_FEATS["null"] = EMB_FEATS["cosws"]


def pool_features(cos):
    """Within-spectrum features for one candidate pool; cos: (n,) array -> list of rows."""
    n = len(cos)
    order = np.argsort(-cos, kind="stable")
    rank = np.empty(n, dtype=int); rank[order] = np.arange(1, n + 1)
    best, second = cos[order[0]], (cos[order[1]] if n > 1 else cos[order[0]])
    best_other = np.where(rank == 1, second, best)        # the best candidate that is not me
    sd = cos.std()
    z = (cos - cos.mean()) / sd if sd > 0 else np.zeros(n)
    return [[float(cos[i]), float(cos[i] - best_other[i]), int(rank[i]), float(z[i]),
             float(best - second)] for i in range(n)]


FEATS = ["hyperscore", "delta_score", "neglog10_evalue", "matched_ion_frac",
         "num_matched_ions", "tot_num_ions", "massdiff", "abs_massdiff", "num_tol_term",
         "num_missed_cleavages"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--emb", default="", help="rerank_psm_embed rows for this run")
    ap.add_argument("--emb-set", default="cos", choices=sorted(EMB_FEATS))
    ap.add_argument("--labfeat", default="", help="the lab's features/<ds>/<run>.parquet")
    a = ap.parse_args()
    lab_cols, lab = [], {}
    if a.labfeat:
        import pandas as pd
        names = pq.ParquetFile(a.labfeat).schema.names
        lf = pq.read_table(a.labfeat, columns=["candidate_id"] + [c for c in names if c.startswith("feat__")]).to_pandas()
        feat = [c for c in lf.columns if c.startswith("feat__")]
        feat = [c for c in feat if lf[c].notna().any() and lf[c].nunique(dropna=True) > 1]
        for c in feat:
            lf[c] = pd.to_numeric(lf[c], errors="coerce")
            lf[c] = lf[c].fillna(0.0 if "mod_count_" in c else lf[c].median())
        lab_cols = ["lab_" + c[len("feat__"):] for c in feat]
        lab = dict(zip(lf["candidate_id"], lf[feat].to_numpy(dtype=float).tolist()))
        print(f"lab features: {len(lab_cols)} columns for {len(lab):,} candidates from {a.labfeat}")
    emb_cols = []
    cosine = {}
    if a.emb:
        col = "cosine_null" if a.emb_set == "null" else "cosine"
        t = pq.read_table(a.emb, columns=["candidate", col])
        cosine = dict(zip(t.column("candidate").to_pylist(), t.column(col).to_pylist()))
        emb_cols = EMB_FEATS[a.emb_set]
        print(f"embedding: {len(cosine):,} candidates from {a.emb} ({col}, set {a.emb_set})")
    n_missing = 0
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    pf = pq.ParquetFile(a.parquet)
    cols = ["spectrum_id", "run_id", "charge", "precursor_mz", "rt", "mz", "intensity",
            "candidates"]
    n_spec = n_psm = 0
    run = None
    mgf = None
    with open(out / "psms.tsv", "w", newline="") as ft:
        w = csv.writer(ft, delimiter="\t", lineterminator="\n")
        w.writerow(["spectrum_id", "run", "peptidoform", "is_decoy", "score", "precursor_mz",
                    "retention_time", "protein_list", "rank"] + [f"rescoring:{f}" for f in FEATS]
                   + [f"rescoring:{f}" for f in emb_cols] + [f"rescoring:{f}" for f in lab_cols])
        fc = open(out / "candidates.tsv", "w", newline="")
        wc = csv.writer(fc, delimiter="\t", lineterminator="\n")
        wc.writerow(["candidate_id", "spectrum_id", "peptidoform", "search_rank", "is_decoy"])
        for batch in pf.iter_batches(batch_size=2000, columns=cols):
            for r in batch.to_pylist():
                if run is None:
                    run = r["run_id"]
                    mgf = open(out / f"{run}.mgf", "w")
                assert r["run_id"] == run
                z = int(r["charge"])
                mgf.write(f"BEGIN IONS\nTITLE={r['spectrum_id']}\nRTINSECONDS={r['rt'] * 60:.4f}\n"
                          f"PEPMASS={r['precursor_mz']:.6f}\nCHARGE={z}+\n")
                mgf.write("".join(f"{m:.5f} {i:.2f}\n" for m, i in zip(r["mz"], r["intensity"])))
                mgf.write("END IONS\n")
                extra = [[] for _ in r["candidates"]]
                if emb_cols:
                    got = [cosine.get(c["candidate_id"]) for c in r["candidates"]]
                    if any(v is None for v in got):
                        # every candidate of this dataset was embedded; a miss means the
                        # rows belong to another run or version -- refuse, never impute
                        n_missing += sum(v is None for v in got)
                        raise SystemExit(f"{r['spectrum_id']}: {n_missing} candidates "
                                         f"missing from {a.emb}")
                    feats = pool_features(np.asarray(got, dtype=np.float64))
                    extra = [f[:len(emb_cols)] for f in feats]
                if lab_cols:
                    miss = [c["candidate_id"] for c in r["candidates"] if c["candidate_id"] not in lab]
                    if miss:
                        raise SystemExit(f"{r['spectrum_id']}: {len(miss)} candidates missing from {a.labfeat}")
                for c, e in zip(r["candidates"], extra):
                    if lab_cols:
                        e = list(e) + lab[c["candidate_id"]]
                    pform = proforma(c["sequence"], c["modifications"], z)
                    wc.writerow([c["candidate_id"], r["spectrum_id"], pform, c["search_rank"], bool(c["is_decoy"])])
                    prots = [p.split()[0] for p in (c["proteins"] or "").split(";") if p.strip()]
                    tot = c["tot_num_ions"] or 0
                    f = [c["msfragger_hyperscore"], c["search_delta_score"],
                         c["search_neglog10_evalue"],
                         (c["num_matched_ions"] or 0) / tot if tot else 0.0,
                         c["num_matched_ions"], c["tot_num_ions"], c["massdiff"],
                         abs(c["massdiff"]) if c["massdiff"] is not None else None,
                         c["num_tol_term"], c["num_missed_cleavages"]]
                    w.writerow([r["spectrum_id"], run, pform,
                                bool(c["is_decoy"]), c["search_neglog10_evalue"],
                                r["precursor_mz"], r["rt"], repr(prots), c["search_rank"]]
                               + ["" if v is None else v for v in f] + e)
                    n_psm += 1
                n_spec += 1
                if a.limit and n_spec >= a.limit:
                    break
            if a.limit and n_spec >= a.limit:
                break
    mgf.close(); fc.close()
    print(f"{run}: {n_spec} spectra, {n_psm} PSMs -> {out}")


if __name__ == "__main__":
    main()
