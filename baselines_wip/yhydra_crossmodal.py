"""Cross-modal spectrum -> peptide retrieval: yHydra vs our alignment student, same queries,
same candidates. PLAN.md A-baselines (yHydra comparison).

    $B/yhydra/env/bin/python yhydra_crossmodal.py DATA_DIR OURS.npz OUT.json

DATA_DIR: the export of ms-contrastive-100k test (export_mgf.py): meta.parquet and
yhydra_embed.npy (yHydra's spectrum tower, rows aligned to meta). OURS.npz:
msdelta.eval_align_test --save-embeddings (teacher spectrum embeddings for the prepared
experimental test rows, in order; student embeddings of every (peptide, charge) candidate).

yHydra's sequence tower takes the 20 standard residues (+BZXJU), length 7-42, no
modifications; carbamidomethyl-C is fixed in its scoring, so C[57.0215] maps to C and any
other modification makes a peptide unrepresentable. The SHARED subset: queries whose
peptide yHydra can represent, candidates = the representable test peptides. On it:
  yhydra   candidates = unique sequences (its tower has no charge input); a hit = the top
           sequence is the query's sequence. Ranked by cosine and by L2.
  ours     candidates = the representable (peptide, charge) pairs, as the student is used;
           reported at sequence level (top candidate's sequence == query's, charge ignored,
           the yHydra-comparable number) and at peptide+charge level (our usual one).
Ours on the FULL set (every query, every candidate) is reported beside it.

All ranking goes through msdelta/eval/filtered_retrieval.py (crossmodal_ranks; loaded by
path, numpy-only, since this env has no torch). The original windows (`window_20ppm`,
`window_1.1Da`: neutral mass, no charge check; the paper's numbers) are kept unchanged, and
the standard filtered report (K77-A) is added under `.../filtered`: open / 20 ppm /
isotope-tolerant 20 ppm on measured precursor m/z vs candidate m/z at the query's charge
(same charge for our peptide+charge candidates; yHydra's sequences have none), on full,
Fbar, F (= F_all), plus net_loss/net_gain and rescue.
"""
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

_spec = importlib.util.spec_from_file_location(
    "filtered_retrieval",
    Path(__file__).resolve().parents[1] / "msdelta" / "eval" / "filtered_retrieval.py")
fr = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(fr)

YH = "/lus/flare/projects/UIC-HPC/khuss/msdelta/baselines/yhydra"
MODEL = f"{YH}/saved_27_06_2021"
PAD = "_"
# EXACTLY yHydra's proteomics_utils.aa_with_pad: pad + pyteomics' std_amino_acids (in
# pyteomics' own order, NOT alphabetical) + non-canonical. Token ids index this list.
# Copied from the env's pyteomics/parser.py:115 (importing it fails on this Python 3.8).
STD_AA = ['Q', 'W', 'E', 'R', 'T', 'Y', 'I', 'P', 'A', 'S',
          'D', 'F', 'G', 'H', 'K', 'L', 'C', 'V', 'N', 'M']
ALPHABET = [PAD] + STD_AA + ["B", "Z", "X", "J", "U"]
MAX_LEN = 42


def to_yhydra(peptide: str):
    """Our notation -> yHydra's plain sequence, or None when it cannot be represented."""
    s = peptide.replace("C[57.0215]", "C")
    if "[" in s or not (7 <= len(s) <= MAX_LEN) or any(a not in ALPHABET[1:] for a in s):
        return None
    return s


MONO = {"G": 57.02146, "A": 71.03711, "S": 87.03203, "P": 97.05276, "V": 99.06841,
        "T": 101.04768, "C": 103.00919, "L": 113.08406, "I": 113.08406, "N": 114.04293,
        "D": 115.02694, "Q": 128.05858, "K": 128.09496, "E": 129.04259, "M": 131.04049,
        "H": 137.05891, "F": 147.06841, "R": 156.10111, "Y": 163.06333, "W": 186.07931}
WATER, PROTON = 18.010565, 1.007276


def peptide_mass(peptide: str) -> float:
    """Neutral monoisotopic mass of a peptide in our notation (`C[57.0215]`, `[42.0106]P`)."""
    import re
    mods = sum(float(x) for x in re.findall(r"\[([-+]?\d+\.?\d*)\]", peptide))
    residues = re.sub(r"\[[^\]]*\]", "", peptide)
    return sum(MONO[a] for a in residues) + mods + WATER


def window_hits(q, c, truth, allowed_fn, metric="cos"):
    """Hit@1 with candidates restricted by `allowed_fn` (fr.crossmodal_filters). A query
    whose true candidate falls outside its own window counts as a miss (reported
    separately as `true_outside`)."""
    rank, inside, sizes = fr.crossmodal_ranks(q, c, truth, allowed_fn, metric)
    return {"hit@1": float((inside & (rank == 0)).mean()), "true_outside": float((~inside).mean()),
            "median_candidates": float(np.median(sizes)), "queries": len(q)}


def topk_hits(queries, cands, truth, metric="cos"):
    """Hit@1/Hit@5/MRR of each query's true candidate among all candidates."""
    rank, _, _ = fr.crossmodal_ranks(queries, cands, truth, None, metric)
    return {"hit@1": float((rank == 0).mean()), "hit@5": float((rank < 5).mean()),
            "mrr": float((1.0 / (rank + 1)).mean()), "queries": len(queries),
            "candidates": len(cands)}


def seq_level_hits(queries, cands, cand_seq, query_seq, chunk=1024):
    """Top candidate's SEQUENCE == query's sequence (charge-collapsed), cosine."""
    q = queries / np.linalg.norm(queries, axis=1, keepdims=True)
    c = cands / np.linalg.norm(cands, axis=1, keepdims=True)
    hit = 0
    for s in range(0, len(q), chunk):
        top = (q[s:s + chunk] @ c.T).argmax(1)
        hit += (cand_seq[top] == query_seq[s:s + chunk]).sum()
    return {"hit@1": hit / len(q), "queries": len(q), "candidates": len(c)}


def main(data_dir, ours_path, out_path):
    import tensorflow as tf
    meta = pq.read_table(f"{data_dir}/meta.parquet").to_pandas()
    rows = meta[meta.in_mp512 & (meta.source == "experimental")].reset_index(drop=True)
    yh_spec = np.load(f"{data_dir}/yhydra_embed.npy")[rows["row"].to_numpy()]
    ours = np.load(ours_path)
    o_spec, o_seq = ours["spectrum"], ours["sequence"]
    grp, cpep, cchg = ours["spectrum_group"], ours["cand_peptide"], ours["cand_charge"]
    # alignment check: our i-th spectrum is the i-th prepared experimental row
    if len(o_spec) != len(rows) or not (cpep[grp] == rows["peptide"].to_numpy()).all():
        raise SystemExit("our embeddings are not aligned with the exported rows")
    if np.isnan(yh_spec).any():
        raise SystemExit("yHydra spectrum embeddings missing for some rows")

    qseq = np.array([to_yhydra(p) for p in rows["peptide"]], dtype=object)
    shared_q = np.array([s is not None for s in qseq])
    cseq = np.array([to_yhydra(p) for p in cpep], dtype=object)
    shared_c = np.array([s is not None for s in cseq])
    uniq = sorted({s for s in cseq[shared_c]})
    index = {s: i for i, s in enumerate(uniq)}
    print(f"[xmodal] queries {len(rows):,} ({shared_q.sum():,} representable); candidates "
          f"{len(cpep):,} (peptide,charge) -> {len(uniq):,} representable sequences", flush=True)

    # yHydra sequence tower
    model = tf.keras.models.load_model(MODEL, custom_objects={"metric_acc": lambda a: a})
    seq_model = tf.keras.Model(inputs=model.get_layer("sequence_input").input,
                               outputs=model.get_layer("seq_emb").output)
    x = np.array([[ALPHABET.index(a) for a in s.ljust(MAX_LEN, PAD)] for s in uniq], np.int32)
    yh_seq = seq_model.predict(x, batch_size=4096).reshape(len(uniq), -1)
    yh_q = yh_spec[shared_q].reshape(shared_q.sum(), -1)
    truth = np.array([index[s] for s in qseq[shared_q]])

    res = {"n_queries_all": int(len(rows)), "n_queries_shared": int(shared_q.sum()),
           "n_candidates_all": int(len(cpep)), "n_sequences_shared": len(uniq)}
    res["yhydra/shared/cos"] = topk_hits(yh_q, yh_seq, truth, "cos")
    res["yhydra/shared/l2"] = topk_hits(yh_q, yh_seq, truth, "l2")

    # ours on the shared subset
    c_idx = np.flatnonzero(shared_c)
    remap = -np.ones(len(cpep), int); remap[c_idx] = np.arange(len(c_idx))
    o_truth = remap[grp[shared_q]]
    res["ours/shared/peptide+charge"] = topk_hits(o_spec[shared_q], o_seq[c_idx], o_truth, "cos")
    res["ours/shared/sequence"] = seq_level_hits(
        o_spec[shared_q], o_seq[c_idx], cseq[c_idx].astype(str), qseq[shared_q].astype(str))
    # precursor-mass windows (yHydra's native setting; real search): both models
    yh_cmass = np.array([peptide_mass(s.replace("C", "C[57.0215]")) for s in uniq])
    o_cmass = np.array([peptide_mass(p) for p in cpep])
    qmz, qz = rows["precursor"].to_numpy(float), rows["charge"].to_numpy(int)
    yh_f = fr.crossmodal_filters(qmz[shared_q], qz[shared_q], yh_cmass)
    o_f = fr.crossmodal_filters(qmz[shared_q], qz[shared_q], o_cmass[c_idx], cchg[c_idx])
    for name in ("20ppm", "1.1Da"):
        res[f"yhydra/shared/l2/window_{name}"] = window_hits(
            yh_q, yh_seq, truth, yh_f[f"legacy_{name}"], "l2")
        res[f"ours/shared/peptide+charge/window_{name}"] = window_hits(
            o_spec[shared_q], o_seq[c_idx], o_truth, o_f[f"legacy_{name}"])
    # ours on everything (our standard number)
    res["ours/full/peptide+charge"] = topk_hits(o_spec, o_seq, grp, "cos")
    # standard filtered report (K77-A); needs a measured precursor for every query
    if fr.measured_precursor(qmz) is None:
        print("[xmodal] no complete measured precursor: filtered reports skipped", flush=True)
    else:
        for metric in ("cos", "l2"):
            res[f"yhydra/shared/{metric}/filtered"] = fr.crossmodal_flatten(fr.crossmodal_report(
                yh_q, yh_seq, truth, qmz[shared_q], qz[shared_q], yh_cmass, metric=metric), "")
        res["ours/shared/peptide+charge/filtered"] = fr.crossmodal_flatten(fr.crossmodal_report(
            o_spec[shared_q], o_seq[c_idx], o_truth, qmz[shared_q], qz[shared_q],
            o_cmass[c_idx], cchg[c_idx]), "")
        res["ours/full/peptide+charge/filtered"] = fr.crossmodal_flatten(fr.crossmodal_report(
            o_spec, o_seq, grp, qmz, qz, o_cmass, cchg), "")
    json.dump(res, open(out_path, "w"), indent=1)
    for k, v in res.items():
        print(k, v if not isinstance(v, dict) else
              {a: (round(b, 4) if isinstance(b, float) else b) for a, b in v.items()}, flush=True)


if __name__ == "__main__":
    main(*sys.argv[1:4])
