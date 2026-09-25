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
"""
import json
import sys

import numpy as np
import pyarrow.parquet as pq

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


def topk_hits(queries, cands, truth, metric="cos", chunk=1024):
    """Hit@1/Hit@5/MRR of each query's true candidate among all candidates."""
    q = np.asarray(queries, np.float32); c = np.asarray(cands, np.float32)
    if metric == "cos":
        q = q / np.linalg.norm(q, axis=1, keepdims=True)
        c = c / np.linalg.norm(c, axis=1, keepdims=True)
    h1 = h5 = rr = 0.0
    for s in range(0, len(q), chunk):
        if metric == "cos":
            sim = q[s:s + chunk] @ c.T
        else:       # -||q - c||^2 without materialising (chunk, N, D)
            qq = q[s:s + chunk]
            sim = -((qq ** 2).sum(1)[:, None] + (c ** 2).sum(1)[None] - 2 * qq @ c.T)
        true = sim[np.arange(len(sim)), truth[s:s + chunk]]
        rank = (sim > true[:, None]).sum(1)          # 0 = best
        h1 += (rank == 0).sum(); h5 += (rank < 5).sum(); rr += (1.0 / (rank + 1)).sum()
    n = len(q)
    return {"hit@1": h1 / n, "hit@5": h5 / n, "mrr": rr / n, "queries": n,
            "candidates": len(c)}


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
    # ours on everything (our standard number)
    res["ours/full/peptide+charge"] = topk_hits(o_spec, o_seq, grp, "cos")
    json.dump(res, open(out_path, "w"), indent=1)
    for k, v in res.items():
        print(k, v if not isinstance(v, dict) else
              {a: (round(b, 4) if isinstance(b, float) else b) for a, b in v.items()}, flush=True)


if __name__ == "__main__":
    main(*sys.argv[1:4])
