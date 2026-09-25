"""Stage 1 of the PSM-reranking eval: embed spectra and candidates, write per-candidate rows.

    python -m msdelta.rerank_psm_embed --run HEK293/0718-1.parquet --encoder ENC \
        --student RUN/final --cache CACHE --out OUT.parquet

Dataset: Gaolaboratory/psm-rerank-hek-hct116 -- one row per MS2 spectrum with MSFragger's
complete top-10 candidate pool (targets and reversed decoys, unfiltered). See its
README/SPECTRA.md. For every candidate this writes MSFragger's own scores plus
`cosine`: the student's embedding of the candidate peptide against the spectrum encoder's
embedding of the spectrum (the same encoder the student was trained to imitate).
Stage 2 (msdelta.rerank_psm_fdr) turns these rows into PSMs at 1% FDR.

Choices, each forced by the data:
  * peptides are written in our corpus notation, residue then `[mass]` (`C[57.0215]`,
    `M[15.9949]`), a leading `[mass]` for N-terminal mods. Fixed carbamidomethyl is
    explicit in this dataset and in our training data alike.
  * spectra above the encoder's max_peaks (512) keep their 512 most intense peaks. The
    training pipeline DROPS such spectra; a reranker cannot skip a spectrum, so this is
    the least-bad reading, and the row records n_peaks so its effect can be split out.
  * `label` / `is_decoy` are carried for scoring only, never used by anything here.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch

REPO_ID = "Gaolaboratory/psm-rerank-hek-hct116"
# Pinned: the 2026-09-25 00:34 update moved <dataset>/<run>.parquet to spectra/ and added
# features/. Every table so far was built from this revision; keep them all on it.
REVISION = "87f5c2756f5de8da8a664ef7e1a4dac8de067a88"
KEEP = ("msfragger_hyperscore", "search_rank", "search_delta_score",
        "search_neglog10_evalue", "num_matched_ions", "tot_num_ions", "massdiff",
        "num_tol_term", "num_missed_cleavages")


def to_notation(sequence: str, modifications) -> str:
    """Dataset sequence + 1-based modification list -> `C[57.0215]`-style string."""
    add = [0.0] * (len(sequence) + 2)            # 0 = N-term, 1..L residues, L+1 = C-term
    for m in modifications or ():
        add[int(m["position"])] += float(m["mass"])
    out = f"[{add[0]:.4f}]" if add[0] else ""
    for i, aa in enumerate(sequence, start=1):
        out += aa + (f"[{add[i]:.4f}]" if add[i] else "")
    if add[-1]:
        out += f"[{add[-1]:.4f}]"
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, help="a local parquet in the psm-rerank-hek-hct116 "
                    "schema, or a path inside that Hub dataset")
    ap.add_argument("--encoder", required=True,
                    help="spectrum encoder: a local dir or a Hub repo id "
                         "(e.g. Gaolaboratory/iona-contrastive-400m)")
    ap.add_argument("--student", required=True,
                    help="peptide embedder: an alignment run's final/ dir or a Hub repo id "
                         "(e.g. Gaolaboratory/iona-peptide-embedder-400m)")
    ap.add_argument("--cache", default="",
                    help="teacher cache (MANIFEST: pooling, width); optional -- without it "
                         "--pooling is used and the width is read from the student's weights")
    ap.add_argument("--pooling", default="mean+max")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-peaks", type=int, default=512)
    ap.add_argument("--max-spectra", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--vectors-out", default="", help="also save the unit vectors (R5): "
                    "spectrum.npy (spectra x D), peptide.npy (candidates x D), fp16, and "
                    "index.parquet (candidate, owner, null_owner), rows in --out order")
    cli = ap.parse_args(argv)

    import pyarrow as pa
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file

    from msdelta.modeling_msdelta import MSDeltaForPreTraining
    from msdelta.processing_msdelta import MSDeltaProcessor
    from msdelta.reranking import (AlignmentCollator, PeptideCollator, PeptideEncoder,
                                   pool_sequence, student_readout)

    device = torch.device("xpu" if torch.xpu.is_available() else "cpu")
    t0 = time.time()
    # a local parquet in the dataset's schema (a user's own run), or a path inside the Hub
    # dataset (pinned revision)
    path = cli.run if Path(cli.run).exists() else hf_hub_download(
        REPO_ID, cli.run, repo_type="dataset", revision=REVISION)
    table = pq.read_table(path, columns=["spectrum_id", "run_id", "dataset", "charge",
                                         "precursor_mz", "n_peaks", "mz", "intensity",
                                         "candidates"])
    rows = table.to_pylist()
    if cli.max_spectra:
        rows = rows[: cli.max_spectra]
    print(f"[embed] {cli.run}: {len(rows):,} spectra ({time.time() - t0:.0f}s)", flush=True)

    def local(path_or_repo: str) -> str:
        if Path(path_or_repo).exists():
            return path_or_repo
        from huggingface_hub import snapshot_download
        return snapshot_download(path_or_repo)

    encoder_dir, student_dir = local(cli.encoder), local(cli.student)
    state = load_file(str(Path(student_dir) / "model.safetensors"))
    if cli.cache:
        manifest = dict(l.split(": ", 1) for l in
                        (Path(cli.cache) / "MANIFEST.txt").read_text().splitlines() if ": " in l)
        pooling, width = manifest["pooling"], int(manifest["embedding_size"])
    else:
        pooling = cli.pooling
        width = int(state["sequence_encoder.projection.3.weight"].shape[0])
    processor = MSDeltaProcessor.from_pretrained(encoder_dir, max_peaks=cli.max_peaks)
    encoder = MSDeltaForPreTraining.from_pretrained(encoder_dir).to(device).eval()
    enc = getattr(encoder, "msdelta", encoder)
    student = PeptideEncoder(embedding_size=width, hidden_size=256,
                             num_layers=4, num_heads=8, pooling=pooling,
                             readout=student_readout(state))
    student.load_state_dict({k.removeprefix("sequence_encoder."): v for k, v in state.items()
                             if k.startswith("sequence_encoder.")})
    student = student.to(device).eval()
    collator = AlignmentCollator(pad_spectra_to=cli.max_peaks)

    # --- spectra -> unit embeddings ---------------------------------------------------
    feats = []
    for r in rows:
        mz = np.asarray(r["mz"], dtype=np.float32); it = np.asarray(r["intensity"], dtype=np.float32)
        keep = (it > 0) & np.isfinite(mz)
        mz, it = mz[keep], it[keep]
        if len(mz) > cli.max_peaks:
            top = np.sort(np.argsort(it)[-cli.max_peaks:])
            mz, it = mz[top], it[top]
        if len(mz) == 0:
            mz, it = np.array([100.0], np.float32), np.array([1.0], np.float32)
        v = processor(torch.from_numpy(mz), torch.from_numpy(it), padding=False)
        m, li = v["mz"], v["log_intensity"]
        if m and isinstance(m[0], list):
            m, li = m[0], li[0]
        feats.append({"mz": m, "log_intensity": li, "peptide": "A", "charge": 2})
    spec = []
    with torch.no_grad():
        for s in range(0, len(feats), cli.batch_size):
            b = collator(feats[s:s + cli.batch_size])
            mask = b["attention_mask"].to(device)
            hidden = enc(mz=b["mz"].to(device), log_intensity=b["log_intensity"].to(device),
                         attention_mask=mask).last_hidden_state
            spec.append(torch.nn.functional.normalize(
                pool_sequence(hidden, mask, pooling).float(), dim=-1).cpu())
    spec = torch.cat(spec)
    print(f"[embed] spectra embedded ({time.time() - t0:.0f}s)", flush=True)

    # --- candidates -> cosine to their spectrum ------------------------------------------
    out = {k: [] for k in ("spectrum_id", "run_id", "dataset", "charge", "n_peaks",
                           "candidate", "peptide", "sequence", "is_decoy", "label",
                           "length", "cosine", *KEEP)}
    peptides, charges, owner = [], [], []
    for i, r in enumerate(rows):
        for c in r["candidates"]:
            pep = to_notation(c["sequence"], c["modifications"])
            peptides.append(pep); charges.append(int(r["charge"])); owner.append(i)
            out["spectrum_id"].append(r["spectrum_id"]); out["run_id"].append(r["run_id"])
            out["dataset"].append(r["dataset"]); out["charge"].append(int(r["charge"]))
            out["n_peaks"].append(int(r["n_peaks"])); out["candidate"].append(c["candidate_id"])
            out["peptide"].append(pep); out["sequence"].append(c["sequence"])
            out["is_decoy"].append(bool(c["is_decoy"])); out["label"].append(int(c["label"]))
            out["length"].append(len(c["sequence"]))
            for k in KEEP:
                out[k].append(c.get(k))
    pc = PeptideCollator()
    cos, null, pep_vecs = [], [], []
    perm = np.random.default_rng(0).permutation(len(rows))
    perm = np.where(perm == np.arange(len(rows)), np.roll(perm, 1), perm)   # never self
    with torch.no_grad():
        for s in range(0, len(peptides), 512):
            b = pc(peptides[s:s + 512], [min(max(z, 0), 7) for z in charges[s:s + 512]])
            emb = student(**{k: v.to(device) for k, v in b.items()}).float().cpu()
            own = spec[torch.tensor(owner[s:s + 512])]
            cos.append((emb * own).sum(-1))
            if cli.vectors_out:
                pep_vecs.append(emb.half().numpy())
            null.append((emb * spec[torch.tensor(perm[owner[s:s + 512]])]).sum(-1))
    out["cosine"] = torch.cat(cos).tolist()
    # LEAKAGE CHECK: the same candidate against a RANDOM other spectrum of the run. A
    # student that learned sequence-only "decoy-ness" would still separate targets from
    # decoys on this column; a clean one gives AUROC ~0.5 (rerank_psm_fdr reports it).
    out["cosine_null"] = torch.cat(null).tolist()
    Path(cli.out).parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.table(out), cli.out)
    if cli.vectors_out:
        vd = Path(cli.vectors_out); vd.mkdir(parents=True, exist_ok=True)
        np.save(vd / "spectrum.npy", spec.half().numpy())
        np.save(vd / "peptide.npy", np.concatenate(pep_vecs))
        pq.write_table(pa.table({"candidate": out["candidate"],
                                 "owner": np.asarray(owner, dtype=np.int64),
                                 "null_owner": perm[np.asarray(owner)].astype(np.int64)}),
                       vd / "index.parquet")
        print(f"[embed] vectors -> {vd}", flush=True)
    print(f"[embed] wrote {len(peptides):,} candidates to {cli.out} "
          f"({time.time() - t0:.0f}s)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
