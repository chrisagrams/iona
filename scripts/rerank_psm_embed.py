"""Stage 1 of the PSM-reranking eval: embed spectra and candidates, write per-candidate rows.

    python scripts/rerank_psm_embed.py --run HEK293/0718-1.parquet --encoder ENC \
        --student RUN/final --cache CACHE --out OUT.parquet

For every candidate, writes MSFragger's scores plus `cosine`, the student's peptide
embedding against the spectrum embedding. Spectra above max_peaks keep their most intense
peaks. Stage 2 (scripts/rerank_psm_fdr.py) turns these rows into PSMs at 1% FDR.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from datasets import Dataset
from huggingface_hub import hf_hub_download, snapshot_download
from safetensors.torch import load_file

from iona.data import map_length_sorted
from iona.modeling_iona import IonaForPreTraining
from iona.processing_iona import IonaProcessor
from iona.reranking import AlignmentCollator, PeptideCollator, PeptideEncoder, embed_spectrum

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
    ap.add_argument("--run", required=True, help="local parquet or path in the Hub dataset")
    ap.add_argument("--repo", help="Hub dataset repo to download --run from")
    ap.add_argument("--revision", default=None, help="dataset revision to pin")
    ap.add_argument("--encoder", required=True,
                    help="spectrum encoder dir or Hub repo id")
    ap.add_argument("--student", required=True,
                    help="peptide embedder dir or Hub repo id")
    ap.add_argument("--cache", default="",
                    help="optional teacher cache dir")
    ap.add_argument("--pooling", default="mean+max")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-peaks", type=int, default=512)
    ap.add_argument("--max-spectra", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=16)
    cli = ap.parse_args(argv)

    device = torch.device("xpu" if torch.xpu.is_available() else "cpu")
    t0 = time.time()
    if not Path(cli.run).exists() and not cli.repo:
        ap.error("--run is not a local file; pass --repo to download it")
    path = cli.run if Path(cli.run).exists() else hf_hub_download(
        cli.repo, cli.run, repo_type="dataset", revision=cli.revision)
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
    processor = IonaProcessor.from_pretrained(encoder_dir, max_peaks=cli.max_peaks)
    encoder = IonaForPreTraining.from_pretrained(encoder_dir).to(device).eval()
    student = PeptideEncoder(embedding_size=width, hidden_size=256,
                             num_layers=4, num_heads=8, pooling=pooling)
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
    def embed_batch(chunk):
        b = collator(chunk)
        return {"spec": embed_spectrum(encoder, b["mz"].to(device), b["log_intensity"].to(device),
                                       b["attention_mask"].to(device), pooling).float().cpu().numpy()}

    with torch.no_grad():
        spec = map_length_sorted(Dataset.from_list(feats), embed_batch, cli.batch_size)
    spec = torch.from_numpy(np.stack(spec["spec"]))
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
    cos = []
    with torch.no_grad():
        for s in range(0, len(peptides), 512):
            b = pc(peptides[s:s + 512], [min(max(z, 0), 7) for z in charges[s:s + 512]])
            emb = student(**{k: v.to(device) for k, v in b.items()}).float().cpu()
            own = spec[torch.tensor(owner[s:s + 512])]
            cos.append((emb * own).sum(-1))
    out["cosine"] = torch.cat(cos).tolist()
    Path(cli.out).parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.table(out), cli.out)
    print(f"[embed] wrote {len(peptides):,} candidates to {cli.out} "
          f"({time.time() - t0:.0f}s)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
