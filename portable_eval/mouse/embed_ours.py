"""Embed the mouse set with the released iona models (CPU or GPU), for the yHydra comparison.

    HF_TOKEN=... python embed_ours.py MOUSE_DIR OUT.npz \
        [--encoder Gaolaboratory/iona-contrastive-400m] [--student Gaolaboratory/iona-peptide-embedder-400m]

Run from the msdelta repo root (it imports msdelta). Spectra -> the spectrum encoder
(mean+max pooled, L2-normalised), exactly as msdelta.rerank_psm_embed does; candidates =
every distinct (peptide, charge) in the set -> the peptide embedder. Writes the .npz that
compare.py (yHydra comparison) reads: spectrum, sequence, spectrum_group, cand_peptide, cand_charge.
"""
import argparse
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mouse_dir"); ap.add_argument("out")
    ap.add_argument("--encoder", default="Gaolaboratory/iona-contrastive-400m")
    ap.add_argument("--student", default="Gaolaboratory/iona-peptide-embedder-400m")
    ap.add_argument("--pooling", default="mean+max")
    ap.add_argument("--max-peaks", type=int, default=512)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--limit", type=int, default=0, help="only the first N spectra (a quick test)")
    cli = ap.parse_args()

    from huggingface_hub import snapshot_download
    from safetensors.torch import load_file

    from msdelta.modeling_msdelta import MSDeltaForPreTraining
    from msdelta.processing_msdelta import MSDeltaProcessor
    from msdelta.reranking import AlignmentCollator, PeptideCollator, PeptideEncoder, pool_sequence, student_readout

    t0 = time.time()
    xpu = getattr(torch, "xpu", None)
    device = torch.device("cuda" if torch.cuda.is_available() else
                          "xpu" if xpu is not None and xpu.is_available() else "cpu")
    meta = pq.read_table(Path(cli.mouse_dir) / "meta.parquet").to_pandas()
    if cli.limit:
        meta = meta.iloc[:cli.limit]
    local = lambda p: p if Path(p).exists() else snapshot_download(p)
    encoder_dir, student_dir = local(cli.encoder), local(cli.student)
    state = load_file(str(Path(student_dir) / "model.safetensors"))
    width = int(state["sequence_encoder.projection.3.weight"].shape[0])
    processor = MSDeltaProcessor.from_pretrained(encoder_dir, max_peaks=cli.max_peaks)
    encoder = MSDeltaForPreTraining.from_pretrained(encoder_dir).to(device).eval()
    enc = getattr(encoder, "msdelta", encoder)
    student = PeptideEncoder(embedding_size=width, hidden_size=256, num_layers=4, num_heads=8,
                             pooling=cli.pooling, readout=student_readout(state))
    student.load_state_dict({k.removeprefix("sequence_encoder."): v for k, v in state.items()
                             if k.startswith("sequence_encoder.")})
    student = student.to(device).eval()
    print(f"[ours] {len(meta):,} spectra on {device}; models loaded ({time.time() - t0:.0f}s)", flush=True)

    feats = []
    for mz, it in zip(meta["mz"], meta["intensity"]):
        mz = np.asarray(mz, np.float32); it = np.asarray(it, np.float32)
        keep = (it > 0) & np.isfinite(mz); mz, it = mz[keep], it[keep]
        if len(mz) > cli.max_peaks:
            top = np.sort(np.argsort(it)[-cli.max_peaks:]); mz, it = mz[top], it[top]
        v = processor(torch.from_numpy(mz), torch.from_numpy(it), padding=False)
        m, li = v["mz"], v["log_intensity"]
        if m and isinstance(m[0], list):
            m, li = m[0], li[0]
        feats.append({"mz": m, "log_intensity": li, "peptide": "A", "charge": 2})
    collator = AlignmentCollator(pad_spectra_to=cli.max_peaks)
    spec = []
    with torch.no_grad():
        for s in range(0, len(feats), cli.batch_size):
            b = collator(feats[s:s + cli.batch_size]); mask = b["attention_mask"].to(device)
            hidden = enc(mz=b["mz"].to(device), log_intensity=b["log_intensity"].to(device),
                         attention_mask=mask).last_hidden_state
            spec.append(torch.nn.functional.normalize(pool_sequence(hidden, mask, cli.pooling).float(), dim=-1).cpu())
            if (s // cli.batch_size) % 100 == 0:
                print(f"[ours] spectra {s + len(b['mz']):,}/{len(feats):,} ({time.time() - t0:.0f}s)", flush=True)
    spec = torch.cat(spec)

    keys = list(zip(meta["peptide"], meta["charge"].astype(int)))
    slot = {}
    for k in keys:
        slot.setdefault(k, len(slot))
    cands = list(slot)
    pc = PeptideCollator(); seq = []
    with torch.no_grad():
        for s in range(0, len(cands), 512):
            chunk = cands[s:s + 512]
            b = pc([p for p, _ in chunk], [min(max(c, 0), 7) for _, c in chunk])
            seq.append(torch.nn.functional.normalize(student(**{k: v.to(device) for k, v in b.items()}).float(), dim=-1).cpu())
    seq = torch.cat(seq)
    np.savez(cli.out, spectrum=spec.numpy(), sequence=seq.numpy(),
             spectrum_group=np.array([slot[k] for k in keys]),
             cand_peptide=np.array([p for p, _ in cands]), cand_charge=np.array([c for _, c in cands]))
    print(f"[ours] wrote {cli.out}: {len(spec):,} spectra, {len(cands):,} candidates ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
