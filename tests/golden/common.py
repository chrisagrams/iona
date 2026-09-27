"""What the golden tests freeze, and the one function that computes it.

Both regenerate.py (writes the references) and test_golden.py (recomputes and compares)
call `compute()`, so the reference and the check can never drift apart in how they
embed, pool or score.

Frozen inputs
  spectra   rows 0..199 of the prepared ms-contrastive-100k validation split
            (eval-data/ms-contrastive-100k-validation-mp512, written by
            `eval_grouped_retrieval prepare`): 50 analytes x (consensus + 3 replicates)
  peptides  the distinct (peptide, charge) pairs of those rows, first 50

Frozen models (read-only)
  pretrained25m    the 25M pretrained checkpoint at step 540,423
  contrastive50m   a 50M contrastive run's final/ (sweep-cont050m_ep01_seed1)
  peptide400m      the Hub release Gaolaboratory/iona-peptide-embedder-400m, from the
                   local HF cache (HF_HUB_OFFLINE=1)

Outputs
  <model>/embeddings   pooled mean+max spectrum embeddings (unit norm), 200 x 2*hidden
  <model>/retrieval    grouped-retrieval metrics on the 200 rows, `all` and `experimental`
                       (the same _variants the eval entry point reports)
  pretrained25m/head   the pretraining head's per-peak log-softmax over valid peaks,
                       unmasked input, first HEAD_ROWS spectra, concatenated (ragged)
  peptide400m/embeddings  50 x 2560 peptide embeddings (unit norm)
"""

from __future__ import annotations

import contextlib
import hashlib
import os
from pathlib import Path

import numpy as np

FLARE = Path("/lus/flare/projects/UIC-HPC/khuss/msdelta")
EVAL_DATA = FLARE / "eval-data" / "ms-contrastive-100k-validation-mp512"
SPECTRUM_MODELS = {
    "pretrained25m": Path("/flare/UIC-HPC/khuss/msdelta/pretrained/"
                          "msdelta-25m-production-01-checkpoint-540423"),
    "contrastive50m": FLARE / "runs" / "sweep-cont050m_ep01_seed1-8860522" / "final",
}
PEPTIDE_MODEL = "Gaolaboratory/iona-peptide-embedder-400m"
N_ROWS = 200
N_PEPTIDES = 50
HEAD_ROWS = 32
BATCH_SIZE = 4
POOLING = "mean+max"
REFERENCE = Path(__file__).resolve().parent / "reference"


def missing_inputs() -> list[str]:
    """Paths the golden tests need and cannot read here (empty = all present)."""
    missing = [str(p) for p in (EVAL_DATA, *SPECTRUM_MODELS.values())
               if not os.access(p, os.R_OK)]
    missing += [str(p / "model.safetensors") for p in SPECTRUM_MODELS.values()
                if not os.access(p / "model.safetensors", os.R_OK)]
    try:
        peptide_model_dir()
    except Exception as error:                      # not in the local HF cache
        missing.append(f"{PEPTIDE_MODEL} ({type(error).__name__})")
    return missing


def peptide_model_dir() -> Path:
    from huggingface_hub import snapshot_download
    return Path(snapshot_download(PEPTIDE_MODEL, local_files_only=True))


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 24), b""):
            h.update(block)
    return h.hexdigest()


def weight_hashes() -> dict[str, str]:
    out = {name: sha256(p / "model.safetensors") for name, p in SPECTRUM_MODELS.items()}
    weights = sorted(peptide_model_dir().glob("*.safetensors"))
    out["peptide400m"] = {w.name: sha256(w) for w in weights}
    return out


def inputs():
    """(rows, peptides, charges): the frozen 200 spectra and 50 peptides."""
    from datasets import load_from_disk
    rows = load_from_disk(str(EVAL_DATA)).select(range(N_ROWS))
    seen = {}
    for p, c in zip(rows["peptide"], rows["charge"]):
        seen.setdefault((p, int(c)), None)
    pairs = list(seen)[:N_PEPTIDES]
    return rows, [p for p, _ in pairs], [c for _, c in pairs]


def _autocast(device: str, dtype: str):
    import torch
    if dtype == "fp32":
        return contextlib.nullcontext()
    return torch.autocast(device_type=device, dtype=torch.bfloat16)


def compute(device: str = "cpu", dtype: str = "fp32", log=print) -> dict[str, np.ndarray]:
    """Every golden output, as float32/float64 numpy arrays keyed `<model>/<output>`."""
    import torch

    from msdelta.eval.eval_grouped_retrieval import _variants
    from msdelta.data.grouped_retrieval import group_ids
    from msdelta.finetuning.contrastive.contrastive import (MSDeltaForContrastive, embed_dataset,
                                                            retrieval_metrics_topk)
    from msdelta.finetuning.contrastive.finetune_contrastive import ContrastiveCollator
    from msdelta.models.modeling_msdelta import MSDeltaForPreTraining
    from msdelta.models.peptide_embedder import PeptideEmbedderModel

    torch.manual_seed(0)
    rows, peptides, charges = inputs()
    groups = group_ids(rows)
    experimental = np.array([s == "experimental" for s in rows["source"]])
    collator = ContrastiveCollator(max_peptide_length=64, pad_spectra_to=0)
    dev = torch.device(device)
    out: dict[str, np.ndarray] = {}
    for name, path in SPECTRUM_MODELS.items():
        log(f"[golden] {name} on {device}/{dtype}")
        encoder = MSDeltaForPreTraining.from_pretrained(str(path)).eval()
        model = MSDeltaForContrastive(encoder, None, pooling=POOLING, kl_weight=0).to(dev).eval()
        with _autocast(device, dtype):
            emb, _ = embed_dataset(model, rows, collator, dev, max_rows=N_ROWS,
                                   batch_size=BATCH_SIZE)
        emb = emb.float()
        out[f"{name}/embeddings"] = emb.numpy()
        metrics = _variants(emb, groups, experimental, dev, retrieval_metrics_topk)
        keys = sorted(metrics)
        out[f"{name}/retrieval_keys"] = np.array(keys)
        out[f"{name}/retrieval"] = np.array([float(metrics[k]) for k in keys])
        if name == "pretrained25m":
            logprobs, lengths = [], []
            with torch.no_grad(), _autocast(device, dtype):
                for start in range(0, HEAD_ROWS, BATCH_SIZE):
                    batch = collator([rows[i] for i in range(start, start + BATCH_SIZE)])
                    batch = {k: batch[k].to(dev) for k in ("mz", "log_intensity",
                                                           "attention_mask")}
                    logits = encoder(**batch).logits.float()
                    mask = batch["attention_mask"].bool()
                    lp = torch.log_softmax(logits.masked_fill(~mask, float("-inf")), dim=-1)
                    for b in range(lp.shape[0]):
                        n = int(mask[b].sum())
                        logprobs.append(lp[b, :n].cpu().numpy())
                        lengths.append(n)
            out[f"{name}/head_logprob"] = np.concatenate(logprobs)
            out[f"{name}/head_lengths"] = np.array(lengths)
        del model, encoder
    log(f"[golden] peptide400m on {device}/{dtype}")
    embedder = PeptideEmbedderModel.from_pretrained(str(peptide_model_dir())).to(dev).eval()
    with _autocast(device, dtype):
        pep = embedder.embed(peptides, charges, batch_size=16)
    out["peptide400m/embeddings"] = pep.float().numpy()
    return out
