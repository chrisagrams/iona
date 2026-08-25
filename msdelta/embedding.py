"""Run the encoder over spectra → token / pooled embeddings.

Shared inference plumbing for the frozen-encoder diagnostics (`probe`,
`retrieval`): pad a batch of preprocessed peak lists, forward through the
encoder, and mean⊕max-pool. `MSEncoder.forward` returns per-token embeddings
`(B, K, D)`; pooling to one vector per spectrum lives here (not in the model)
because the probes also consume the raw tokens.
"""
from __future__ import annotations

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from sentence_transformers.sentence_transformer.modules import Pooling



@torch.no_grad()
def encode_batch(enc, mzs, lis, device):
    """Pad one batch of (mz_p, log_int) peak lists and forward through `enc`.

    `mzs`/`lis` are lists of 1-D CPU tensors (empty tensors allowed). Returns
    (tokens (B,K,D), mask (B,K) True=real), both on `device`."""
    mz = pad_sequence(mzs, batch_first=True)               # (B,K), 0-padded
    li = pad_sequence(lis, batch_first=True)
    lens = torch.tensor([m.numel() for m in mzs])
    mask = torch.arange(mz.shape[1])[None, :] < lens[:, None]   # True = real peak
    tokens = enc(mz.to(device), li.to(device), (~mask).to(device))
    return tokens, mask.to(device)


@torch.no_grad()
def embed_spectra(enc, specs, device, *, batch_size=128):
    """One mean⊕max-pooled vector per (mz_p, log_int) spectrum.

    Returns an (n, D) float32 matrix (NOT L2-normalised); empty spectra get a
    zero row so the output stays index-aligned with `specs` (they can never be a
    real neighbour, i.e. a miss)."""
    enc.to(device).eval()
    pooler = Pooling(
        embedding_dimension=enc.cfg.d_model,
        pooling_mode=tuple(enc.cfg.pooling_modes),
    ).to(device)
    n = len(specs)
    emb = None
    for s in range(0, n, batch_size):
        e = min(s + batch_size, n)
        mzs = [m for m, _ in specs[s:e]]
        lis = [l for _, l in specs[s:e]]
        if max((m.numel() for m in mzs), default=0) == 0:
            continue                            # whole batch empty; rows stay zero
        tokens, mask = encode_batch(enc, mzs, lis, device)
        pooled = pooler({
            "token_embeddings": tokens,
            "attention_mask": mask.long(),
        })["sentence_embedding"].float().cpu().numpy().astype(np.float32)
        if emb is None:
            emb = np.zeros((n, pooled.shape[1]), dtype=np.float32)
        nonempty = mask.any(dim=1).cpu().numpy()
        emb[np.arange(s, e)[nonempty]] = pooled[nonempty]
    if emb is None:
        raise RuntimeError("no spectrum produced any peaks after preprocessing")
    return emb
