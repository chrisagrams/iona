"""Create token and spectrum embeddings."""

from __future__ import annotations

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from tqdm.auto import tqdm


def pool_tokens(tokens: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Apply mean and maximum pooling to real peaks."""
    m = mask.unsqueeze(-1)
    summed = (tokens * m).sum(1)
    mean = summed / m.sum(1).clamp_min(1)
    mx = tokens.masked_fill(~m, float("-inf")).max(1).values
    mx = torch.nan_to_num(mx, neginf=0.0)
    return torch.cat([mean, mx], dim=-1)


@torch.no_grad()
def encode_batch(model, mzs, log_intensities, device):
    """Pad and encode one batch of peak lists."""
    mz = pad_sequence(mzs, batch_first=True)
    log_intensity = pad_sequence(log_intensities, batch_first=True)
    lens = torch.tensor([m.numel() for m in mzs])
    mask = torch.arange(mz.shape[1])[None, :] < lens[:, None]
    attention_mask = mask.to(device)
    outputs = model(
        mz=mz.to(device),
        log_intensity=log_intensity.to(device),
        attention_mask=attention_mask,
    )
    return outputs.last_hidden_state, attention_mask


@torch.no_grad()
def embed_spectra(enc, specs, device, *, batch_size=128, progress_desc: str | None = None):
    """Return one float32 vector for each spectrum."""
    enc.to(device).eval()
    n = len(specs)
    emb = None
    starts = range(0, n, batch_size)
    if progress_desc is not None:
        starts = tqdm(
            starts,
            total=(n + batch_size - 1) // batch_size,
            desc=progress_desc,
            unit="batch",
            leave=False,
        )
    for s in starts:
        e = min(s + batch_size, n)
        mzs = [m for m, _ in specs[s:e]]
        lis = [log_intensity for _, log_intensity in specs[s:e]]
        if max((m.numel() for m in mzs), default=0) == 0:
            continue
        tokens, mask = encode_batch(enc, mzs, lis, device)
        pooled = pool_tokens(tokens, mask).float().cpu().numpy().astype(np.float32)
        if emb is None:
            emb = np.zeros((n, pooled.shape[1]), dtype=np.float32)
        nonempty = mask.any(dim=1).cpu().numpy()
        emb[np.arange(s, e)[nonempty]] = pooled[nonempty]
    if emb is None:
        raise RuntimeError("no spectrum produced any peaks after preprocessing")
    return emb
