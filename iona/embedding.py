"""Create token and spectrum embeddings."""

from __future__ import annotations

import numpy as np
import torch
from datasets import Dataset
from torch.nn.utils.rnn import pad_sequence

from iona.data import map_length_sorted


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
def embed_spectra(enc, specs, device, *, batch_size=128):
    """Return one float32 vector for each spectrum; spectra without peaks embed as zeros."""
    enc.to(device).eval()
    rows = Dataset.from_dict({
        "mz": [m.tolist() for m, _ in specs],
        "log_intensity": [log_intensity.tolist() for _, log_intensity in specs],
    })

    def forward(batch):
        mzs = [torch.tensor(r["mz"], dtype=torch.float32) for r in batch]
        if max(m.numel() for m in mzs) == 0:
            return {"emb": [None] * len(batch)}
        lis = [torch.tensor(r["log_intensity"], dtype=torch.float32) for r in batch]
        tokens, mask = encode_batch(enc, mzs, lis, device)
        pooled = pool_tokens(tokens, mask).float().cpu().numpy()
        nonempty = mask.any(dim=1).cpu().numpy()
        return {"emb": [p if keep else None for p, keep in zip(pooled, nonempty)]}

    vectors = list(map_length_sorted(rows, forward, batch_size)["emb"]) if len(rows) else []
    width = next((len(v) for v in vectors if v is not None), None)
    if width is None:
        raise RuntimeError("no spectrum produced any peaks after preprocessing")
    return np.stack([np.zeros(width) if v is None else v for v in vectors]).astype(np.float32)
