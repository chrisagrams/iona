"""Create token and spectrum embeddings."""

from __future__ import annotations

import numpy as np
import torch
from datasets import Dataset

from iona.inference import PredictionTrainer, collate_spectra


def pool_tokens(tokens: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Apply mean and maximum pooling to real peaks."""
    m = mask.unsqueeze(-1)
    summed = (tokens * m).sum(1)
    mean = summed / m.sum(1).clamp_min(1)
    mx = tokens.masked_fill(~m, float("-inf")).max(1).values
    mx = torch.nan_to_num(mx, neginf=0.0)
    return torch.cat([mean, mx], dim=-1)


def embed_spectra(enc, specs, device, *, batch_size=128):
    """Return one float32 vector for each spectrum; spectra without peaks embed as zeros."""
    enc.to(device).eval()
    nonempty = np.flatnonzero([m.numel() for m, _ in specs])
    if not len(nonempty):
        raise RuntimeError("no spectrum produced any peaks after preprocessing")
    rows = Dataset.from_dict({
        "mz": [specs[i][0].tolist() for i in nonempty],
        "log_intensity": [specs[i][1].tolist() for i in nonempty],
    })

    def predict(model, inputs):
        tokens = model(**inputs).last_hidden_state
        return pool_tokens(tokens, inputs["attention_mask"]).float()

    trainer = PredictionTrainer(
        enc, predict, data_collator=collate_spectra, batch_size=batch_size, device=device
    )
    pooled = trainer.predict_sorted(rows)
    emb = np.zeros((len(specs), pooled.shape[1]), dtype=np.float32)
    emb[nonempty] = pooled
    return emb
