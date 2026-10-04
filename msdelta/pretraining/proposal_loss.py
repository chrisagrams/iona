"""K195a-P: a proposal for the masked-intensity pretraining loss, reported next to (or trained instead of) today's.

Today's loss (MSDeltaForPreTraining): KL(t || p) over each spectrum's masked peaks, with p = softmax of the intensity
logits and t = the peaks' LINEAR intensity shares, renormalised over the masked peaks; reduction = batchmean (sum over
peaks, mean over spectra). The largest masked peaks dominate it. The proposal tempers the target: t_i proportional to
I_i ** power over the masked peaks (power = 0.5: square-root intensity, the spectral-library-search convention;
power = 1 reproduces today's loss exactly, power -> 0 weighs every masked peak equally).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor


def tempered_intensity_kl(logits: Tensor, labels: Tensor, mask_positions: Tensor, power: float) -> Tensor:
    """KL(t || softmax(logits)) over the masked peaks, t proportional to labels ** power; batchmean like today's loss."""
    selected = mask_positions.bool()
    if not selected.any():
        return logits.new_zeros(())
    log_prob = F.log_softmax(logits.float().masked_fill(~selected, float("-inf")), dim=-1).masked_fill(~selected, 0.0)
    target = labels.float().clamp_min(0.0).masked_fill(~selected, 0.0).pow(power)
    target = target / target.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    return F.kl_div(log_prob, target, reduction="batchmean")
