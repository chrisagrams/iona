"""Contrastive post-training pieces: view augmentation, projection head, SupCon loss.

The pretrained `MSEncoder` (masked-intensity KL) emits per-peak tokens; a spectrum
embedding is the mean⊕max pool of those tokens (`probe._pool`, 2*d_model). This
module adds the machinery to post-train that encoder so the *raw* cosine geometry
of the pooled embedding is good (no eval-time whitening needed):

  - `augment`         two independently-augmented views of a spectrum (the positive pair)
  - `ProjectionHead`  MLP on the pooled vector, used only for the loss (discarded at eval)
  - `SupConLoss`      supervised-contrastive loss (generalizes InfoNCE / NT-Xent)

The infonce-vs-supcon choice is encoded upstream in how `data.contrastive_collate`
builds the per-row `label` (twin-only vs all-same-peptide), so this loss is a single
label-driven objective.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class AugmentConfig:
    """Stochastic spectrum augmentation applied to already-preprocessed peaks.

    Operates on the streamed (mz, log_int, intensity_prob) views (already
    top-N/threshold/normalized by `data.preprocess_spectrum`), so two draws give
    two correlated-but-different views of the same spectrum.
    """
    peak_dropout_prob: float = 0.2      # per-peak Bernoulli drop
    intensity_jitter_sigma: float = 0.2  # multiply intensity by exp(N(0, sigma))
    subsample_min_frac: float = 0.6     # keep a random fraction in [frac, 1.0] of peaks
    mz_jitter_da: float = 0.0           # Gaussian m/z noise (feeds the Δm bias only)


def augment(
    mz: torch.Tensor,
    log_int: torch.Tensor,
    intensity_prob: torch.Tensor,
    cfg: AugmentConfig,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return one augmented view (mz, log_int, intensity_prob).

    Peak dropout + top-N subsample thin the token set (the dominant augmentation
    for this m/z-free model); intensity jitter perturbs the input feature and the
    KL target; optional m/z jitter perturbs the Δm bias. Always keeps ≥1 peak and
    re-normalizes so log_int max = 1 and intensity_prob sums to 1 (matching
    `preprocess_spectrum`).
    """
    K = int(mz.numel())
    if K == 0:
        return mz, log_int, intensity_prob

    idx = torch.arange(K)

    # Peak dropout.
    if cfg.peak_dropout_prob > 0:
        keep = torch.rand(K) >= cfg.peak_dropout_prob
        idx = idx[keep]

    # Top-N subsample: keep a random count in [ceil(frac*K), #survivors].
    if idx.numel() > 0 and cfg.subsample_min_frac < 1.0:
        lo = max(1, int(math.ceil(cfg.subsample_min_frac * K)))
        hi = int(idx.numel())
        if hi > lo:
            n_keep = int(torch.randint(lo, hi + 1, (1,)))
            sel = torch.randperm(idx.numel())[:n_keep]
            idx = idx[sel]

    # Never drop everything — fall back to the single most intense peak.
    if idx.numel() == 0:
        idx = intensity_prob.argmax().reshape(1)

    idx, _ = torch.sort(idx)
    mz2 = mz[idx]
    li2 = log_int[idx]
    pp2 = intensity_prob[idx]

    # Intensity jitter in the (approx) intensity domain — same factor on both
    # the input feature and the KL target so a view stays self-consistent.
    if cfg.intensity_jitter_sigma > 0:
        factor = (torch.randn(idx.numel()) * cfg.intensity_jitter_sigma).exp()
        li2 = li2 * factor
        pp2 = pp2 * factor

    pp2 = pp2 / pp2.sum().clamp_min(1e-12)
    li2 = li2 / li2.max().clamp_min(1e-8)

    if cfg.mz_jitter_da > 0:
        mz2 = mz2 + torch.randn(idx.numel()) * cfg.mz_jitter_da

    return mz2.contiguous(), li2.contiguous(), pp2.contiguous()


class ProjectionHead(nn.Module):
    """MLP head on the pooled (2*d_model) spectrum vector: Linear-GELU-Linear,
    L2-normalized output. Trained with the contrastive loss and discarded at
    eval (retrieval scores the pooled encoder output directly)."""

    def __init__(self, in_dim: int, hidden: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.net(x)
        return F.normalize(z, dim=-1)


class SupConLoss(nn.Module):
    """Supervised-contrastive loss (Khosla et al.), label-driven.

    Given L2-normalized projections ``z`` (N, D) and integer ``labels`` (N,),
    positives of anchor i are the other rows sharing its label. With one positive
    per anchor this is exactly InfoNCE / NT-Xent; with multiple it's the SupCon
    generalization. The infonce-vs-supcon mode is chosen by how the collate
    assigns labels (twin-only vs all-same-peptide), so nothing branches here.

    Anchors with zero positives are dropped from the mean (no divide-by-zero),
    mirroring the safe masking in `IntensityHead.loss`.
    """

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, z: torch.Tensor, labels: torch.Tensor) -> tuple[torch.Tensor, dict]:
        N = z.size(0)
        device = z.device

        # Cosine similarity logits (z already L2-normalized), diagonal removed.
        sim = (z @ z.t()) / self.temperature
        self_mask = torch.eye(N, dtype=torch.bool, device=device)
        sim = sim.masked_fill(self_mask, float("-inf"))

        # Positive mask: same label, excluding self.
        labels = labels.view(-1, 1)
        pos_mask = (labels == labels.t()) & ~self_mask   # (N, N) bool

        # log-softmax over all non-self entries (the denominator is every other
        # row — positives and negatives alike, standard SupCon "out" form). The
        # self entry is -inf in `sim` (correctly excluded from the denominator);
        # zero it in log_prob so the diagonal's (-inf)*0 in the masked sum below
        # can't produce NaN.
        log_prob = F.log_softmax(sim, dim=1).masked_fill(self_mask, 0.0)

        pos_count = pos_mask.sum(1)                       # |P(i)|
        valid = pos_count > 0                             # anchors with ≥1 positive

        # Mean log-prob over each anchor's positives.
        pos_log_prob = (log_prob * pos_mask).sum(1) / pos_count.clamp_min(1)
        loss = -pos_log_prob[valid].mean() if valid.any() else z.new_zeros(())

        parts = {
            "pos_frac": valid.float().mean().detach(),
            "avg_pos": pos_count.float().mean().detach(),
        }
        return loss, parts
