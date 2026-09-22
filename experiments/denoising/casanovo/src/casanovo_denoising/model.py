"""Frozen Casanovo spectrum encoder with a peak-level noise classifier."""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F
from casanovo.denovo.model import Spec2Pep
from torch import Tensor, nn

HEAD_INITIALIZER_RANGE = 0.02


class PeakDenoisingHead(nn.Module):
    """Predict one noise logit per encoded peak (same shape as MSDelta's head)."""

    def __init__(self, input_size: int, hidden_size: int = 128, dropout: float = 0.1):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1),
        )
        for module in self.modules():
            if isinstance(module, nn.Linear):
                module.weight.data.normal_(mean=0.0, std=HEAD_INITIALIZER_RANGE)
                module.bias.data.zero_()

    def forward(self, hidden_states: Tensor) -> Tensor:
        return self.projection(hidden_states).squeeze(-1).float()


class CasanovoDenoiser(nn.Module):
    """Classify peaks as noise from frozen Casanovo encoder states."""

    def __init__(
        self,
        encoder: nn.Module,
        hidden_size: int,
        head_hidden_size: int = 128,
        head_dropout: float = 0.1,
    ):
        super().__init__()
        self.encoder = encoder
        self.encoder.requires_grad_(False)
        self.encoder.eval()
        self.head = PeakDenoisingHead(hidden_size, head_hidden_size, head_dropout)

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint: str | Path,
        head_hidden_size: int = 128,
        head_dropout: float = 0.1,
    ) -> CasanovoDenoiser:
        spec2pep = Spec2Pep.load_from_checkpoint(str(checkpoint), map_location="cpu")
        hidden_size = int(spec2pep.hparams.get("dim_model", 512))
        encoder = spec2pep.encoder
        del spec2pep
        return cls(encoder, hidden_size, head_hidden_size, head_dropout)

    def train(self, mode: bool = True):
        super().train(mode)
        self.encoder.eval()
        return self

    def forward(self, mz: Tensor, intensity: Tensor, labels: Tensor | None = None):
        """Return ``(loss, logits, valid)``; ``loss`` is ``None`` without labels."""
        with torch.no_grad():
            memory, memory_padding_mask = self.encoder(mz, intensity)
        # Position 0 is Casanovo's global spectrum token, not a peak.
        peak_hidden = memory[:, 1:, :]
        peak_padding_mask = memory_padding_mask[:, 1:]
        logits = self.head(peak_hidden)

        if labels is None:
            return None, logits, ~peak_padding_mask
        assert peak_hidden.shape[1] == labels.shape[1], (
            f"encoder produced {peak_hidden.shape[1]} peak states for {labels.shape[1]} labels"
        )
        valid = (labels != -100) & ~peak_padding_mask
        loss = (
            F.binary_cross_entropy_with_logits(logits[valid], labels[valid].float())
            if valid.any()
            else logits.sum() * 0.0
        )
        return loss, logits, valid


def count_parameters(module: nn.Module, trainable_only: bool = False) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad or not trainable_only)
