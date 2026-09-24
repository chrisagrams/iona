"""InstaNovo-FM spectrum encoder with a peak-level noise classifier."""

from __future__ import annotations

import contextlib
import json
from importlib import resources
from pathlib import Path
from urllib.parse import urlsplit

import torch
import torch.nn.functional as F
from instanovo_fm.model.encoder import MODEL_TYPE, FoundationModel
from torch import Tensor, nn

HEAD_INITIALIZER_RANGE = 0.02
DEFAULT_CHECKPOINT = "instanovo-fm-v0.1.0"
CACHE_DIR = Path.home() / ".cache" / "instanovo-fm"


def resolve_checkpoint(name: str | Path) -> Path:
    """Return a local checkpoint path for a file path or a registered model ID.

    Registered IDs (``instanovo_fm/models.json``, e.g. ``instanovo-fm-v0.1.0``)
    are downloaded to the same cache ``FoundationModel.from_pretrained`` uses,
    ``~/.cache/instanovo-fm``, if not already there.
    """
    path = Path(name).expanduser()
    if path.is_file():
        return path.resolve()
    registry_file = resources.files("instanovo_fm").joinpath("models.json")
    registry = json.loads(registry_file.read_text(encoding="utf-8"))[MODEL_TYPE]
    if str(name) not in registry:
        raise FileNotFoundError(
            f"{name} is neither a file nor a registered model; registered: {sorted(registry)}"
        )
    url = registry[str(name)]["remote"]
    cached = CACHE_DIR / urlsplit(url).path.split("/")[-1]
    if not cached.exists():
        from instanovo.utils.file_downloader import download_file

        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        download_file(url, cached, str(name), cached.name)
    return cached


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


class InstaNovoFMDenoiser(nn.Module):
    """Classify peaks as noise from InstaNovo-FM encoder states.

    The encoder is frozen unless ``train_encoder=True`` (full fine-tuning).
    """

    def __init__(
        self,
        encoder: FoundationModel,
        head_hidden_size: int = 128,
        head_dropout: float = 0.1,
        train_encoder: bool = False,
    ):
        super().__init__()
        if encoder.use_meta_token:
            # The meta token needs instrument/collision-energy metadata the
            # denoising dataset does not have.
            raise NotImplementedError("checkpoints with meta_token.enabled are not supported")
        # The masked-reconstruction heads are unused; drop them so they are
        # not counted as encoder parameters.
        del encoder.prediction_heads
        self.encoder = encoder
        self.train_encoder = train_encoder
        if not train_encoder:
            self.encoder.requires_grad_(False)
            self.encoder.eval()
        self.head = PeakDenoisingHead(encoder.dim_model, head_hidden_size, head_dropout)

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint: str | Path,
        head_hidden_size: int = 128,
        head_dropout: float = 0.1,
        random_init: bool = False,
        train_encoder: bool = False,
    ) -> InstaNovoFMDenoiser:
        """Build from an InstaNovo-FM checkpoint file.

        With ``random_init=True`` only the checkpoint's config is used: a fresh
        ``FoundationModel`` with the same architecture is constructed, i.e.
        the untrained model pretraining starts from. Its weights depend on
        the global torch seed.
        """
        # FoundationModel.load uses weights_only=False (the checkpoint pickles
        # an OmegaConf config); only load checkpoints from a trusted source.
        encoder, _ = FoundationModel.load(str(checkpoint))
        if random_init:
            encoder = FoundationModel(
                dim_model=encoder.dim_model,
                n_heads=encoder.n_heads,
                dim_feedforward=encoder.dim_feedforward,
                n_layers=encoder.n_layers,
                dropout=encoder.dropout,
                n_peaks=encoder.n_peaks,
                max_mz=encoder.max_mz,
                min_mz=encoder.min_mz,
                max_charge=encoder.max_charge,
                peak_encoder_type=encoder.peak_encoder_type,
                mz_task=encoder.mz_task,
                use_meta_token=encoder.use_meta_token,
                cfg=encoder.cfg,
            )
        return cls(encoder, head_hidden_size, head_dropout, train_encoder)

    def train(self, mode: bool = True):
        super().train(mode)
        if not self.train_encoder:
            self.encoder.eval()
        return self

    def encode_peaks(self, spectra: Tensor) -> tuple[Tensor, Tensor]:
        """Return per-peak hidden states ``(B, L, D)`` and the padding mask.

        Follows ``FoundationModel.encode`` step for step, but keeps the peak
        tokens instead of the latent token. Padding is detected the same way,
        as all-zero ``[m/z, intensity]`` rows.
        """
        fm = self.encoder
        x = fm._embed_peaks(spectra)
        x = fm._apply_ion_ladder(x, spectra)
        pad_mask = spectra.sum(dim=-1) == 0
        x = fm.apply_pad_token_replacement(x, pad_mask)
        attn_bias, pairwise_feats = fm._compute_attn_bias(spectra, spectra_mask=pad_mask)
        x, num_prepended, _, _ = fm._add_special_tokens(x, None, None, None)
        attn_bias = fm._pad_attn_bias(attn_bias, num_prepended)
        pairwise_feats = fm._pad_pairwise_feats(pairwise_feats, num_prepended)
        src_key_padding_mask = fm._create_padding_mask(pad_mask, num_prepended)
        if isinstance(fm.encoder, nn.TransformerEncoder):
            x = fm.encoder(src=x, mask=None, src_key_padding_mask=src_key_padding_mask)
        else:
            x = fm.encoder(
                src=x,
                src_mask=None,
                src_key_padding_mask=src_key_padding_mask,
                attn_bias=attn_bias,
                pairwise_feats=pairwise_feats,
                is_causal=False,
            )
        # The prepended latent token is not a peak.
        return x[:, num_prepended:], pad_mask

    def forward(self, mz: Tensor, intensity: Tensor, labels: Tensor | None = None):
        """Return ``(loss, logits, valid)``; ``loss`` is ``None`` without labels."""
        with contextlib.nullcontext() if self.train_encoder else torch.no_grad():
            peak_hidden, peak_padding_mask = self.encode_peaks(
                torch.stack([mz, intensity], dim=-1)
            )
        logits = self.head(peak_hidden)

        if labels is None:
            return None, logits, ~peak_padding_mask
        valid = (labels != -100) & ~peak_padding_mask
        loss = (
            F.binary_cross_entropy_with_logits(logits[valid], labels[valid].float())
            if valid.any()
            else logits.sum() * 0.0
        )
        return loss, logits, valid


def count_parameters(module: nn.Module, trainable_only: bool = False) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad or not trainable_only)
