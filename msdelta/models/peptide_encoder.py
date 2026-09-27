"""The peptide encoder: a (modified) peptide plus its precursor charge -> the spectrum-embedding
space of a frozen msdelta spectrum encoder (trained by msdelta.finetuning.alignment).

A standard Hugging Face model, like the spectrum encoder:

    from msdelta.models.peptide_encoder import PeptideEncoderModel
    model = PeptideEncoderModel.from_pretrained("Gaolaboratory/iona-peptide-embedder-400m").eval()
    emb = model.embed(["PEPTIDEK", "AC[57.0215]M[15.9949]K"], charges=[2, 2])   # (2, D), unit norm

(The Hub repo keeps its original name.) `from_pretrained` reads every layout saved so far:
the standard layout (model_type "msdelta-peptide-encoder", or "msdelta-peptide-embedder" as
saved before the rename), an alignment training run's `final/` (raw weights, keys
`sequence_encoder.*`, no config) and the first Hub release (config with model_type
"iona-peptide-embedder"). Peptide notation: residues, each modification as `[mass delta]`
right after its residue (`C[57.0215]`); an N-terminal modification is a leading `[mass]`.

Until 2026-09-27 this was the "peptide embedder" (msdelta.models.peptide_embedder,
PeptideEmbedderConfig / PeptideEmbedderModel). The old module is an alias of this one and the
old class names are aliases at the end of this file, so old imports, scripts and pickles keep
working. `PeptideEncoder` (no suffix) is the inner sequence tower, as before.

PeptideEncoder, PeptideCollator and parse_peptide moved here from msdelta.rescoring.reranking,
which re-exports them.
"""

from __future__ import annotations

import contextlib
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from transformers import PretrainedConfig, PreTrainedModel

from msdelta.models.fourier import FourierFeatures

# 20 standard residues plus `n`, which this corpus uses as an N-terminal marker.
RESIDUES = "ACDEFGHIKLMNPQRSTVWYn"
PAD, UNK = 0, 1
RESIDUE_TO_ID = {residue: index + 2 for index, residue in enumerate(RESIDUES)}
VOCAB_SIZE = len(RESIDUE_TO_ID) + 2
_MOD = re.compile(r"\[([+-]?[0-9.]+)\]")



# ---------------------------------------------------------------------------- probes

# Off unless MSDELTA_PROBES is set, so a real training run pays one boolean per call site.
# Turn them on for debug-queue runs: MSDELTA_PROBES=1.
PROBES = os.environ.get("MSDELTA_PROBES", "") not in ("", "0", "false", "False")

# Force scaled_dot_product_attention onto the unfused MATH backend when set.
#
# oneDNN's fused SDPA kernel is a documented source of GPU page faults on Intel GPUs --
# pytorch/pytorch#195319 reports exactly this fault class from an out-of-bounds access in
# the fused kernel's second-tile handling, triggered by a rank-4 attention mask, with
# "force the MATH backend" as the first workaround. nn.TransformerEncoder builds a rank-4
# mask internally whenever a key_padding_mask is supplied, which this model always does.
#
# Not on by default: MATH materialises the full attention matrix and is slower. This is
# a diagnostic switch, and a fallback if it turns out to be the fix.
SDPA_MATH = os.environ.get("MSDELTA_SDPA_MATH", "") not in ("", "0", "false", "False")


def _sdpa_context():
    """MATH-only SDPA when MSDELTA_SDPA_MATH is set, otherwise a no-op."""
    if not SDPA_MATH:
        return contextlib.nullcontext()
    from torch.nn.attention import SDPBackend, sdpa_kernel
    return sdpa_kernel(SDPBackend.MATH)


def probe(where: str, *, sync: Tensor | None = None, **tensors) -> None:
    """Check tensors at a named point, and optionally make the device catch up first.

    `sync` is the important argument. XPU kernels are asynchronous, so a GPU page fault
    is reported wherever the host happens to be when the driver notices, which can be
    dozens of steps past whatever caused it -- job 8840356 faulted "at step 56" and job
    8840238 "at step 73", and neither number means anything without a synchronise. Pass
    any tensor on the device and this blocks until the queue drains, so the fault is
    attributed to the stage that actually caused it.

    Everything else checked here is cheap and has already bitten once: dtype mismatches
    (8840257, 8840336), indices past the end of an embedding table, and non-finite values.
    """
    if not PROBES:
        return
    if sync is not None and sync.device.type == "xpu":
        torch.xpu.synchronize()
    for name, value in tensors.items():
        if not isinstance(value, Tensor):
            continue
        if value.dtype.is_floating_point and not torch.isfinite(value).all():
            n = int((~torch.isfinite(value)).sum())
            raise RuntimeError(f"[probe {where}] {name}: {n} non-finite of {value.numel()}")


def probe_index(where: str, name: str, index: Tensor, limit: int) -> None:
    """An out-of-range embedding index reads unmapped memory on GPU rather than raising.

    On CPU nn.Embedding raises IndexError. On GPU the gather is unchecked, so the read
    lands wherever the arithmetic points and surfaces as `type: 0 (NotPresent),
    access: 0 (Read)` -- which is exactly the signature of FT9. Checking explicitly turns
    a page fault into a sentence naming the tensor and the offending value.
    """
    if not PROBES or index.numel() == 0:
        return
    low, high = int(index.min()), int(index.max())
    if low < 0 or high >= limit:
        raise RuntimeError(
            f"[probe {where}] {name} out of range for a table of {limit}: "
            f"min {low}, max {high}"
        )


# "weighted" variants scale each peak's contribution before averaging. A mass spectrum
# is mostly noise -- the denoise corpus is 53% noise by peak count -- so an unweighted
# mean over 512 peaks is dominated by peaks that carry no identity, and the max is taken
# over those same dimensions. Weights can be intensity (free, and intense fragments are
# what identify a peptide) or a denoiser's P(signal), which is the same information
# learned rather than assumed.
POOLING_MODES = ("mean", "mean+max", "weighted_mean", "weighted_mean+max")


def pool_sequence(tokens: Tensor, mask: Tensor, mode: str = "mean+max",
                  weights: Tensor | None = None) -> Tensor:
    """Reduce variable-length token embeddings to one vector.

    `mean` is the field's default -- sentence_transformers uses it for every encoder
    model, and SBERT's comparison of CLS/mean/max put mean ahead. `mean+max` is what this
    repo's SpectrumRetrievalHead and embedding.pool_tokens already do, so it is kept as
    the default here to stay consistent with numbers measured against that space; it also
    doubles the width, which is a real cost on the peptide side where a sequence is ~15
    residues and a max over 15 tokens is closer to noise than to a feature.

    Both towers must use the same mode. The alignment target is whatever the teacher
    emits, so a mismatch is not a modelling choice, it is a shape error waiting to happen.
    """
    if mode not in POOLING_MODES:
        raise ValueError(f"pooling must be one of {POOLING_MODES}, got {mode!r}")
    mask = mask.bool().unsqueeze(-1)
    if mode.startswith("weighted"):
        if weights is None:
            raise ValueError(f"pooling {mode!r} needs per-peak weights")
        # Normalised so the result stays on the same scale as the unweighted mean;
        # otherwise every downstream threshold and the L2 target space shift with it.
        w = (weights.unsqueeze(-1) * mask).clamp_min(0)
        mean = (tokens * w).sum(1) / w.sum(1).clamp_min(1e-9)
    else:
        mean = (tokens * mask).sum(1) / mask.sum(1).clamp_min(1)
    if mode in ("mean", "weighted_mean"):
        return mean
    maximum = torch.nan_to_num(tokens.masked_fill(~mask, float("-inf")).max(1).values,
                               neginf=0.0)
    return torch.cat([mean, maximum], dim=-1)


def pooled_width(hidden_size: int, mode: str) -> int:
    """Width `pool_sequence` produces, so the projection can be sized without a forward."""
    # Keyed on whether a max is concatenated, not on an exact name, so the weighted
    # variants size correctly too.
    return 2 * hidden_size if mode.endswith("mean+max") else hidden_size


def parse_peptide(peptide: str) -> tuple[list[int], list[float]]:
    """Split a modified peptide into residue ids and per-residue modification masses.

    Peptides arrive as `SAC[57.0215]GVC[57.0215]PGR`. A bracket binds to the residue
    before it, which is this corpus's convention. The mass is kept as a continuous
    per-residue feature rather than folded into a token: a mass is a number, two
    different masses on one residue are two different chemical species, and a token
    vocabulary would either collapse them or explode.
    """
    ids: list[int] = []
    masses: list[float] = []
    index = 0
    while index < len(peptide):
        char = peptide[index]
        if char == "[":
            end = peptide.find("]", index)
            if end < 0:
                break
            try:
                mass = float(peptide[index + 1 : end])
            except ValueError:
                mass = 0.0
            if masses:
                masses[-1] += mass
            else:
                # A leading bracket has no preceding residue; attach it to an N-terminal
                # marker rather than discarding it.
                ids.append(RESIDUE_TO_ID["n"])
                masses.append(mass)
            index = end + 1
            continue
        ids.append(RESIDUE_TO_ID.get(char, UNK))
        masses.append(0.0)
        index += 1
    return ids, masses


def peptide_key(peptide: str, charge: int, by_charge: bool = True) -> str:
    """Identity used to decide whether two spectra are "the same sequence".

    Charge-aware by default: one peptide at charge 2 and the same peptide at charge 3
    fragment differently enough that their spectra are not really replicates, and merging
    them would make the task look easier than it is.
    """
    return f"{peptide}_{charge}" if by_charge else peptide


@dataclass
class PeptideCollator:
    """Pad parsed peptides into a batch."""

    max_length: int = 64
    # Pad every batch to max_length instead of to the longest peptide in it, so the shape
    # is identical on every step. See AlignmentCollator for why that is worth the few
    # wasted positions.
    pad_to_max: bool = False

    def __call__(self, peptides: list[str], charges: list[int]) -> dict[str, Tensor]:
        parsed = [parse_peptide(p) for p in peptides]
        width = (self.max_length if self.pad_to_max else
                 max(min(max((len(i) for i, _ in parsed), default=1), self.max_length), 1))
        batch = len(parsed)
        residues = torch.zeros(batch, width, dtype=torch.long)
        modifications = torch.zeros(batch, width, dtype=torch.float32)
        sequence_mask = torch.zeros(batch, width, dtype=torch.long)
        for row, (ids, masses) in enumerate(parsed):
            length = min(len(ids), width)
            if length == 0:
                continue
            residues[row, :length] = torch.tensor(ids[:length], dtype=torch.long)
            modifications[row, :length] = torch.tensor(masses[:length], dtype=torch.float32)
            sequence_mask[row, :length] = 1
        return {
            "residues": residues,
            "modifications": modifications,
            "sequence_mask": sequence_mask,
            "charge": torch.tensor(charges, dtype=torch.long),
        }


class PeptideEncoder(nn.Module):
    """Encode a modified peptide plus its charge into a fixed-size embedding."""

    def __init__(
        self,
        embedding_size: int,
        hidden_size: int = 256,
        num_layers: int = 4,
        num_heads: int = 8,
        max_length: int = 64,
        n_charges: int = 8,
        mod_n_freqs: int = 16,
        dropout: float = 0.1,
        pooling: str = "mean+max",
        readout: str = "pool",
    ):
        super().__init__()
        self.pooling = pooling
        # How the residue tokens become ONE vector. "pool" (default, every model before
        # A3): `pooling` over the tokens -- mean+max is nearly order-blind, which is why
        # the A1 student ranks an adjacent-residue swap above the truth 30% of the time.
        # "cls": a learned token prepended to the sequence, its output is the embedding.
        # "attn": a learned query attending over the tokens. Both keep order information
        # the transformer already computed (positions are embedded). PLAN.md A3.
        if readout not in ("pool", "cls", "attn"):
            raise ValueError(f"readout must be pool, cls or attn, not {readout!r}")
        self.readout = readout
        if readout == "cls":
            self.cls = nn.Parameter(torch.randn(1, 1, hidden_size) * 0.02)
        if readout == "attn":
            self.attn_query = nn.Parameter(torch.randn(1, 1, hidden_size) * 0.02)
            self.attn = nn.MultiheadAttention(hidden_size, num_heads, dropout=dropout,
                                              batch_first=True)
        self.residue = nn.Embedding(VOCAB_SIZE, hidden_size, padding_idx=PAD)
        self.position = nn.Embedding(max_length, hidden_size)
        self.charge = nn.Embedding(n_charges, hidden_size)
        # 1e-2..1e3 spans a whole modification down to fine isotopic structure.
        self.mod_features = FourierFeatures(mod_n_freqs, 1e-2, 1e3)
        self.mod_projection = nn.Linear(self.mod_features.out_dim, hidden_size)

        layer = nn.TransformerEncoderLayer(
            d_model=hidden_size, nhead=num_heads, dim_feedforward=4 * hidden_size,
            dropout=dropout, batch_first=True, norm_first=True, activation="gelu",
        )
        # Skip the NestedTensor conversion. The padding here is a few residues out of 64,
        # so it buys nothing, and it is a second fast path to reason about. See forward()
        # for the one that actually bites.
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers,
                                             enable_nested_tensor=False)
        self.norm = nn.LayerNorm(hidden_size)
        width = pooled_width(hidden_size, pooling) if readout == "pool" else hidden_size
        self.projection = nn.Sequential(
            nn.Linear(width, width), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(width, embedding_size),
        )

    def forward(self, residues, modifications, sequence_mask, charge) -> Tensor:
        probe_index("student.in", "residues", residues, self.residue.num_embeddings)
        probe_index("student.in", "charge", charge, self.charge.num_embeddings)
        probe("student.in", modifications=modifications)
        length = residues.shape[1]
        position = torch.arange(length, device=residues.device).clamp_max(
            self.position.num_embeddings - 1
        )
        probe_index("student.pos", "position", position, self.position.num_embeddings)
        hidden = self.residue(residues) + self.position(position)[None]
        # Only modified residues get a mass contribution; an unmodified residue must not
        # be handed the Fourier encoding of zero as though it were a real modification.
        modified = (modifications.abs() > 1e-6).unsqueeze(-1)
        mod = self.mod_projection(self.mod_features(modifications).to(hidden.dtype))
        hidden = hidden + mod * modified
        hidden = hidden + self.charge(charge.clamp(0, self.charge.num_embeddings - 1))[:, None]
        if self.readout == "cls":
            hidden = torch.cat([self.cls.expand(hidden.shape[0], -1, -1).to(hidden.dtype),
                                hidden], dim=1)
            sequence_mask = torch.cat([torch.ones_like(sequence_mask[:, :1]),
                                       sequence_mask], dim=1)
        # Run the stack with autocast off, in whatever dtype the weights are.
        #
        # torch's TransformerEncoderLayer has a fused fast path
        # (torch._transformer_encoder_layer_fwd) that it takes only when grad is disabled
        # -- eval, never training -- and that kernel does not honour autocast. It is meant
        # to be guarded by
        #
        #     elif torch.is_autocast_enabled():   # transformer.py:869
        #
        # but the no-argument form of that call reports CUDA's autocast state, so under
        # torch.autocast("xpu") it returns False and the guard never fires. Job 8840257
        # trained 200 steps and died on its first evaluation: "expected scalar type
        # BFloat16 but found Float".
        #
        # What the kernel cannot tolerate is a MISMATCH, not bf16. Forcing fp32 fixed the
        # single-tile case and then broke DeepSpeed, which holds the parameters in bf16 --
        # job 8840336 died at step 0 with the same error inverted, "expected scalar type
        # Float but found BFloat16". Matching the activations to the parameters is right
        # in both: fp32 against fp32 weights on one tile, bf16 against bf16 under ZeRO-2.
        # It also keeps training and evaluation numerically identical, which is the other
        # reason not to leave this to autocast.
        param_dtype = next(self.encoder.parameters()).dtype
        with torch.autocast(device_type=hidden.device.type, enabled=False):
            hidden = hidden.to(param_dtype)
            with _sdpa_context():
                hidden = self.norm(
                    self.encoder(hidden, src_key_padding_mask=~sequence_mask.bool())
                )
        probe("student.encoded", sync=hidden, hidden=hidden)
        if self.readout == "cls":
            pooled = hidden[:, 0]
        elif self.readout == "attn":
            query = self.attn_query.expand(hidden.shape[0], -1, -1).to(hidden.dtype)
            pooled = self.attn(query, hidden, hidden,
                               key_padding_mask=~sequence_mask.bool(),
                               need_weights=False)[0][:, 0]
        else:
            pooled = pool_sequence(hidden, sequence_mask, self.pooling)
        out = F.normalize(self.projection(pooled).float(), dim=-1)
        probe("student.out", sync=out, out=out)
        return out




def student_readout(state: dict, prefix: str = "sequence_encoder.") -> str:
    """Which PeptideEncoder readout a saved student used, read off its weight names, so
    loaders need no extra config (every pre-A3 student has neither key -> "pool")."""
    keys = {k[len(prefix):] if k.startswith(prefix) else k for k in state}
    return "cls" if "cls" in keys else "attn" if "attn_query" in keys else "pool"




class PeptideEncoderConfig(PretrainedConfig):
    """Architecture of a PeptideEncoder, plus which spectrum encoder's space it maps into."""

    model_type = "msdelta-peptide-encoder"

    def __init__(self, embedding_size: int = 2560, hidden_size: int = 256, num_layers: int = 4,
                 num_heads: int = 8, max_length: int = 64, n_charges: int = 8,
                 mod_n_freqs: int = 16, dropout: float = 0.1, pooling: str = "mean+max",
                 readout: str = "pool", spectrum_model: str | None = None,
                 spectrum_pooling: str = "mean+max", **kwargs):
        super().__init__(**kwargs)
        self.embedding_size, self.hidden_size, self.num_layers = embedding_size, hidden_size, num_layers
        self.num_heads, self.max_length, self.n_charges = num_heads, max_length, n_charges
        self.mod_n_freqs, self.dropout, self.pooling, self.readout = mod_n_freqs, dropout, pooling, readout
        self.spectrum_model, self.spectrum_pooling = spectrum_model, spectrum_pooling


# model_type of the standard layout as saved before the rename (2026-09-27); still loaded as is.
LEGACY_MODEL_TYPE = "msdelta-peptide-embedder"

_ARCH = ("embedding_size", "hidden_size", "num_layers", "num_heads", "max_length", "n_charges",
         "mod_n_freqs")


class PeptideEncoderModel(PreTrainedModel):
    """PeptideEncoder as a PreTrainedModel. The encoder sits at `sequence_encoder`, so the
    weight names are exactly those of an alignment checkpoint and of the first Hub release."""

    config_class = PeptideEncoderConfig
    base_model_prefix = "sequence_encoder"
    main_input_name = "residues"

    def __init__(self, config: PeptideEncoderConfig):
        super().__init__(config)
        self.sequence_encoder = PeptideEncoder(
            embedding_size=config.embedding_size, hidden_size=config.hidden_size,
            num_layers=config.num_layers, num_heads=config.num_heads, max_length=config.max_length,
            n_charges=config.n_charges, mod_n_freqs=config.mod_n_freqs, dropout=config.dropout,
            pooling=config.pooling, readout=config.readout)
        self.post_init()

    def _init_weights(self, module):
        # PeptideEncoder initialises itself (torch defaults); keep that, as training always did.
        pass

    def forward(self, residues, modifications, sequence_mask, charge) -> Tensor:
        """Unit-norm embeddings (batch, embedding_size), from PeptideCollator's batch."""
        return self.sequence_encoder(residues, modifications, sequence_mask, charge)

    @torch.no_grad()
    def embed(self, peptides: list[str], charges: list[int], batch_size: int = 512) -> Tensor:
        """Embed peptides (our bracket notation) at their precursor charges; returns CPU float32."""
        device = next(self.parameters()).device
        collate = PeptideCollator(max_length=self.config.max_length)
        out = []
        for start in range(0, len(peptides), batch_size):
            batch = collate(list(peptides[start:start + batch_size]),
                            [min(max(int(c), 0), self.config.n_charges - 1)
                             for c in charges[start:start + batch_size]])
            out.append(self(**{k: v.to(device) for k, v in batch.items()}).float().cpu())
        return torch.cat(out) if out else torch.zeros(0, self.config.embedding_size)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *args, **kwargs):
        """Standard layout via transformers; the two older layouts are read and converted."""
        path = Path(str(pretrained_model_name_or_path))
        if not path.exists():
            from huggingface_hub import snapshot_download
            path = Path(snapshot_download(str(pretrained_model_name_or_path),
                                          token=kwargs.get("token")))
        config_file = path / "config.json"
        model_type = (json.loads(config_file.read_text()).get("model_type")
                      if config_file.exists() else None)
        if model_type == PeptideEncoderConfig.model_type:
            return super().from_pretrained(str(path), *args, **kwargs)
        if model_type == LEGACY_MODEL_TYPE:
            # Same layout, older name: build the config ourselves (transformers would warn about
            # the model_type mismatch) and let the standard loader read the weights.
            if kwargs.get("config") is None:
                raw = json.loads(config_file.read_text())
                raw["model_type"] = PeptideEncoderConfig.model_type
                kwargs["config"] = PeptideEncoderConfig.from_dict(raw)
            return super().from_pretrained(str(path), *args, **kwargs)
        return cls._from_legacy(path, json.loads(config_file.read_text()) if model_type else None,
                                num_heads=kwargs.get("num_heads", 8))

    @classmethod
    def _from_legacy(cls, path: Path, hub_config: dict | None, num_heads: int = 8) -> "PeptideEncoderModel":
        """An alignment run's final/ does not record the attention head count (not in the weights);
        every trained student used 8. Pass num_heads= to from_pretrained if one did not."""
        from safetensors.torch import load_file
        state = load_file(str(path / "model.safetensors"))
        state = {k: v for k, v in state.items() if k.startswith("sequence_encoder.")}
        if not state:
            raise ValueError(f"{path}: no sequence_encoder.* weights -- not a peptide encoder")
        if hub_config is not None:                         # first Hub release
            config = PeptideEncoderConfig(**{k: hub_config[k] for k in _ARCH if k in hub_config},
                                          pooling=hub_config.get("pooling", "mean+max"),
                                          readout=hub_config.get("readout", "pool"),
                                          spectrum_model=hub_config.get("spectrum_model"))
        else:                                              # an alignment run's final/: read the shapes
            layers = {int(k.split(".")[3]) for k in state if k.startswith("sequence_encoder.encoder.layers.")}
            config = PeptideEncoderConfig(
                embedding_size=int(state["sequence_encoder.projection.3.weight"].shape[0]),
                hidden_size=int(state["sequence_encoder.residue.weight"].shape[1]),
                num_layers=len(layers),
                max_length=int(state["sequence_encoder.position.weight"].shape[0]),
                n_charges=int(state["sequence_encoder.charge.weight"].shape[0]),
                mod_n_freqs=int(state["sequence_encoder.mod_features.freqs"].shape[0])
                if "sequence_encoder.mod_features.freqs" in state else 16,
                num_heads=num_heads, readout=student_readout(state))
        model = cls(config)
        missing, unexpected = model.load_state_dict(state, strict=False)
        missing = [k for k in missing if not k.endswith("mod_features.freqs")]
        if missing or unexpected:
            raise ValueError(f"{path}: weights do not match (missing {missing[:5]}, unexpected {unexpected[:5]})")
        return model


# Names before the 2026-09-27 rename ("peptide embedder" -> "peptide encoder"); kept so old code,
# scripts and pickles keep working. New code should use the Encoder names.
PeptideEmbedderConfig = PeptideEncoderConfig
PeptideEmbedderModel = PeptideEncoderModel
