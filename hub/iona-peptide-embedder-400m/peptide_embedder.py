"""iona-peptide-embedder-400m: map a (modified) peptide + precursor charge to the spectrum-embedding
space of iona-contrastive-400m. Self-contained (torch only); a faithful copy of
msdelta.reranking.PeptideEncoder (readout "pool") and its tokenizer.

    from peptide_embedder import PeptideEmbedder
    model = PeptideEmbedder.from_pretrained("path/or/snapshot/dir").eval()
    emb = model.embed(["PEPTIDEK", "AC[57.0215]M[15.9949]K"], charges=[2, 2])   # (2, 2560), unit norm

Peptide notation: residues, each modification as `[mass delta]` right after its residue
(`C[57.0215]`, `M[15.9949]`, `N[0.9840]`); an N-terminal modification is a leading
`[mass]` (e.g. `[42.0106]PEPTIDE`).
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import Tensor, nn

RESIDUES = "ACDEFGHIKLMNPQRSTVWYn"          # `n` = N-terminal marker for a leading [mass]
PAD, UNK = 0, 1
RESIDUE_TO_ID = {r: i + 2 for i, r in enumerate(RESIDUES)}
VOCAB_SIZE = len(RESIDUE_TO_ID) + 2


def parse_peptide(peptide: str) -> tuple[list[int], list[float]]:
    """Residue ids and per-residue modification masses (a bracket binds to the residue before it)."""
    ids: list[int] = []
    masses: list[float] = []
    i = 0
    while i < len(peptide):
        c = peptide[i]
        if c == "[":
            end = peptide.find("]", i)
            if end < 0:
                break
            try:
                mass = float(peptide[i + 1:end])
            except ValueError:
                mass = 0.0
            if masses:
                masses[-1] += mass
            else:
                ids.append(RESIDUE_TO_ID["n"]); masses.append(mass)
            i = end + 1
            continue
        ids.append(RESIDUE_TO_ID.get(c, UNK)); masses.append(0.0)
        i += 1
    return ids, masses


def collate(peptides: list[str], charges: list[int], max_length: int = 64) -> dict[str, Tensor]:
    parsed = [parse_peptide(p) for p in peptides]
    width = max(min(max((len(ids) for ids, _ in parsed), default=1), max_length), 1)
    b = len(parsed)
    residues = torch.zeros(b, width, dtype=torch.long)
    mods = torch.zeros(b, width, dtype=torch.float32)
    mask = torch.zeros(b, width, dtype=torch.long)
    for row, (ids, masses) in enumerate(parsed):
        n = min(len(ids), width)
        if n:
            residues[row, :n] = torch.tensor(ids[:n]); mods[row, :n] = torch.tensor(masses[:n])
            mask[row, :n] = 1
    return {"residues": residues, "modifications": mods, "sequence_mask": mask,
            "charge": torch.tensor(charges, dtype=torch.long)}


class FourierFeatures(nn.Module):
    def __init__(self, n_freqs: int, f_min: float, f_max: float, clamp_abs: float = 2000.0):
        super().__init__()
        self.register_buffer("freqs", torch.logspace(math.log10(f_min), math.log10(f_max), n_freqs))
        self.out_dim = 2 * n_freqs
        self.clamp_abs = clamp_abs

    def forward(self, x: Tensor) -> Tensor:
        x = x.float().clamp(-self.clamp_abs, self.clamp_abs)
        phase = 2.0 * math.pi * x.unsqueeze(-1) * self.freqs.float()
        return torch.cat([phase.sin(), phase.cos()], dim=-1)


class PeptideEmbedder(nn.Module):
    def __init__(self, embedding_size: int = 2560, hidden_size: int = 256, num_layers: int = 4,
                 num_heads: int = 8, max_length: int = 64, n_charges: int = 8,
                 mod_n_freqs: int = 16, dropout: float = 0.1):
        super().__init__()
        self.max_length = max_length
        self.residue = nn.Embedding(VOCAB_SIZE, hidden_size, padding_idx=PAD)
        self.position = nn.Embedding(max_length, hidden_size)
        self.charge = nn.Embedding(n_charges, hidden_size)
        self.mod_features = FourierFeatures(mod_n_freqs, 1e-2, 1e3)
        self.mod_projection = nn.Linear(self.mod_features.out_dim, hidden_size)
        layer = nn.TransformerEncoderLayer(d_model=hidden_size, nhead=num_heads,
                                           dim_feedforward=4 * hidden_size, dropout=dropout,
                                           batch_first=True, norm_first=True, activation="gelu")
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(hidden_size)
        width = 2 * hidden_size                                    # mean + max pooling
        self.projection = nn.Sequential(nn.Linear(width, width), nn.GELU(), nn.Dropout(dropout),
                                        nn.Linear(width, embedding_size))

    def forward(self, residues, modifications, sequence_mask, charge) -> Tensor:
        pos = torch.arange(residues.shape[1], device=residues.device).clamp_max(
            self.position.num_embeddings - 1)
        h = self.residue(residues) + self.position(pos)[None]
        modified = (modifications.abs() > 1e-6).unsqueeze(-1)
        h = h + self.mod_projection(self.mod_features(modifications).to(h.dtype)) * modified
        h = h + self.charge(charge.clamp(0, self.charge.num_embeddings - 1))[:, None]
        dtype = next(self.encoder.parameters()).dtype
        with torch.autocast(device_type=h.device.type, enabled=False):
            h = self.norm(self.encoder(h.to(dtype), src_key_padding_mask=~sequence_mask.bool()))
        m = sequence_mask.bool().unsqueeze(-1)
        mean = (h * m).sum(1) / m.sum(1).clamp_min(1)
        mx = torch.nan_to_num(h.masked_fill(~m, float("-inf")).max(1).values, neginf=0.0)
        return F.normalize(self.projection(torch.cat([mean, mx], -1)).float(), dim=-1)

    @torch.no_grad()
    def embed(self, peptides: list[str], charges: list[int], batch_size: int = 512) -> Tensor:
        device = next(self.parameters()).device
        out = []
        for s in range(0, len(peptides), batch_size):
            b = collate(peptides[s:s + batch_size], charges[s:s + batch_size], self.max_length)
            out.append(self(**{k: v.to(device) for k, v in b.items()}).cpu())
        return torch.cat(out)

    @classmethod
    def from_pretrained(cls, path: str) -> "PeptideEmbedder":
        """`path`: a local dir, or a Hub repo id (needs huggingface_hub)."""
        p = Path(path)
        if not p.exists():
            from huggingface_hub import snapshot_download
            p = Path(snapshot_download(path))
        cfg = json.loads((p / "config.json").read_text())
        model = cls(**{k: cfg[k] for k in ("embedding_size", "hidden_size", "num_layers",
                                           "num_heads", "max_length", "n_charges", "mod_n_freqs")})
        from safetensors.torch import load_file
        state = {k.removeprefix("sequence_encoder."): v
                 for k, v in load_file(str(p / "model.safetensors")).items()}
        model.load_state_dict(state, strict=True)
        return model
