"""Parquet streaming dataset, per-spectrum preprocessing, and MPM collate."""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator

import numpy as np
import pyarrow.parquet as pq
import torch
from torch.utils.data import IterableDataset, get_worker_info


@dataclass
class PreprocessConfig:
    intensity_threshold_frac: float = 0.01
    top_n: int = 150


N_CHARGES = 8  # embedding rows; index 0 = unknown/default, 1..N-1 = charge z (clamped)

# Monoisotopic residue masses (Da); for computing precursor m/z from the label.
_RESIDUE_MASS = {
    "G": 57.02146, "A": 71.03711, "S": 87.03203, "P": 97.05276, "V": 99.06841,
    "T": 101.04768, "C": 103.00919, "L": 113.08406, "I": 113.08406, "N": 114.04293,
    "D": 115.02694, "Q": 128.05858, "K": 128.09496, "E": 129.04259, "M": 131.04049,
    "H": 137.05891, "F": 147.06841, "R": 156.10111, "Y": 163.06333, "W": 186.07931,
}
_WATER = 18.0105646
_PROTON = 1.0072765
_MOD_RE = re.compile(r"\[([0-9.]+)\]")


def charge_index(peptide_charge: str | None) -> int:
    """Parse charge from 'PEPTIDE_z' → index in [0, N_CHARGES). 0 = unknown."""
    if not peptide_charge:
        return 0
    parts = str(peptide_charge).rsplit("_", 1)
    if len(parts) == 2 and parts[1].isdigit():
        return min(int(parts[1]), N_CHARGES - 1)
    return 0


def precursor_mz(peptide_charge: str | None) -> float:
    """Compute precursor m/z from 'PEPTIDE_z' (mod-aware: sums [x] bracket
    masses + residues + water, /z). Returns 0.0 if unparseable.

    Precursor m/z is an *observed* input at deployment (instrument PEPMASS);
    here we source it from the label for the consensus set, where the observed
    value isn't stored. Same physical quantity — not leakage.
    """
    if not peptide_charge:
        return 0.0
    parts = str(peptide_charge).rsplit("_", 1)
    if len(parts) != 2 or not parts[1].isdigit():
        return 0.0
    pep, z = parts[0], int(parts[1])
    if z < 1:
        return 0.0
    mods = sum(float(x) for x in _MOD_RE.findall(pep))
    seq = _MOD_RE.sub("", pep)
    mass = _WATER + mods
    for a in seq:
        m = _RESIDUE_MASS.get(a)
        if m is None:
            return 0.0
        mass += m
    return (mass + z * _PROTON) / z


@dataclass
class MaskConfig:
    """Masked-intensity task (v9). Mask a fraction of peaks' intensity and
    predict it. m/z is never masked — it flows only through the Δm bias
    (tokens are m/z-free), so the bias is the sole carrier of m/z structure.
    """
    mask_ratio: float = 0.15
    min_masked: int = 1


class ConsensusParquet(IterableDataset):
    """Streaming parquet dataset over consensus spectra row-groups.

    Each shard is a parquet file with columns ``m/z`` (list<float>) and
    ``int`` (list<float>). Memory is bounded by one row-group per worker
    (~263 MB raw arrow). Random access within a row-group, random order
    across row-groups, infinite iteration."""

    def __init__(
        self,
        paths: Iterable[str | Path],
        preprocess: PreprocessConfig | None = None,
        mz_col: str = "m/z",
        int_col: str = "int",
        seed: int = 0,
    ):
        super().__init__()
        self.paths: list[Path] = sorted(Path(p) for p in paths)
        if not self.paths:
            raise ValueError("ConsensusParquet needs at least one .parquet path")
        self.preprocess = preprocess or PreprocessConfig()
        self.mz_col = mz_col
        self.int_col = int_col
        self.seed = seed

        # Build (path, row_group) work units up front. Uses parquet metadata
        # only — cheap, ~ms per shard.
        self._units: list[tuple[Path, int]] = []
        for p in self.paths:
            n_rg = pq.ParquetFile(p).num_row_groups
            self._units.extend((p, rg) for rg in range(n_rg))

    def __iter__(self) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
        info = get_worker_info()
        if info is None:
            worker_id, num_workers = 0, 1
        else:
            worker_id, num_workers = info.id, info.num_workers

        # Stable but worker-distinct RNG.
        rng = np.random.default_rng(self.seed + 1_000_003 * worker_id)
        my_units = self._units[worker_id::num_workers]
        if not my_units:
            return

        cols = [self.mz_col, self.int_col, "peptide_charge"]
        epoch = 0
        while True:
            order = rng.permutation(len(my_units))
            for u in order:
                path, rg_idx = my_units[u]
                pf = pq.ParquetFile(path)
                tbl = pf.read_row_group(rg_idx, columns=cols)
                mz_col = tbl.column(self.mz_col)
                int_col = tbl.column(self.int_col)
                pc_col = tbl.column("peptide_charge")
                n = len(tbl)
                row_order = rng.permutation(n)
                for i in row_order:
                    mz_list = mz_col[int(i)].as_py()
                    int_list = int_col[int(i)].as_py()
                    if not mz_list:
                        continue
                    mz_t = torch.tensor(mz_list, dtype=torch.float32)
                    int_t = torch.tensor(int_list, dtype=torch.float32)
                    mz_p, li_p, pp = preprocess_spectrum(mz_t, int_t, self.preprocess)
                    if mz_p.numel() == 0:
                        continue
                    pc = pc_col[int(i)].as_py()
                    yield mz_p, li_p, pp, charge_index(pc), precursor_mz(pc)
                # Drop arrow table reference; GC reclaims ~263 MB before next rg.
                del tbl, mz_col, int_col, pc_col
            epoch += 1


def preprocess_spectrum(
    mz: torch.Tensor,
    intensity: torch.Tensor,
    cfg: PreprocessConfig,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Drop low-intensity peaks, take top-N, produce two intensity views.

    Returns (mz, log_int, intensity_prob), all float32 of length K ≤ cfg.top_n:
      - log_int        log1p(intensity) divided by its per-spectrum max — the
                       input feature embedded by `PeakEmbed` (concentrated in
                       [~0.5, 1.0] by construction, which is fine for the input).
      - intensity_prob raw intensity normalised to sum-to-1 across the kept
                       peaks — the KL target for `IntensityHead.loss`. Treating
                       the spectrum as a probability distribution over m/z is
                       what makes isotope/residue intensity ratios (e.g. M+1 ≈
                       20% M+0 for ¹³C) first-class in the loss.
    """
    mz = mz.to(torch.float32)
    intensity = intensity.to(torch.float32)

    if intensity.numel() == 0:
        return mz, intensity, intensity

    base = float(intensity.max())
    if base <= 0:
        empty = torch.empty(0, dtype=torch.float32)
        return empty, empty, empty

    keep = intensity >= cfg.intensity_threshold_frac * base
    mz = mz[keep]
    intensity = intensity[keep]

    if mz.numel() > cfg.top_n:
        topk = torch.topk(intensity, cfg.top_n, sorted=False)
        mz = mz[topk.indices]
        intensity = topk.values

    log_int = torch.log1p(intensity)
    log_int = log_int / log_int.max().clamp_min(1e-8)

    intensity_prob = intensity / intensity.sum().clamp_min(1e-12)

    return mz.contiguous(), log_int.contiguous(), intensity_prob.contiguous()


def _pad_batch(batch):
    """Pad a list of (mz, log_int, intensity_prob[, charge[, precursor_mz]]) to (B, K_max).
    Returns mz, log_int, intensity_prob, key_padding_mask, charge (B,) long,
    prec_mz (B,) float, Ks."""
    B = len(batch)
    Ks = [int(item[0].numel()) for item in batch]
    K_max = max(max(Ks) if Ks else 1, 1)
    mz = torch.zeros(B, K_max, dtype=torch.float32)
    log_int = torch.zeros(B, K_max, dtype=torch.float32)
    intensity_prob = torch.zeros(B, K_max, dtype=torch.float32)
    key_padding_mask = torch.ones(B, K_max, dtype=torch.bool)
    charge = torch.zeros(B, dtype=torch.long)
    prec_mz = torch.zeros(B, dtype=torch.float32)
    for b, (item, K) in enumerate(zip(batch, Ks)):
        charge[b] = int(item[3]) if len(item) > 3 else 0
        prec_mz[b] = float(item[4]) if len(item) > 4 else 0.0
        if K == 0:
            continue
        mz[b, :K] = item[0]
        log_int[b, :K] = item[1]
        intensity_prob[b, :K] = item[2]
        key_padding_mask[b, :K] = False
    return mz, log_int, intensity_prob, key_padding_mask, charge, prec_mz, Ks


def pad_collate(batch) -> dict[str, torch.Tensor]:
    """Clean padded batch, no masking — for inference (probes, attention probe)."""
    mz, log_int, intensity_prob, kpm, charge, prec_mz, _ = _pad_batch(batch)
    return {"mz": mz, "log_int": log_int, "intensity_prob": intensity_prob,
            "key_padding_mask": kpm, "charge": charge, "precursor_mz": prec_mz}


def mask_intensity_collate(
    batch: list[tuple[torch.Tensor, torch.Tensor]],
    cfg: MaskConfig,
) -> dict[str, torch.Tensor]:
    """Mask a fraction of peaks' intensity for masked-intensity prediction (v9 / v13).

    Returns:
      mz                (B, K_max)  float32  — real m/z (feeds the Δm bias only)
      log_int           (B, K_max)  float32  — input feature for PeakEmbed (log1p÷max)
      intensity_prob    (B, K_max)  float32  — KL target (raw intensity / sum)
      key_padding_mask  (B, K_max)  bool     — True at padding
      mask_positions    (B, K_max)  bool     — True where intensity is hidden & predicted

    m/z is never masked. PeakEmbed swaps in [MASK] at mask_positions so the input
    intensity there doesn't leak; the KL loss reads `intensity_prob` at masked
    positions, re-normalises across the masked subset, and minimises KL against
    the model's softmax over masked logits.
    """
    mz, log_int, intensity_prob, key_padding_mask, charge, prec_mz, Ks = _pad_batch(batch)
    mask_positions = torch.zeros_like(key_padding_mask)
    for b, K in enumerate(Ks):
        if K == 0:
            continue
        n_mask = max(cfg.min_masked, int(round(K * cfg.mask_ratio)))
        n_mask = min(n_mask, K)
        idx = torch.randperm(K)[:n_mask]
        mask_positions[b, idx] = True
    return {
        "mz": mz,
        "log_int": log_int,
        "intensity_prob": intensity_prob,
        "key_padding_mask": key_padding_mask,
        "mask_positions": mask_positions,
        "charge": charge,
        "precursor_mz": prec_mz,
    }


def split_paths(root: str | Path, n_val: int) -> tuple[list[Path], list[Path]]:
    """Deterministic file-level split: last n_val shards (sorted) become val."""
    root = Path(root)
    shards = sorted(root.glob("*.parquet"))
    if len(shards) <= n_val:
        raise ValueError(f"only {len(shards)} shards; need > {n_val} for a split")
    return shards[:-n_val], shards[-n_val:]
