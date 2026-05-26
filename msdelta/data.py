"""Parquet streaming dataset, per-spectrum preprocessing, and MPM collate."""
from __future__ import annotations

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


def charge_index(peptide_charge: str | None) -> int:
    """Parse charge from 'PEPTIDE_z' → index in [0, N_CHARGES). 0 = unknown."""
    if not peptide_charge:
        return 0
    parts = str(peptide_charge).rsplit("_", 1)
    if len(parts) == 2 and parts[1].isdigit():
        return min(int(parts[1]), N_CHARGES - 1)
    return 0


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
                    mz_p, li_p = preprocess_spectrum(mz_t, int_t, self.preprocess)
                    if mz_p.numel() == 0:
                        continue
                    yield mz_p, li_p, charge_index(pc_col[int(i)].as_py())
                # Drop arrow table reference; GC reclaims ~263 MB before next rg.
                del tbl, mz_col, int_col, pc_col
            epoch += 1


def preprocess_spectrum(
    mz: torch.Tensor,
    intensity: torch.Tensor,
    cfg: PreprocessConfig,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Drop low-intensity peaks, take top-N, log+normalize intensity.

    Returns (mz, log_int) as float32, both of length K ≤ cfg.top_n.
    """
    mz = mz.to(torch.float32)
    intensity = intensity.to(torch.float32)

    if intensity.numel() == 0:
        return mz, intensity

    base = float(intensity.max())
    if base <= 0:
        empty = torch.empty(0, dtype=torch.float32)
        return empty, empty

    keep = intensity >= cfg.intensity_threshold_frac * base
    mz = mz[keep]
    intensity = intensity[keep]

    if mz.numel() > cfg.top_n:
        topk = torch.topk(intensity, cfg.top_n, sorted=False)
        mz = mz[topk.indices]
        intensity = topk.values

    log_int = torch.log1p(intensity)
    log_int = log_int / log_int.max().clamp_min(1e-8)
    return mz.contiguous(), log_int.contiguous()


def _pad_batch(batch):
    """Pad a list of (mz, log_int[, charge]) to (B, K_max).
    Returns mz, log_int, key_padding_mask, charge (B,) long, Ks."""
    B = len(batch)
    Ks = [int(item[0].numel()) for item in batch]
    K_max = max(max(Ks) if Ks else 1, 1)
    mz = torch.zeros(B, K_max, dtype=torch.float32)
    log_int = torch.zeros(B, K_max, dtype=torch.float32)
    key_padding_mask = torch.ones(B, K_max, dtype=torch.bool)
    charge = torch.zeros(B, dtype=torch.long)
    for b, (item, K) in enumerate(zip(batch, Ks)):
        charge[b] = int(item[2]) if len(item) > 2 else 0
        if K == 0:
            continue
        mz[b, :K] = item[0]
        log_int[b, :K] = item[1]
        key_padding_mask[b, :K] = False
    return mz, log_int, key_padding_mask, charge, Ks


def pad_collate(batch) -> dict[str, torch.Tensor]:
    """Clean padded batch, no masking — for inference (probes, attention probe)."""
    mz, log_int, kpm, charge, _ = _pad_batch(batch)
    return {"mz": mz, "log_int": log_int, "key_padding_mask": kpm, "charge": charge}


def mask_intensity_collate(
    batch: list[tuple[torch.Tensor, torch.Tensor]],
    cfg: MaskConfig,
) -> dict[str, torch.Tensor]:
    """Mask a fraction of peaks' intensity for masked-intensity prediction (v9).

    Returns:
      mz                (B, K_max)  float32  — real m/z (feeds the Δm bias only)
      log_int           (B, K_max)  float32  — real log-intensity (also the target)
      key_padding_mask  (B, K_max)  bool     — True at padding
      mask_positions    (B, K_max)  bool     — True where intensity is hidden & predicted

    m/z is never masked. log_int is passed clean; PeakEmbed swaps in [MASK] at
    mask_positions so the input intensity there doesn't leak, and the loss reads
    the target from log_int at mask_positions.
    """
    mz, log_int, key_padding_mask, charge, Ks = _pad_batch(batch)
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
        "key_padding_mask": key_padding_mask,
        "mask_positions": mask_positions,
        "charge": charge,
    }


def split_paths(root: str | Path, n_val: int) -> tuple[list[Path], list[Path]]:
    """Deterministic file-level split: last n_val shards (sorted) become val."""
    root = Path(root)
    shards = sorted(root.glob("*.parquet"))
    if len(shards) <= n_val:
        raise ValueError(f"only {len(shards)} shards; need > {n_val} for a split")
    return shards[:-n_val], shards[-n_val:]
