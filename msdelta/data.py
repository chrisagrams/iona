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


@dataclass
class MPMConfig:
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

        cols = [self.mz_col, self.int_col]
        epoch = 0
        while True:
            order = rng.permutation(len(my_units))
            for u in order:
                path, rg_idx = my_units[u]
                pf = pq.ParquetFile(path)
                tbl = pf.read_row_group(rg_idx, columns=cols)
                mz_col = tbl.column(self.mz_col)
                int_col = tbl.column(self.int_col)
                n = len(tbl)
                row_order = rng.permutation(n)
                for i in row_order:
                    mz_list = mz_col[int(i)].as_py()
                    int_list = int_col[int(i)].as_py()
                    if not mz_list:
                        continue
                    mz_t = torch.tensor(mz_list, dtype=torch.float32)
                    int_t = torch.tensor(int_list, dtype=torch.float32)
                    out = preprocess_spectrum(mz_t, int_t, self.preprocess)
                    if out[0].numel() == 0:
                        continue
                    yield out
                # Drop arrow table reference; GC reclaims ~263 MB before next rg.
                del tbl, mz_col, int_col
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


def mpm_collate(
    batch: list[tuple[torch.Tensor, torch.Tensor]],
    mpm: MPMConfig,
) -> dict[str, torch.Tensor]:
    """Pad, mask, and gather targets for masked peak modeling.

    Returns a dict with:
      mz, log_int             (B, K_max)  float32; masked positions are zero
      key_padding_mask        (B, K_max)  bool, True at padding
      mask_positions          (B, K_max)  bool, True where the model must predict
      target_mz, target_logint (M,)       float32; concatenated across batch
      target_index            (M, 2)      int64; (batch_idx, pos_idx) of each target
    """
    B = len(batch)
    Ks = [int(mz.numel()) for mz, _ in batch]
    K_max = max(Ks) if Ks else 1
    K_max = max(K_max, 1)

    mz = torch.zeros(B, K_max, dtype=torch.float32)
    log_int = torch.zeros(B, K_max, dtype=torch.float32)
    key_padding_mask = torch.ones(B, K_max, dtype=torch.bool)
    mask_positions = torch.zeros(B, K_max, dtype=torch.bool)

    targets_mz: list[torch.Tensor] = []
    targets_logint: list[torch.Tensor] = []
    target_index: list[torch.Tensor] = []

    for b, ((sp_mz, sp_log_int), K) in enumerate(zip(batch, Ks)):
        if K == 0:
            continue
        mz[b, :K] = sp_mz
        log_int[b, :K] = sp_log_int
        key_padding_mask[b, :K] = False

        n_mask = max(mpm.min_masked, int(round(K * mpm.mask_ratio)))
        n_mask = min(n_mask, K)
        idx = torch.randperm(K)[:n_mask]
        mask_positions[b, idx] = True

        targets_mz.append(sp_mz[idx])
        targets_logint.append(sp_log_int[idx])
        target_index.append(
            torch.stack([torch.full((n_mask,), b, dtype=torch.long), idx.to(torch.long)], dim=1)
        )

        # Keep real m/z at masked positions — the [MASK] embedding swap in
        # PeakEmbed already prevents direct leakage through the token, and
        # the encoder zeros the Δm bias on pairs that touch a masked
        # position so the m/z can't leak via the bias→logits path either.
        # log_int IS zeroed (it's a prediction target and only flows through
        # the embedding, which is replaced by [MASK] anyway).
        log_int[b, idx] = 0.0

    if targets_mz:
        target_mz = torch.cat(targets_mz)
        target_logint = torch.cat(targets_logint)
        target_index_t = torch.cat(target_index)
    else:
        target_mz = torch.empty(0, dtype=torch.float32)
        target_logint = torch.empty(0, dtype=torch.float32)
        target_index_t = torch.empty(0, 2, dtype=torch.long)

    return {
        "mz": mz,
        "log_int": log_int,
        "key_padding_mask": key_padding_mask,
        "mask_positions": mask_positions,
        "target_mz": target_mz,
        "target_logint": target_logint,
        "target_index": target_index_t,
    }


def split_paths(root: str | Path, n_val: int) -> tuple[list[Path], list[Path]]:
    """Deterministic file-level split: last n_val shards (sorted) become val."""
    root = Path(root)
    shards = sorted(root.glob("*.parquet"))
    if len(shards) <= n_val:
        raise ValueError(f"only {len(shards)} shards; need > {n_val} for a split")
    return shards[:-n_val], shards[-n_val:]
