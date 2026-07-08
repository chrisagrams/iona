"""Device abstraction so the same training code runs on NVIDIA (CUDA) and
Intel (XPU) GPUs without branching at every call site.

Selection order:
  1. explicit override via config `train.device` ("cuda" / "xpu" / "cpu"), else
  2. the first available accelerator (cuda, then xpu), else
  3. cpu.

Everything downstream (`set_device`, `manual_seed_all`, `empty_cache`,
`init_process_group` backend) routes through the resolved backend, so train.py
never mentions cuda or xpu directly.
"""
from __future__ import annotations

import os
import torch


def _has_xpu() -> bool:
    return hasattr(torch, "xpu") and torch.xpu.is_available()


def _has_cuda() -> bool:
    return torch.cuda.is_available()


def resolve_accelerator(prefer: str | None = None) -> str:
    """Return the accelerator family to use: 'cuda', 'xpu', or 'cpu'.

    `prefer` is the config's train.device. "cuda"/"xpu"/"cpu" force that family
    (validated against availability); "auto" or None auto-detects.
    """
    if prefer and prefer not in ("auto", ""):
        fam = prefer.split(":", 1)[0]
        if fam == "cuda" and not _has_cuda():
            raise RuntimeError("train.device=cuda but no CUDA device is available")
        if fam == "xpu" and not _has_xpu():
            raise RuntimeError("train.device=xpu but no XPU device is available")
        return fam
    if _has_cuda():
        return "cuda"
    if _has_xpu():
        return "xpu"
    return "cpu"


def _mod(family: str):
    """Return the torch submodule (torch.cuda / torch.xpu) for a family."""
    return getattr(torch, family)


def ddp_backend(family: str) -> str:
    """Collective backend matching the accelerator: nccl for CUDA, xccl for XPU."""
    return {"cuda": "nccl", "xpu": "xccl", "cpu": "gloo"}[family]


def set_device(family: str, local_rank: int) -> None:
    if family in ("cuda", "xpu"):
        _mod(family).set_device(local_rank)


def device_str(family: str, local_rank: int) -> str:
    """e.g. 'cuda:0' / 'xpu:0' / 'cpu'."""
    return "cpu" if family == "cpu" else f"{family}:{local_rank}"


def manual_seed_all(family: str, seed: int) -> None:
    if family in ("cuda", "xpu"):
        _mod(family).manual_seed_all(seed)


def empty_cache(family: str) -> None:
    if family in ("cuda", "xpu"):
        _mod(family).empty_cache()


def read_dist_env() -> tuple[int, int, int] | None:
    """Discover (rank, world_size, local_rank) from the launcher environment.

    Supports torchrun (RANK/WORLD_SIZE/LOCAL_RANK, used on Polaris) and
    Aurora's mpiexec/PALS (PMI_RANK/PMI_SIZE + PALS_LOCAL_RANKID). Returns None
    for a plain single-process launch.
    """
    if os.environ.get("RANK") is not None and os.environ.get("WORLD_SIZE") is not None:
        return (
            int(os.environ["RANK"]),
            int(os.environ["WORLD_SIZE"]),
            int(os.environ.get("LOCAL_RANK", 0)),
        )
    if os.environ.get("PMI_RANK") is not None and os.environ.get("PMI_SIZE") is not None:
        rank = int(os.environ["PMI_RANK"])
        world = int(os.environ["PMI_SIZE"])
        local = int(
            os.environ.get("PALS_LOCAL_RANKID")
            or os.environ.get("MPI_LOCALRANKID")
            or 0
        )
        # torch's init_method="env://" reads RANK/WORLD_SIZE from os.environ;
        # mpiexec/PALS only sets PMI_* so mirror them here. MASTER_ADDR/PORT are
        # exported by the PBS launcher before mpiexec (single-node rendezvous).
        os.environ.setdefault("RANK", str(rank))
        os.environ.setdefault("WORLD_SIZE", str(world))
        os.environ.setdefault("LOCAL_RANK", str(local))
        return rank, world, local
    return None
