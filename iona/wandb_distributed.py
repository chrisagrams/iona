"""Initialize W&B consistently for single- and multi-node training."""

from __future__ import annotations

import os
import socket
from typing import Any

import wandb


def init_wandb_run(
    *,
    project: str,
    run_name: str,
    config: dict[str, Any],
    shared: bool = False,
    role: str = "pretrain",
    run_id: str | None = None,
    entity: str | None = None,
) -> wandb.Run | None:
    """Create one W&B client per node, sharing a run across multiple nodes."""
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    local_world_size = int(os.environ.get("LOCAL_WORLD_SIZE", "1"))

    # A single client can monitor every GPU visible on its physical host.
    if local_rank != 0:
        return None

    is_multinode = world_size > local_world_size
    is_primary = rank == 0 and role == "pretrain"
    xpu_metrics_url = os.environ.get("IONA_XPU_METRICS_URL")
    run_config = (
        {
            **config,
            "distributed": {
                "num_nodes": max(1, world_size // local_world_size),
                "world_size": world_size,
                "gpus_per_node": local_world_size,
                "total_gpus": world_size,
            },
        }
        if is_primary
        else None
    )

    settings_kwargs: dict[str, Any] = {}
    # Pretraining monitors all host XPUs, including the sidecar devices. Sidecars
    # inherit this endpoint but would duplicate its metrics on their own timelines.
    if xpu_metrics_url and role == "pretrain":
        settings_kwargs["x_stats_open_metrics_endpoints"] = {"xpu": xpu_metrics_url}
    if is_multinode or shared or role != "pretrain":
        run_id = run_id or os.environ.get("WANDB_RUN_ID")
        if not run_id:
            raise RuntimeError("WANDB_RUN_ID is required for shared W&B logging")
        settings_kwargs.update(
            mode="shared",
            x_label=f"{socket.gethostname()}-{role}"
            if shared or role != "pretrain"
            else socket.gethostname(),
            x_primary=is_primary,
            x_update_finish_state=is_primary,
            x_stats_gpu_device_ids=list(range(local_world_size)),
        )
    settings = wandb.Settings(**settings_kwargs) if settings_kwargs else None

    return wandb.init(
        project=project,
        id=run_id,
        entity=entity,
        name=run_name if is_primary else None,
        config=run_config,
        settings=settings,
    )
