"""Initialize W&B consistently for single- and multi-node training."""

from __future__ import annotations

import os
import socket
from typing import Any

import wandb


def init_wandb_run(
    *, project: str, run_name: str, config: dict[str, Any]
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
    is_primary = rank == 0
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

    settings = None
    if is_multinode:
        if not os.environ.get("WANDB_RUN_ID"):
            raise RuntimeError("WANDB_RUN_ID must be set before launching a multi-node job")
        settings = wandb.Settings(
            mode="shared",
            x_label=socket.gethostname(),
            x_primary=is_primary,
            x_update_finish_state=is_primary,
            x_stats_gpu_device_ids=list(range(local_world_size)),
        )

    return wandb.init(
        project=project,
        name=run_name if is_primary else None,
        config=run_config,
        settings=settings,
    )
