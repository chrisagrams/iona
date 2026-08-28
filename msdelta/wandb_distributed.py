"""Initialize W&B consistently for single- and multi-node training."""

from __future__ import annotations

import os
import socket

import wandb


def init_wandb_run(*, project: str, run_name: str):
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
    init_kwargs = {"project": project}

    if is_primary:
        init_kwargs["name"] = run_name

    if is_multinode:
        if not os.environ.get("WANDB_RUN_ID"):
            raise RuntimeError("WANDB_RUN_ID must be set before launching a multi-node job")
        init_kwargs["settings"] = wandb.Settings(
            mode="shared",
            x_label=socket.gethostname(),
            x_primary=is_primary,
            x_update_finish_state=is_primary,
            x_stats_gpu_device_ids=list(range(local_world_size)),
        )

    run = wandb.init(**init_kwargs)
    if is_primary:
        run.config.update(
            {
                "distributed/num_nodes": int(
                    os.environ.get(
                        "MSDELTA_NUM_HOSTS",
                        max(1, world_size // local_world_size),
                    )
                ),
                "distributed/world_size": world_size,
                "distributed/gpus_per_node": local_world_size,
                "distributed/total_gpus": int(os.environ.get("MSDELTA_TOTAL_GPUS", world_size)),
            },
            allow_val_change=True,
        )
    return run
