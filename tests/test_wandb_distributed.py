"""Tests for node-aware W&B initialization."""

from __future__ import annotations

import os
import unittest
from unittest.mock import MagicMock, patch

from msdelta.wandb_distributed import init_wandb_run


class InitWandbRunTest(unittest.TestCase):
    def test_non_node_leader_does_not_initialize_wandb(self):
        env = {
            "RANK": "1",
            "WORLD_SIZE": "8",
            "LOCAL_RANK": "1",
            "LOCAL_WORLD_SIZE": "4",
            "WANDB_RUN_ID": "test-run",
        }
        with (
            patch.dict(os.environ, env, clear=True),
            patch("msdelta.wandb_distributed.wandb.init") as init,
        ):
            self.assertIsNone(init_wandb_run(project="project", run_name="run"))
            init.assert_not_called()

    def test_primary_uses_shared_mode_and_logs_topology(self):
        env = {
            "RANK": "0",
            "WORLD_SIZE": "8",
            "LOCAL_RANK": "0",
            "LOCAL_WORLD_SIZE": "4",
            "WANDB_RUN_ID": "test-run",
            "MSDELTA_NUM_HOSTS": "2",
            "MSDELTA_TOTAL_GPUS": "8",
        }
        run = MagicMock()
        settings = MagicMock()
        with (
            patch.dict(os.environ, env, clear=True),
            patch("msdelta.wandb_distributed.socket.gethostname", return_value="node-a"),
            patch(
                "msdelta.wandb_distributed.wandb.Settings", return_value=settings
            ) as make_settings,
            patch("msdelta.wandb_distributed.wandb.init", return_value=run) as init,
        ):
            self.assertIs(init_wandb_run(project="project", run_name="run"), run)

        make_settings.assert_called_once_with(
            mode="shared",
            x_label="node-a",
            x_primary=True,
            x_update_finish_state=True,
            x_stats_gpu_device_ids=[0, 1, 2, 3],
        )
        init.assert_called_once_with(project="project", name="run", settings=settings)
        run.config.update.assert_called_once_with(
            {
                "distributed/num_nodes": 2,
                "distributed/world_size": 8,
                "distributed/gpus_per_node": 4,
                "distributed/total_gpus": 8,
            },
            allow_val_change=True,
        )

    def test_worker_cannot_finish_shared_run(self):
        env = {
            "RANK": "4",
            "WORLD_SIZE": "8",
            "LOCAL_RANK": "0",
            "LOCAL_WORLD_SIZE": "4",
            "WANDB_RUN_ID": "test-run",
        }
        run = MagicMock()
        settings = MagicMock()
        with (
            patch.dict(os.environ, env, clear=True),
            patch("msdelta.wandb_distributed.socket.gethostname", return_value="node-b"),
            patch(
                "msdelta.wandb_distributed.wandb.Settings", return_value=settings
            ) as make_settings,
            patch("msdelta.wandb_distributed.wandb.init", return_value=run) as init,
        ):
            self.assertIs(init_wandb_run(project="project", run_name="run"), run)

        make_settings.assert_called_once_with(
            mode="shared",
            x_label="node-b",
            x_primary=False,
            x_update_finish_state=False,
            x_stats_gpu_device_ids=[0, 1, 2, 3],
        )
        init.assert_called_once_with(project="project", settings=settings)
        run.config.update.assert_not_called()

    def test_multinode_requires_preconfigured_run_id(self):
        env = {
            "RANK": "0",
            "WORLD_SIZE": "8",
            "LOCAL_RANK": "0",
            "LOCAL_WORLD_SIZE": "4",
        }
        with patch.dict(os.environ, env, clear=True):
            with self.assertRaisesRegex(RuntimeError, "WANDB_RUN_ID"):
                init_wandb_run(project="project", run_name="run")

    def test_single_node_keeps_standard_wandb_mode(self):
        env = {
            "RANK": "0",
            "WORLD_SIZE": "4",
            "LOCAL_RANK": "0",
            "LOCAL_WORLD_SIZE": "4",
        }
        run = MagicMock()
        with (
            patch.dict(os.environ, env, clear=True),
            patch("msdelta.wandb_distributed.wandb.Settings") as make_settings,
            patch("msdelta.wandb_distributed.wandb.init", return_value=run) as init,
        ):
            self.assertIs(init_wandb_run(project="project", run_name="run"), run)

        make_settings.assert_not_called()
        init.assert_called_once_with(project="project", name="run")


if __name__ == "__main__":
    unittest.main()
