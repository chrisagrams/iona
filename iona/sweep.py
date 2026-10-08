"""Run W&B sweep trials on disjoint node groups within one PBS allocation.

Each Parsl task claims ``--nodes_per_run`` whole nodes and runs one
``wandb agent --count 1`` there; the sweep's command launches
``pbs/aurora-run.sh`` on the task's nodes. Submit from ``pbs/aurora-sweep.pbs``.
"""

from __future__ import annotations

import argparse
import logging
import os
import time
from concurrent.futures import FIRST_COMPLETED, Future, wait
from pathlib import Path

import parsl
import wandb
from parsl import bash_app
from parsl.config import Config
from parsl.executors import MPIExecutor
from parsl.launchers import SimpleLauncher
from parsl.providers import LocalProvider

logger = logging.getLogger(__name__)


@bash_app
def sweep_trial(
    wandb_bin: str,
    sweep_id: str,
    repo_dir: str,
    stdout: str | None = None,
    stderr: str | None = None,
    parsl_resource_specification: dict | None = None,
) -> str:
    # Parsl exports the nodes claimed for this task as PARSL_MPI_NODELIST.
    return (
        f'cd {repo_dir} && IONA_HOSTS="$PARSL_MPI_NODELIST" {wandb_bin} agent --count 1 {sweep_id}'
    )


def sweep_running(sweep_id: str) -> bool:
    try:
        return wandb.Api().sweep(sweep_id).state.upper() == "RUNNING"
    except Exception:
        # An agent exits promptly on a finished sweep, so keep going on API errors.
        logger.exception("Could not read state of sweep %s", sweep_id)
        return True


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sweep_id", required=True, help="entity/project/sweep_id")
    parser.add_argument("--nodes_per_run", type=int, required=True)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    # Parsl writes its own log under run_dir; keep it out of the job output.
    logging.getLogger("parsl").propagate = False

    repo_dir = os.environ["REPO_DIR"]
    xpus_per_host = int(os.environ["XPUS_PER_HOST"])
    run_dir = Path(os.environ["LOG_ROOT"]) / "sweeps" / os.environ["JOB_ID"]
    run_dir.mkdir(parents=True, exist_ok=True)

    # Parsl hands out nodes from PBS_NODEFILE; give it each host exactly once.
    hosts = os.environ["JOB_HOSTS"].split(",")
    nodefile = run_dir / "nodes"
    nodefile.write_text("".join(f"{host}\n" for host in hosts))
    os.environ["PBS_NODEFILE"] = str(nodefile)

    slots = len(hosts) // args.nodes_per_run
    if slots < 1:
        parser.error(f"--nodes_per_run {args.nodes_per_run} exceeds the {len(hosts)} job nodes")
    if idle := len(hosts) - slots * args.nodes_per_run:
        logger.warning("%d nodes will stay idle with --nodes_per_run %d", idle, args.nodes_per_run)

    deadline = int(os.environ["IONA_JOB_DEADLINE_EPOCH"]) - int(
        os.environ["IONA_CHECKPOINT_MARGIN_SECONDS"]
    )
    parsl.load(
        Config(
            run_dir=str(run_dir / "parsl"),
            executors=[
                MPIExecutor(
                    label="trials",
                    address="127.0.0.1",
                    mpi_launcher="mpiexec",
                    max_workers_per_block=slots,
                    provider=LocalProvider(
                        launcher=SimpleLauncher(),
                        init_blocks=1,
                        min_blocks=1,
                        max_blocks=1,
                    ),
                )
            ],
        )
    )
    resource_specification = {
        "num_nodes": args.nodes_per_run,
        "ranks_per_node": xpus_per_host,
        "num_ranks": args.nodes_per_run * xpus_per_host,
    }
    wandb_bin = str(Path(os.environ["VENV_DIR"]) / "bin" / "wandb")

    pending: set[Future] = set()
    while True:
        # Past the deadline a new trial would only preempt itself on its first step.
        while len(pending) < slots and time.time() < deadline and sweep_running(args.sweep_id):
            pending.add(
                sweep_trial(
                    wandb_bin,
                    args.sweep_id,
                    repo_dir,
                    stdout=parsl.AUTO_LOGNAME,
                    stderr=parsl.AUTO_LOGNAME,
                    parsl_resource_specification=resource_specification,
                )
            )
        if not pending:
            break
        done, pending = wait(pending, return_when=FIRST_COMPLETED)
        for future in done:
            if error := future.exception():
                logger.error("Trial failed: %s", error)
    parsl.dfk().cleanup()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
