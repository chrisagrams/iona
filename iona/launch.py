"""Launch Iona pretraining from a batch job on a platform described by :mod:`iona.env`.

Run after the platform's modules are loaded, for example from pbs/aurora-pretrain.pbs:

    python -m iona.launch --platform aurora pretrain [extra iona.train arguments]
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from iona.env import PLATFORMS, Platform

# iona.train arguments the launcher sets itself, dropped from the copied args file.
_LAUNCHER_ARGS = ("--deepspeed", "--logging_nan_inf_filter")


def read_args_file(path: Path) -> dict[str, str]:
    """Return the last value of each ``--name value`` pair in an args file."""
    tokens = path.read_text().split()
    return {
        token: tokens[index + 1]
        for index, token in enumerate(tokens[:-1])
        if token.startswith("--") and not tokens[index + 1].startswith("--")
    }


def latest_checkpoint(run_dir: Path) -> Path | None:
    """Return the highest complete ``checkpoint-N`` directory in ``run_dir``."""
    candidates = []
    for path in run_dir.glob("checkpoint-*"):
        match = re.fullmatch(r"checkpoint-(\d+)", path.name)
        if match and path.is_dir() and (path / "trainer_state.json").is_file():
            candidates.append((int(match.group(1)), path))
    return max(candidates)[1] if candidates else None


def gradient_accumulation_steps(global_batch: int, micro_batch: int, world_size: int) -> int:
    """Return the accumulation that turns per-device micro-batches into ``global_batch``."""
    per_step = micro_batch * world_size
    if global_batch % per_step:
        raise ValueError(
            f"global batch size {global_batch} is not divisible by "
            f"{micro_batch} micro-batch x {world_size} devices"
        )
    if global_batch < per_step:
        raise ValueError(f"global batch size {global_batch} is smaller than one micro-batch step")
    return global_batch // per_step


@dataclass(frozen=True)
class RunPlan:
    """Everything specific to one training run on a set of hosts."""

    name: str
    hosts: list[str]
    run_dir: Path
    log_dir: Path
    wandb_run_id: str
    probe_execution: str
    gradient_accumulation_steps: int
    resume_from_checkpoint: Path | None

    @classmethod
    def create(cls, platform: Platform, hosts: list[str]) -> RunPlan:
        values = read_args_file(platform.args_file)
        name = platform.run_name or (
            f"{values.get('--run_name', platform.args_file.parent.name)}-{platform.job_id}"
        )
        run_dir = platform.checkpoint_dir / name
        probe_execution = platform.probe_execution or values.get("--probe_execution", "sidecar")
        if probe_execution != "off":
            _check_probe_data(platform, values)

        resume = None
        if platform.resume_from_checkpoint == "latest":
            resume = latest_checkpoint(run_dir)
            if resume is None:
                raise FileNotFoundError(f"no complete checkpoint-N directories found in {run_dir}")
        elif platform.resume_from_checkpoint:
            resume = Path(platform.resume_from_checkpoint)

        return cls(
            name=name,
            hosts=hosts,
            run_dir=run_dir,
            log_dir=platform.log_dir or platform.log_root / name / "logs",
            wandb_run_id=platform.wandb_run_id or name,
            probe_execution=probe_execution,
            gradient_accumulation_steps=gradient_accumulation_steps(
                platform.global_batch_size,
                platform.micro_batch_size,
                len(hosts) * platform.devices_per_host,
            ),
            resume_from_checkpoint=resume,
        )

    def env(self, platform: Platform) -> dict[str, str]:
        """Variables this run's ranks and the rank wrapper read."""
        return {
            "MASTER_ADDR": self.hosts[0],
            "MASTER_PORT": str(platform.master_port),
            "IONA_WORLD_SIZE": str(len(self.hosts) * platform.devices_per_host),
            "LOG_DIR": str(self.log_dir),
            "WANDB_DIR": str(self.log_dir),
            "WANDB_RESUME": "allow",
            "WANDB_RUN_ID": self.wandb_run_id,
        }

    def train_args(self, platform: Platform) -> list[str]:
        args_file = self.run_dir / "training-ddp.args"
        kept = [
            line
            for line in platform.args_file.read_text().splitlines()
            if not line.split() or line.split()[0] not in _LAUNCHER_ARGS
        ]
        args_file.write_text("\n".join([*kept, "--logging_nan_inf_filter false", ""]))
        args = [
            "--args_file",
            str(args_file),
            "--dataset_cache_dir",
            str(platform.dataset_cache_dir),
            "--preprocessed_dataset_dir",
            str(platform.preprocessed_dataset_dir),
            "--preprocessed_probe_dir",
            str(platform.preprocessed_probe_dir),
            "--run_name",
            self.name,
            "--output_dir",
            str(self.run_dir),
            "--per_device_train_batch_size",
            str(platform.micro_batch_size),
            "--gradient_accumulation_steps",
            str(self.gradient_accumulation_steps),
            "--probe_execution",
            self.probe_execution,
            *platform.train_args(self.probe_execution),
        ]
        if self.resume_from_checkpoint is not None:
            args += ["--resume_from_checkpoint", str(self.resume_from_checkpoint)]
        return args


def _check_probe_data(platform: Platform, values: dict[str, str]) -> None:
    probe_dir = platform.preprocessed_probe_dir
    required = []
    if int(values.get("--denoise_steps", "0")) > 0:
        required.append("denoise")
    if int(values.get("--retrieval_steps", "0")) > 0:
        required += ["retrieval", "retrieval-evaluation"]
    for kind in required:
        if probe_dir is None or not (probe_dir / kind / "dataset_dict.json").is_file():
            raise FileNotFoundError(
                f"finalized {kind} probes not found in {probe_dir}; run aurora-preprocess.pbs"
            )


def launch(platform: Platform, plan: RunPlan, extra_args: list[str]) -> int:
    """Run iona.train on the plan's hosts and return the launcher's exit code."""
    plan.run_dir.mkdir(parents=True, exist_ok=True)
    plan.log_dir.mkdir(parents=True, exist_ok=True)
    node_list = plan.run_dir / "nodes"
    node_list.write_text("".join(f"{host}\n" for host in plan.hosts))
    env = {**platform.runtime_env(), **platform.to_env(), **plan.env(platform)}
    command = [
        "env",
        *(arg for name in platform.runtime_unset() for arg in ("-u", name)),
        *(f"{name}={value}" for name, value in env.items()),
        *platform.rank_command(
            plan.hosts,
            node_list,
            [sys.executable, "-m", "iona.train", *plan.train_args(platform), *extra_args],
        ),
    ]
    print(
        f"{datetime.now().isoformat(timespec='seconds')} job={platform.job_id} run={plan.name} "
        f"hosts={len(plan.hosts)} devices_per_host={platform.devices_per_host} "
        f"global_batch={platform.global_batch_size} micro_batch={platform.micro_batch_size} "
        f"accumulation={plan.gradient_accumulation_steps}",
        flush=True,
    )
    print(f"checkpoints={plan.run_dir} logs={plan.log_dir}", flush=True)
    print(
        f"deadline={platform.job_deadline_epoch} resume={plan.resume_from_checkpoint}", flush=True
    )
    print("nodes:\n" + "".join(f"  {host}\n" for host in plan.hosts), end="", flush=True)
    stem = plan.log_dir / f"train-{platform.job_id}"
    with stem.with_suffix(".out").open("w") as out, stem.with_suffix(".err").open("w") as err:
        return subprocess.run(command, stdout=out, stderr=err).returncode


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--platform", choices=sorted(PLATFORMS), required=True)
    commands = parser.add_subparsers(dest="command", required=True)
    pretrain = commands.add_parser("pretrain", help="Pretrain on every host in the job.")
    pretrain.add_argument("train_args", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)

    platform = PLATFORMS[args.platform]()
    platform.prepare()
    plan = RunPlan.create(platform, platform.hosts)
    return launch(platform, plan, args.train_args)


if __name__ == "__main__":
    sys.exit(main())
