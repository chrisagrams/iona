"""Typed settings read from the environment.

Every value Iona takes from the environment is a field on one of these classes;
no other module reads the environment. Iona's own variables carry an ``IONA_``
prefix (``IONA_ARGS_FILE`` fills ``args_file``); variables owned by a scheduler,
launcher, or library keep their names through validation aliases.

A :class:`Platform` is built once per job: it validates the job's inputs,
prepares the node (mounts, checks), and launches training ranks. It hands itself
to child processes as ``NAME=value`` pairs (:meth:`IonaSettings.to_env`), so a
child rebuilds the same settings without recomputing job-wide values such as the
walltime deadline. :class:`RankEnv` is what each training process reads.
"""

from __future__ import annotations

import shutil
import socket
import subprocess
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Annotated, Any, ClassVar, Literal

from pydantic import AliasChoices, BeforeValidator, Field, field_validator, model_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict


def _split_commas(value: Any) -> Any:
    if isinstance(value, str):
        return [item for item in value.split(",") if item]
    return value


# Comma-separated in the environment ("0,1,2"), a list in Python.
CommaList = Annotated[list[int], NoDecode, BeforeValidator(_split_commas)]
CommaStrList = Annotated[list[str], NoDecode, BeforeValidator(_split_commas)]


def _format_env(value: Any) -> str:
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, (list, tuple)):
        return ",".join(str(item) for item in value)
    return str(value)


class IonaSettings(BaseSettings):
    """Settings read from ``IONA_*`` variables, serializable for child processes."""

    model_config = SettingsConfigDict(
        env_prefix="IONA_", case_sensitive=False, validate_by_name=True
    )

    def to_env(self) -> dict[str, str]:
        """Return ``NAME=value`` pairs that rebuild these settings in a child process."""
        prefix = self.model_config.get("env_prefix", "")
        env = {}
        for name, field in type(self).model_fields.items():
            value = getattr(self, name)
            if value is None:
                continue
            alias = field.validation_alias
            if isinstance(alias, AliasChoices):
                alias = alias.choices[0]
            env_name = alias if isinstance(alias, str) else f"{prefix}{name}".upper()
            env[env_name] = _format_env(value)
        return env


class RankEnv(IonaSettings):
    """Values a training or probe process reads from its environment."""

    rank: int = Field(0, validation_alias="RANK")
    world_size: int = Field(1, validation_alias="WORLD_SIZE")
    local_rank: int | None = Field(None, validation_alias="LOCAL_RANK")
    local_world_size: int = Field(1, validation_alias="LOCAL_WORLD_SIZE")
    job_deadline_epoch: float | None = None
    checkpoint_margin_seconds: float = Field(900, ge=0)
    xpu_metrics_url: str | None = None
    wandb_run_id: str | None = Field(None, validation_alias="WANDB_RUN_ID")
    wandb_dir: Path | None = Field(None, validation_alias="WANDB_DIR")


class Platform(IonaSettings):
    """Job-wide inputs and how a platform launches training ranks.

    Subclasses fill platform defaults in :meth:`resolve` and override the hooks
    below to prepare the node and launch ranks.
    """

    repo_dir: Path = Field(default_factory=Path.cwd)
    args_file: Path
    log_root: Path
    log_dir: Path | None = None
    checkpoint_dir: Path | None = None
    hf_home: Path | None = None
    dataset_cache_dir: Path | None = None
    preprocessed_dataset_dir: Path | None = None
    preprocessed_probe_dir: Path | None = None
    hf_offline: bool = True
    run_name: str | None = None
    resume_from_checkpoint: str | None = None
    devices_per_host: int = Field(1, ge=1)
    micro_batch_size: int = Field(64, ge=1)
    global_batch_size: int = Field(512, ge=1)
    probe_execution: Literal["inline", "sidecar", "off"] | None = None
    checkpoint_margin_seconds: int = Field(900, ge=0)
    job_deadline_epoch: int | None = None
    hosts: CommaStrList = []
    master_port: int = 29500
    tmpdir: Path = Path("/tmp")
    omp_num_threads: int = Field(4, ge=1)
    wandb_run_id: str | None = Field(None, validation_alias="WANDB_RUN_ID")

    @field_validator("log_root", "log_dir")
    @classmethod
    def _absolute(cls, path: Path | None) -> Path | None:
        if path is not None and not path.is_absolute():
            raise ValueError("must be an absolute shared-filesystem path")
        return path

    @model_validator(mode="after")
    def _resolve(self) -> Platform:
        self.resolve()
        return self

    def resolve(self) -> None:
        """Fill defaults that depend on other fields; raise ValueError if invalid.

        Subclasses set their own defaults and call ``super().resolve()``.
        """
        if not self.args_file.is_absolute():
            self.args_file = self.repo_dir / self.args_file
        if self.checkpoint_dir is None:
            raise ValueError("checkpoint_dir is required")
        if not self.checkpoint_dir.is_absolute():
            self.checkpoint_dir = self.repo_dir / self.checkpoint_dir
        if self.hf_home is None:
            raise ValueError("hf_home is required")
        self.hosts = self.hosts or self.default_hosts()
        self.dataset_cache_dir = self.dataset_cache_dir or self.hf_home / "datasets"
        self.preprocessed_dataset_dir = (
            self.preprocessed_dataset_dir or self.dataset_cache_dir / "msdelta-preprocessed"
        )
        self.preprocessed_probe_dir = (
            self.preprocessed_probe_dir or self.dataset_cache_dir / "msdelta-probes"
        )

    def default_hosts(self) -> list[str]:
        return [socket.gethostname()]

    @property
    def job_id(self) -> str:
        return "local"

    def prepare(self) -> None:
        """Run one-time job setup and checks that need the node, such as mounts."""
        if not self.args_file.is_file():
            raise FileNotFoundError(f"arguments file not found: {self.args_file}")

    def runtime_env(self) -> dict[str, str]:
        """Variables that libraries in every rank read (HF, threading, temp files)."""
        env = {
            "OMP_NUM_THREADS": str(self.omp_num_threads),
            "MKL_NUM_THREADS": str(self.omp_num_threads),
            "OPENBLAS_NUM_THREADS": str(self.omp_num_threads),
            # Multiprocessing binds AF_UNIX sockets under TMPDIR, which must stay well
            # under the 107-byte socket path limit; a scheduler's TMPDIR can exceed it.
            "TMPDIR": str(self.tmpdir),
            "HF_HOME": str(self.hf_home),
            "HF_HUB_CACHE": str(self.hf_home / "hub"),
            "HF_DATASETS_CACHE": str(self.dataset_cache_dir),
        }
        if self.hf_offline:
            env["HF_HUB_OFFLINE"] = "1"
            env["HF_DATASETS_OFFLINE"] = "1"
        return env

    def runtime_unset(self) -> list[str]:
        """Variables to remove from every rank's environment."""
        return []

    def train_args(self, probe_execution: str) -> list[str]:
        """Platform-specific iona.train arguments, such as the DDP backend."""
        return []

    def rank_command(self, hosts: Sequence[str], node_list: Path, argv: list[str]) -> list[str]:
        """Return a command that runs ``argv`` once per device on ``hosts``."""
        raise NotImplementedError(f"{type(self).__name__} cannot launch ranks")


class PBSPlatform(Platform):
    """A PBS job: hosts, job ID, and walltime deadline come from the scheduler."""

    repo_dir: Path = Field(
        default_factory=Path.cwd, validation_alias=AliasChoices("IONA_REPO_DIR", "PBS_O_WORKDIR")
    )
    pbs_nodefile: Path | None = Field(None, validation_alias="PBS_NODEFILE")
    pbs_jobid: str = Field("local", validation_alias="PBS_JOBID")
    job_walltime_seconds: int | None = Field(None, ge=1)

    def resolve(self) -> None:
        super().resolve()
        if self.job_deadline_epoch is None:
            walltime = self.job_walltime_seconds or _qstat_walltime_seconds(self.pbs_jobid)
            if self.checkpoint_margin_seconds >= walltime:
                raise ValueError("checkpoint_margin_seconds must be less than the walltime")
            self.job_deadline_epoch = int(time.time()) + walltime

    def default_hosts(self) -> list[str]:
        if self.pbs_nodefile is None:
            return super().default_hosts()
        return list(dict.fromkeys(self.pbs_nodefile.read_text().split()))

    @property
    def job_id(self) -> str:
        return self.pbs_jobid.split(".", 1)[0]


def _qstat_walltime_seconds(job_id: str) -> int:
    output = subprocess.run(
        ["qstat", "-f", job_id], check=True, capture_output=True, text=True
    ).stdout
    for line in output.splitlines():
        key, _, value = line.strip().partition(" = ")
        if key == "Resource_List.walltime":
            hours, minutes, seconds = (int(part) for part in value.split(":"))
            return hours * 3600 + minutes * 60 + seconds
    raise ValueError(f"could not find Resource_List.walltime for PBS job {job_id}")


class AuroraPlatform(PBSPlatform):
    """ALCF Aurora: Intel XPU tiles, PALS mpiexec, oneCCL, and DAOS.

    The launch follows the ALCF Aurora PyTorch DDP guidance:
    https://docs.alcf.anl.gov/aurora/data-science/frameworks/pytorch/
    """

    # Tiles and CPU cores per rank for the supported per-host layouts. With 8 ranks,
    # tiles 4, 5, 10 and 11 stay free for posttraining sidecars.
    TRAINING_TILES: ClassVar[dict[int, list[int]]] = {
        8: [0, 1, 2, 3, 6, 7, 8, 9],
        12: list(range(12)),
    }
    CPU_BIND: ClassVar[dict[int, str]] = {
        8: "verbose,list:4-7:8-11:12-15:16-19:56-59:60-63:64-67:68-71",
        12: "verbose,list:4-7:8-11:12-15:16-19:20-23:24-27:56-59:60-63:64-67:68-71:72-75:76-79",
    }
    NO_PROXY: ClassVar[str] = (
        "admin,localhost,*.cm.aurora.alcf.anl.gov,aurora-*,*.aurora.alcf.anl.gov,"
        "*.alcf.anl.gov,127.0.0.1"
    )

    devices_per_host: int = Field(8, ge=1, le=12)
    daos_pool: str
    daos_cont: str = "msdelta-training"
    training_tiles: CommaList = []
    denoise_xpu_tiles: CommaList = [4, 5]
    retrieval_xpu_tiles: CommaList = [10, 11]
    cpu_bind: str | None = None
    telegraf_bin: Path | None = None
    xpu_telegraf_config: Path | None = None
    telegraf_startup_seconds: int = Field(120, ge=1)
    xpu_metrics_port: int = 9274
    proxy: str = "http://proxy.alcf.anl.gov:3128"

    def resolve(self) -> None:
        self.checkpoint_dir = self.checkpoint_dir or self.daos_root / "checkpoints"
        self.hf_home = self.hf_home or self.daos_root / "hf-cache"
        super().resolve()
        self.training_tiles = self.training_tiles or self.TRAINING_TILES.get(
            self.devices_per_host, list(range(self.devices_per_host))
        )
        if len(self.training_tiles) != self.devices_per_host:
            raise ValueError("training_tiles must list one tile per device on a host")
        self.cpu_bind = self.cpu_bind or self.CPU_BIND.get(self.devices_per_host, "none")
        self.xpu_telegraf_config = (
            self.xpu_telegraf_config or self.repo_dir / "pbs" / "xpu-telegraf.conf"
        )
        if self.telegraf_bin is None and (found := shutil.which("telegraf")):
            self.telegraf_bin = Path(found)

    def train_args(self, probe_execution: str) -> list[str]:
        args = ["--ddp_backend", "xccl"]
        if probe_execution == "sidecar":
            self._check_sidecar_tiles()
            args += [
                "--sidecar_launcher",
                str(self.repo_dir / "pbs" / "aurora-probe.sh"),
                "--sidecar_denoise_device",
                f"xpu:{_format_env(self.denoise_xpu_tiles)}",
                "--sidecar_retrieval_device",
                f"xpu:{_format_env(self.retrieval_xpu_tiles)}",
            ]
        return args

    def _check_sidecar_tiles(self) -> None:
        used = set(self.training_tiles)
        for kind, tiles in (
            ("denoise", self.denoise_xpu_tiles),
            ("retrieval", self.retrieval_xpu_tiles),
        ):
            for tile in tiles:
                if not 0 <= tile <= 11:
                    raise ValueError(f"{kind} sidecar tile {tile} is not between 0 and 11")
                if tile in used:
                    raise ValueError(f"{kind} sidecar tile {tile} overlaps another assignment")
                used.add(tile)

    @property
    def daos_root(self) -> Path:
        return Path("/tmp") / self.daos_pool / self.daos_cont

    def prepare(self) -> None:
        super().prepare()
        subprocess.run(
            ["launch-dfuse-with-caching.sh", f"{self.daos_pool}:{self.daos_cont}"], check=True
        )
        if not self.daos_root.is_dir():
            raise FileNotFoundError(f"DAOS dfuse mount not found: {self.daos_root}")
        if self.preprocessed_dataset_dir is None or not self.preprocessed_dataset_dir.is_dir():
            raise FileNotFoundError(
                f"preprocessed dataset not found: {self.preprocessed_dataset_dir}"
            )
        if self.telegraf_bin is not None and not self.telegraf_bin.is_file():
            raise FileNotFoundError(f"telegraf binary not found: {self.telegraf_bin}")
        if self.xpu_telegraf_config is None or not self.xpu_telegraf_config.is_file():
            raise FileNotFoundError(
                f"XPU Telegraf configuration not found: {self.xpu_telegraf_config}"
            )
        self._check_visible_tiles()

    def _check_visible_tiles(self) -> None:
        command = [
            "env",
            *(f"{name}={value}" for name, value in self._device_env().items()),
            sys.executable,
            "-c",
            "import torch; print(torch.xpu.device_count()); "
            "print(f'torch={torch.__version__} xccl={torch.distributed.is_xccl_available()}')",
        ]
        visible, versions = subprocess.run(
            command, check=True, capture_output=True, text=True
        ).stdout.splitlines()
        print(versions, flush=True)
        if int(visible) != self.devices_per_host:
            raise RuntimeError(
                f"expected {self.devices_per_host} visible XPU tiles, found {visible}"
            )

    def _device_env(self) -> dict[str, str]:
        return {
            "ZE_AFFINITY_MASK": _format_env(self.training_tiles),
            "ZE_FLAT_DEVICE_HIERARCHY": "FLAT",
            "ONEAPI_DEVICE_SELECTOR": "level_zero:gpu",
        }

    def runtime_env(self) -> dict[str, str]:
        return {
            **super().runtime_env(),
            **self._device_env(),
            "CCL_PROCESS_LAUNCHER": "pmix",
            "CCL_ATL_TRANSPORT": "mpi",
            "FI_MR_CACHE_MONITOR": "userfaultfd",
            **dict.fromkeys(
                ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy", "ftp_proxy"), self.proxy
            ),
            "no_proxy": self.NO_PROXY,
            "NO_PROXY": self.NO_PROXY,
        }

    def runtime_unset(self) -> list[str]:
        return ["CCL_ZE_IPC", "CCL_ZE_IPC_EXCHANGE", "SYCL_DEVICE_FILTER"]

    def rank_command(self, hosts: Sequence[str], node_list: Path, argv: list[str]) -> list[str]:
        return [
            "mpiexec",
            "--verbose",
            "--envall",
            "--pmi=pmix",
            "--no-vni",
            "-n",
            str(len(hosts) * self.devices_per_host),
            "--ppn",
            str(self.devices_per_host),
            "--cpu-bind",
            str(self.cpu_bind),
            f"--hostfile={node_list}",
            "bash",
            str(self.repo_dir / "pbs" / "aurora-rank.sh"),
            *argv,
        ]


PLATFORMS: dict[str, type[Platform]] = {"aurora": AuroraPlatform}
