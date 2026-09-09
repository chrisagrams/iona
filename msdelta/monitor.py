"""Stream Intel XPU telemetry from ``xpu-smi`` to W&B."""

from __future__ import annotations

import csv
import shutil
import subprocess
import threading
from collections.abc import Iterable
from typing import Any, TextIO

from wandb.sdk.interface.interface_shared import InterfaceShared

import wandb

_METRICS = {
    "GPU Utilization (%)": "gpu",
    "GPU Power (W)": "powerWatts",
    "GPU Frequency (MHz)": "smClock",
    "GPU Memory Utilization (%)": "memory",
}


def _parse_value(value: str) -> float | None:
    value = value.strip()
    if not value or value.upper() == "N/A":
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _parse_row(row: dict[str, str]) -> tuple[str, dict[str, float]] | None:
    """Convert one xpu-smi CSV row into a timestamp and W&B payload."""
    device_id = row.get("DeviceId", "").strip()
    timestamp = row.get("Timestamp", "").strip()
    if not device_id or not timestamp:
        return None

    payload: dict[str, float] = {}
    for column, metric in _METRICS.items():
        value = _parse_value(row.get(column, ""))
        if value is not None:
            payload[f"gpu.{device_id}.{metric}"] = value
    return timestamp, payload


class XpuSmiWandbMonitor:
    """Manage a background ``xpu-smi dump`` process for one physical node."""

    def __init__(self, run: wandb.Run, device_ids: Iterable[int] | None = None):
        self.run = run
        # xpu-smi uses physical device IDs, not PyTorch's flattened tile IDs.
        self.device_ids = list(device_ids) if device_ids is not None else [-1]
        self._process: subprocess.Popen[str] | None = None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._logged_sample = False

    def start(self) -> bool:
        """Start collection, returning false when xpu-smi cannot be used."""
        executable = shutil.which("xpu-smi")
        if executable is None:
            print("[xpu-metrics] xpu-smi not found; XPU metrics are disabled", flush=True)
            return False
        if not self.device_ids:
            return False

        command = [
            executable,
            "dump",
            "-d",
            *(str(device_id) for device_id in self.device_ids),
            "-m",
            "0",
            "1",
            "2",
            "5",
        ]
        try:
            self._process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                # Keep CLI diagnostics in the job log, outside the CSV stream.
                stderr=None,
                text=True,
                bufsize=1,
            )
        except OSError as error:
            print(f"[xpu-metrics] could not start xpu-smi: {error}", flush=True)
            return False

        self._thread = threading.Thread(target=self._collect, name="xpu-smi-wandb", daemon=True)
        print(f"[xpu-metrics] starting: {' '.join(command)}", flush=True)
        self._thread.start()
        return True

    def _collect(self) -> None:
        process = self._process
        if process is None or process.stdout is None:
            return
        try:
            self._read_samples(process.stdout)
            if not self._stop.is_set():
                returncode = process.wait(timeout=5)
                print(
                    f"[xpu-metrics] xpu-smi exited with code {returncode}; "
                    f"logged samples: {self._logged_sample}",
                    flush=True,
                )
        except Exception as error:
            if not self._stop.is_set():
                print(f"[xpu-metrics] collector stopped: {error}", flush=True)

    def _read_samples(self, stream: TextIO) -> None:
        reader = csv.DictReader(stream, skipinitialspace=True)
        if reader.fieldnames is None:
            return
        reader.fieldnames = [column.strip() for column in reader.fieldnames]
        if not {"Timestamp", "DeviceId"}.issubset(reader.fieldnames) or not any(
            column in reader.fieldnames for column in _METRICS
        ):
            raise ValueError(f"unexpected xpu-smi CSV header: {reader.fieldnames!r}")
        sample_timestamp: str | None = None
        sample: dict[str, Any] = {}

        for row in reader:
            if self._stop.is_set():
                break
            parsed = _parse_row(row)
            if parsed is None:
                continue
            timestamp, metrics = parsed
            if sample_timestamp is not None and timestamp != sample_timestamp and sample:
                self._log_sample(sample)
                sample = {}
            sample_timestamp = timestamp
            sample.update(metrics)

        if sample and not self._stop.is_set():
            self._log_sample(sample)

    def _log_sample(self, sample: dict[str, Any]) -> None:
        interface = self.run._interface
        if interface is None:
            raise RuntimeError("W&B run has no active interface")
        if not isinstance(interface, InterfaceShared):
            raise TypeError(f"unsupported W&B interface: {type(interface).__name__}")
        interface.publish_stats(sample)
        if not self._logged_sample:
            self._logged_sample = True
            print("[xpu-metrics] first sample logged to W&B System metrics", flush=True)

    def stop(self) -> None:
        """Stop collection before the associated W&B run is finished."""
        self._stop.set()
        process = self._process
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
        if self._thread is not None:
            self._thread.join(timeout=5)

    def __enter__(self) -> XpuSmiWandbMonitor:
        self.start()
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.stop()
