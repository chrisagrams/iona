"""Stream Intel XPU telemetry from ``xpu-smi`` to W&B."""

from __future__ import annotations

import csv
import shutil
import subprocess
import threading
from collections.abc import Iterable
from typing import Any, TextIO

import wandb

_METRICS = {
    "GPU Utilization (%)": "gpu_utilization_percent",
    "GPU Power (W)": "gpu_power_watts",
    "GPU Frequency (MHz)": "gpu_frequency_mhz",
    "GPU Memory Utilization (%)": "gpu_memory_utilization_percent",
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
            payload[f"xpu/device_{device_id}/{metric}"] = value
    return timestamp, payload


class XpuSmiWandbMonitor:
    """Manage a background ``xpu-smi dump`` process for one physical node."""

    def __init__(self, run: wandb.Run, device_ids: Iterable[int]):
        self.run = run
        self.device_ids = list(device_ids)
        self._process: subprocess.Popen[str] | None = None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()

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
                stderr=subprocess.DEVNULL,
                text=True,
                bufsize=1,
            )
        except OSError as error:
            print(f"[xpu-metrics] could not start xpu-smi: {error}", flush=True)
            return False

        self._thread = threading.Thread(target=self._collect, name="xpu-smi-wandb", daemon=True)
        self._thread.start()
        print(f"[xpu-metrics] monitoring devices {self.device_ids}", flush=True)
        return True

    def _collect(self) -> None:
        process = self._process
        if process is None or process.stdout is None:
            return
        try:
            self._read_samples(process.stdout)
        except Exception as error:
            if not self._stop.is_set():
                print(f"[xpu-metrics] collector stopped: {error}", flush=True)

    def _read_samples(self, stream: TextIO) -> None:
        reader = csv.DictReader(stream, skipinitialspace=True)
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
                self.run.log(sample)
                sample = {}
            sample_timestamp = timestamp
            sample.update(metrics)

        if sample and not self._stop.is_set():
            self.run.log(sample)

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
