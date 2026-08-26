"""Preprocessing and collation for MSDelta mass spectra."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from transformers import BatchFeature, FeatureExtractionMixin


def _as_spectrum_batch(values: Any, name: str) -> tuple[list[torch.Tensor], bool]:
    """Normalize one spectrum or a batch of spectra into a tensor list."""
    if isinstance(values, torch.Tensor):
        if values.ndim == 1:
            return [values], True
        if values.ndim == 2:
            return [row for row in values], False
        raise ValueError(f"{name} must be one- or two-dimensional")
    if not isinstance(values, (list, tuple)):
        values = list(values)
    if not values:
        return [torch.empty(0)], True
    first = values[0]
    if isinstance(first, (list, tuple, torch.Tensor)) or hasattr(first, "ndim"):
        return [torch.as_tensor(row) for row in values], False
    return [torch.as_tensor(values)], True


class MSDeltaProcessor(FeatureExtractionMixin):
    """Convert raw centroided spectra into padded MSDelta model inputs."""

    model_input_names = ["mz", "log_intensity", "attention_mask"]

    def __init__(
        self,
        intensity_threshold_frac: float = 0.01,
        max_peaks: int = 150,
        padding_value: float = 0.0,
        **kwargs,
    ):
        if not 0.0 <= intensity_threshold_frac <= 1.0:
            raise ValueError("intensity_threshold_frac must be in [0, 1]")
        if max_peaks <= 0:
            raise ValueError("max_peaks must be positive")
        super().__init__(
            intensity_threshold_frac=intensity_threshold_frac,
            max_peaks=max_peaks,
            padding_value=padding_value,
            **kwargs,
        )
        self.intensity_threshold_frac = intensity_threshold_frac
        self.max_peaks = max_peaks
        self.padding_value = padding_value

    def _process_one(
        self, mz: torch.Tensor, intensity: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mz = torch.as_tensor(mz, dtype=torch.float32)
        intensity = torch.as_tensor(intensity, dtype=torch.float32)
        if mz.ndim != 1 or intensity.ndim != 1:
            raise ValueError("each mz and intensity spectrum must be one-dimensional")
        if mz.shape != intensity.shape:
            raise ValueError("each mz and intensity spectrum must have equal lengths")
        if mz.numel() == 0:
            raise ValueError("spectra must contain at least one peak")
        if not torch.isfinite(mz).all() or not torch.isfinite(intensity).all():
            raise ValueError("mz and intensity values must be finite")
        if (intensity < 0).any():
            raise ValueError("intensity values must be nonnegative")
        base_peak = intensity.max()
        if base_peak <= 0:
            raise ValueError("spectra must contain at least one positive intensity")

        keep = intensity >= self.intensity_threshold_frac * base_peak
        mz = mz[keep]
        intensity = intensity[keep]
        if mz.numel() > self.max_peaks:
            selected = torch.topk(intensity, self.max_peaks, sorted=False).indices.sort().values
            mz = mz[selected]
            intensity = intensity[selected]
        log_intensity = torch.log1p(intensity)
        log_intensity = log_intensity / log_intensity.max().clamp_min(1e-8)
        labels = intensity / intensity.sum().clamp_min(1e-12)
        return mz.contiguous(), log_intensity.contiguous(), labels.contiguous()

    def __call__(
        self,
        mz,
        intensity,
        *,
        padding: bool | str = True,
        truncation: bool = True,
        max_length: int | None = None,
        pad_to_multiple_of: int | None = None,
        return_tensors: str | None = None,
        return_labels: bool = False,
    ) -> BatchFeature:
        """Process raw m/z and intensity arrays into model-ready features."""
        mz_batch, mz_was_single = _as_spectrum_batch(mz, "mz")
        intensity_batch, intensity_was_single = _as_spectrum_batch(intensity, "intensity")
        if len(mz_batch) != len(intensity_batch) or mz_was_single != intensity_was_single:
            raise ValueError("mz and intensity must describe the same number of spectra")

        processed = [self._process_one(m, i) for m, i in zip(mz_batch, intensity_batch)]
        limit = self.max_peaks if max_length is None else max_length
        if limit <= 0:
            raise ValueError("max_length must be positive")
        if truncation:
            processed = [(m[:limit], li[:limit], y[:limit]) for m, li, y in processed]
        elif any(m.numel() > limit for m, _, _ in processed) and padding == "max_length":
            raise ValueError("a spectrum exceeds max_length while truncation is disabled")

        lengths = [m.numel() for m, _, _ in processed]
        target_length: int | None
        if padding == "max_length":
            target_length = limit
        elif padding:
            target_length = max(lengths)
        else:
            target_length = None
        if target_length is not None and pad_to_multiple_of:
            target_length = (
                (target_length + pad_to_multiple_of - 1) // pad_to_multiple_of
            ) * pad_to_multiple_of

        data: dict[str, list] = {
            "mz": [],
            "log_intensity": [],
            "attention_mask": [],
        }
        if return_labels:
            data["labels"] = []
        for mass, log_int, labels in processed:
            length = mass.numel()
            padded_length = length if target_length is None else target_length
            pad = padded_length - length
            if pad < 0:
                raise ValueError("a spectrum exceeds the requested padded length")
            data["mz"].append(torch.cat([mass, mass.new_full((pad,), self.padding_value)]).tolist())
            data["log_intensity"].append(
                torch.cat([log_int, log_int.new_full((pad,), self.padding_value)]).tolist()
            )
            data["attention_mask"].append([1] * length + [0] * pad)
            if return_labels:
                data["labels"].append(torch.cat([labels, labels.new_zeros(pad)]).tolist())

        if mz_was_single and not padding and return_tensors is None:
            data = {name: values[0] for name, values in data.items()}
        return BatchFeature(data=data, tensor_type=return_tensors)


@dataclass
class MSDeltaDataCollatorForPreTraining:
    """Pad processed spectra and sample masked peaks for pretraining."""

    mask_ratio: float = 0.15
    min_masked: int = 1
    pad_to_multiple_of: int | None = None

    def __post_init__(self) -> None:
        if not 0.0 <= self.mask_ratio <= 1.0:
            raise ValueError("mask_ratio must be in [0, 1]")
        if self.min_masked < 0:
            raise ValueError("min_masked must be nonnegative")

    def __call__(self, features: list[dict]) -> dict[str, torch.Tensor]:
        if not features:
            raise ValueError("features must not be empty")
        lengths = [len(feature["mz"]) for feature in features]
        max_length = max(lengths)
        if self.pad_to_multiple_of:
            max_length = (
                (max_length + self.pad_to_multiple_of - 1) // self.pad_to_multiple_of
            ) * self.pad_to_multiple_of
        batch_size = len(features)
        mz = torch.zeros(batch_size, max_length, dtype=torch.float32)
        log_intensity = torch.zeros_like(mz)
        labels = torch.zeros_like(mz)
        attention_mask = torch.zeros(batch_size, max_length, dtype=torch.long)
        mask_positions = torch.zeros(batch_size, max_length, dtype=torch.bool)
        for row, (feature, length) in enumerate(zip(features, lengths)):
            if length == 0:
                continue
            mz[row, :length] = torch.as_tensor(feature["mz"], dtype=torch.float32)
            log_intensity[row, :length] = torch.as_tensor(
                feature["log_intensity"], dtype=torch.float32
            )
            target = feature.get("labels", feature.get("intensity_prob"))
            if target is None:
                raise ValueError("pretraining features must include labels")
            labels[row, :length] = torch.as_tensor(target, dtype=torch.float32)
            attention_mask[row, :length] = 1
            n_masked = min(
                length,
                max(self.min_masked, int(round(length * self.mask_ratio))),
            )
            if n_masked:
                mask_positions[row, torch.randperm(length)[:n_masked]] = True
        return {
            "mz": mz,
            "log_intensity": log_intensity,
            "attention_mask": attention_mask,
            "mask_positions": mask_positions,
            "labels": labels,
        }


MSDeltaProcessor.register_for_auto_class("AutoProcessor")
