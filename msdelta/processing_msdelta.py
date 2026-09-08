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
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
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

        selected = torch.nonzero(
            intensity >= self.intensity_threshold_frac * base_peak, as_tuple=False
        ).squeeze(-1)
        mz = mz[selected]
        intensity = intensity[selected]
        if mz.numel() > self.max_peaks:
            top = torch.topk(intensity, self.max_peaks, sorted=False).indices.sort().values
            mz = mz[top]
            intensity = intensity[top]
            selected = selected[top]
        log_intensity = torch.log1p(intensity)
        log_intensity = log_intensity / log_intensity.max().clamp_min(1e-8)
        labels = intensity / intensity.sum().clamp_min(1e-12)
        return mz.contiguous(), log_intensity.contiguous(), labels.contiguous(), selected

    def process_denoising_example(self, mz, intensity, noise) -> dict[str, list | int]:
        """Process one spectrum while preserving peak/noise-label alignment."""
        noise = torch.as_tensor(noise, dtype=torch.bool)
        mz_tensor = torch.as_tensor(mz, dtype=torch.float32)
        intensity_tensor = torch.as_tensor(intensity, dtype=torch.float32)
        if noise.ndim != 1 or noise.shape != mz_tensor.shape:
            raise ValueError("mz, intensity, and noise must have equal one-dimensional shapes")
        mass, log_intensity, _, selected = self._process_one(mz_tensor, intensity_tensor)
        return {
            "mz": mass.tolist(),
            "log_intensity": log_intensity.tolist(),
            "labels": noise[selected].float().tolist(),
        }

    def process_retrieval_example(self, consensus, experimental) -> dict[str, list]:
        """Process one consensus spectrum and its three experimental replicates."""
        if len(experimental) != 3:
            raise ValueError("retrieval examples must contain exactly three experimental spectra")
        spectra = [consensus, *experimental]
        processed = [
            self(spectrum["mz"], spectrum["intensity"], padding=False) for spectrum in spectra
        ]
        return {
            "mz": [values["mz"] for values in processed],
            "log_intensity": [values["log_intensity"] for values in processed],
        }

    def pad(
        self,
        encoded_inputs: list[dict[str, Any]],
        *,
        padding: bool | str = True,
        max_length: int | None = None,
        return_tensors: str | None = None,
        **kwargs,
    ) -> BatchFeature:
        """Pad processed peak-classification examples to a common length."""
        lengths = [len(example["mz"]) for example in encoded_inputs]
        if padding == "max_length":
            if max_length is None:
                raise ValueError("max_length is required with padding='max_length'")
            target_length = max_length
        else:
            target_length = max(lengths)

        data: dict[str, list] = {
            "mz": [],
            "log_intensity": [],
            "attention_mask": [],
            "labels": [],
        }
        for example, length in zip(encoded_inputs, lengths):
            pad = target_length - length
            data["mz"].append(example["mz"] + [self.padding_value] * pad)
            data["log_intensity"].append(example["log_intensity"] + [self.padding_value] * pad)
            data["attention_mask"].append([1] * length + [0] * pad)
            data["labels"].append(example["labels"] + [-100.0] * pad)
        return BatchFeature(data=data, tensor_type=return_tensors)

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

        processed = [self._process_one(m, i)[:3] for m, i in zip(mz_batch, intensity_batch)]
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


@dataclass
class MSDeltaDataCollatorForRetrieval:
    """Flatten and pad four-spectrum analyte groups for contrastive training."""

    def __call__(self, features: list[dict]) -> dict[str, torch.Tensor]:
        if not features:
            raise ValueError("features must not be empty")
        mzs: list[list[float]] = []
        log_intensities: list[list[float]] = []
        group_ids: list[int] = []
        for group_id, feature in enumerate(features):
            group_mz = feature["mz"]
            group_intensity = feature["log_intensity"]
            if len(group_mz) != 4 or len(group_intensity) != 4:
                raise ValueError("each retrieval example must contain four spectra")
            mzs.extend(group_mz)
            log_intensities.extend(group_intensity)
            group_ids.extend([group_id] * 4)

        target_length = max(max((len(mz) for mz in mzs), default=0), 1)
        batch_size = len(mzs)
        mz = torch.zeros(batch_size, target_length, dtype=torch.float32)
        log_intensity = torch.zeros_like(mz)
        attention_mask = torch.zeros(batch_size, target_length, dtype=torch.long)
        for index, (mass, intensity) in enumerate(zip(mzs, log_intensities)):
            length = len(mass)
            if length == 0:
                continue
            mz[index, :length] = torch.as_tensor(mass, dtype=torch.float32)
            log_intensity[index, :length] = torch.as_tensor(intensity, dtype=torch.float32)
            attention_mask[index, :length] = 1
        return {
            "mz": mz,
            "log_intensity": log_intensity,
            "attention_mask": attention_mask,
            "group_ids": torch.tensor(group_ids, dtype=torch.long),
        }


MSDeltaProcessor.register_for_auto_class("AutoProcessor")


class MSDeltaRerankingProcessor(MSDeltaProcessor):
    """Process spectra and tokenize the dataset's localized modified residues."""

    model_input_names = [
        "mz",
        "log_intensity",
        "attention_mask",
        "peptide_input_ids",
        "peptide_attention_mask",
    ]

    def __init__(self, peptide_vocab=None, peptide_max_length=25, **kwargs):
        super().__init__(**kwargs)
        self.peptide_vocab = peptide_vocab or [
            "[PAD]",
            *list("ACDEFGHIKLMNPQRSTVWY"),
            "C[57.0215]",
            "M[15.9949]",
        ]
        self.peptide_max_length = peptide_max_length
        if peptide_max_length <= 0:
            raise ValueError("peptide_max_length must be positive")
        if self.peptide_vocab[0] != "[PAD]" or len(set(self.peptide_vocab)) != len(
            self.peptide_vocab
        ):
            raise ValueError("peptide_vocab must be unique with [PAD] at index zero")

    def tokenize_peptide(self, peptide: str) -> list[int]:
        """Consume every character; never discard unknown or misplaced modifications."""
        vocab = {token: index for index, token in enumerate(self.peptide_vocab)}
        ids = []
        position = 0
        while position < len(peptide):
            end = position + 1
            if end < len(peptide) and peptide[end] == "[":
                closing = peptide.find("]", end)
                if closing == -1:
                    raise ValueError(f"unterminated modification in {peptide!r}")
                end = closing + 1
            token = peptide[position:end]
            if token not in vocab or vocab[token] == 0:
                raise ValueError(f"unsupported peptide residue {token!r} in {peptide!r}")
            ids.append(vocab[token])
            position = end
        if not ids or len(ids) > self.peptide_max_length:
            raise ValueError(f"peptides must contain 1–{self.peptide_max_length} residues")
        return ids

    def encode_peptides(self, peptides: list[str]) -> BatchFeature:
        ids = [self.tokenize_peptide(peptide) for peptide in peptides]
        if not ids:
            raise ValueError("peptides must not be empty")
        length = max(map(len, ids))
        return BatchFeature(
            data={
                "peptide_input_ids": [row + [0] * (length - len(row)) for row in ids],
                "peptide_attention_mask": [
                    [1] * len(row) + [0] * (length - len(row)) for row in ids
                ],
            },
            tensor_type="pt",
        )


@dataclass
class MSDeltaDataCollatorForReranking:
    """Pad grouped training rows or individual spectrum/peptide evaluation rows."""

    def __call__(self, features: list[dict]) -> dict[str, torch.Tensor]:
        if not features:
            raise ValueError("features must not be empty")
        batch = {}
        grouped = "peptide_id" in features[0]
        if "mz" in features[0]:
            spectra = [
                (mz, intensity)
                for row in features
                for mz, intensity in (
                    zip(row["mz"], row["log_intensity"])
                    if grouped
                    else [(row["mz"], row["log_intensity"])]
                )
            ]
            lengths = torch.tensor([len(mz) for mz, _ in spectra])
            if not len(lengths) or (lengths == 0).any():
                raise ValueError("spectra must contain at least one peak")
            batch["mz"] = torch.nn.utils.rnn.pad_sequence(
                [torch.tensor(mz, dtype=torch.float32) for mz, _ in spectra], batch_first=True
            )
            batch["log_intensity"] = torch.nn.utils.rnn.pad_sequence(
                [torch.tensor(i, dtype=torch.float32) for _, i in spectra], batch_first=True
            )
            batch["attention_mask"] = torch.arange(int(lengths.max()))[None, :] < lengths[:, None]
        if "peptide_input_ids" in features[0]:
            batch["peptide_input_ids"] = torch.nn.utils.rnn.pad_sequence(
                [torch.tensor(row["peptide_input_ids"], dtype=torch.long) for row in features],
                batch_first=True,
            )
            batch["peptide_attention_mask"] = batch["peptide_input_ids"] != 0
        if grouped:
            batch["peptide_labels"] = torch.tensor([row["peptide_id"] for row in features])
            batch["spectrum_labels"] = torch.tensor(
                [row["peptide_id"] for row in features for _ in row["mz"]]
            )
        if "evaluation_labels" in features[0]:
            batch["evaluation_labels"] = torch.tensor([r["evaluation_labels"] for r in features])
        return batch


MSDeltaRerankingProcessor.register_for_auto_class("AutoProcessor")
