"""Evaluate the released Prosit Transformer on the Prosit 2020 HCD holdout.

The model sees exactly the authors' inputs: their released TAPE LMDB ``test``
split (their conversion of ``prediction_hcd_ho.hdf5``) read through their TAPE
fork's ``PrositFragmentationDataset`` and fed to their released torch weights.
This skips their TensorFlow-based file converters, which only move arrays
between LMDB and HDF5.

Scoring follows their ``cleanTapeOutput``: clip negative predictions to 0,
normalize to the base peak, set impossible ions to -1, then Prosit's masked
spectral distance. The same predictions are also scored with the spectral angle
used by ``msdelta.intensity`` so the two protocols can be compared directly.
Predictions are aligned to the holdout HDF5 row by row, and the Prosit RNN
predictions shipped in that file give the reference number.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import torch
from tape import ProteinBertForValuePredictionFragmentationProsit
from tape.datasets import PrositFragmentationDataset
from torch.utils.data import DataLoader
from tqdm import tqdm

MAX_SEQUENCE = 30
FRAGMENT_CHARGES = 3
CHUNK = 100_000


def predict(model, dataset, batch_size: int, num_workers: int, device: torch.device):
    """Return raw model outputs and the LMDB targets, both in dataset order."""
    loader = DataLoader(
        dataset, batch_size=batch_size, num_workers=num_workers, collate_fn=dataset.collate_fn
    )
    predictions, targets = [], []
    with torch.no_grad():
        for batch in tqdm(loader, desc="predict"):
            targets.append(batch.pop("targets").numpy())
            batch = {name: tensor.to(device, non_blocking=True) for name, tensor in batch.items()}
            predictions.append(model(**batch)[0].float().cpu().numpy())
    return np.concatenate(predictions), np.concatenate(targets)


def clean_predictions(prediction: np.ndarray, charge: np.ndarray, length: np.ndarray):
    """The authors' cleanTapeOutput steps, vectorized."""
    intensities = np.clip(prediction.astype(np.float64), 0.0, None)
    with np.errstate(divide="ignore", invalid="ignore"):
        intensities = intensities / intensities.max(axis=1, keepdims=True)
    # (spectrum, ion number - 1, series y/b, neutral loss, fragment charge - 1)
    intensities = intensities.reshape(len(intensities), MAX_SEQUENCE - 1, 2, 1, FRAGMENT_CHARGES)
    # mask_outofrange: array[i, length - 1:, ...] = -1
    beyond_peptide = np.arange(MAX_SEQUENCE - 1)[None, :] >= (length[:, None] - 1)
    # mask_outofcharge: array[i, ..., charge:] = -1 (fragment charge above precursor)
    above_charge = np.arange(1, FRAGMENT_CHARGES + 1)[None, :] > charge[:, None]
    impossible = beyond_peptide[:, :, None, None, None] | above_charge[:, None, None, None, :]
    intensities = np.where(impossible, -1.0, intensities)
    return intensities.reshape(len(intensities), -1)


def authors_spectral_angle(true: np.ndarray, pred: np.ndarray, epsilon: float = 1e-7):
    """cleanTapeOutput.masked_spectral_distance, returned as 1 - distance, NaN -> 0."""
    pred_masked = ((true + 1) * pred) / (true + 1 + epsilon)
    true_masked = ((true + 1) * true) / (true + 1 + epsilon)

    def normalize(array):  # sklearn.preprocessing.normalize: zero rows stay zero
        norm = np.linalg.norm(array, axis=1, keepdims=True)
        return array / np.where(norm == 0, 1.0, norm)

    with np.errstate(invalid="ignore"):
        product = np.sum(normalize(pred_masked) * normalize(true_masked), axis=1)
        angle = 1 - 2 * np.arccos(product) / np.pi
    return np.nan_to_num(angle)


def msdelta_spectral_angle(true: np.ndarray, pred: np.ndarray):
    """msdelta.modeling_msdelta.masked_spectral_angle over slots with true >= 0."""
    mask = true >= 0
    pred = np.where(mask, np.nan_to_num(pred), 0.0)
    true = np.where(mask, np.clip(true, 0.0, None), 0.0)
    pred = pred / np.maximum(np.linalg.norm(pred, axis=1, keepdims=True), 1e-12)
    true = true / np.maximum(np.linalg.norm(true, axis=1, keepdims=True), 1e-12)
    cosine = np.clip(np.sum(pred * true, axis=1), -1 + 1e-7, 1 - 1e-7)
    return 1 - 2 * np.arccos(cosine) / np.pi


def summarize(angles: np.ndarray, charge: np.ndarray, prefix: str) -> dict:
    metrics = {
        f"{prefix}spectral_angle_median": float(np.median(angles)),
        f"{prefix}spectral_angle_mean": float(np.mean(angles)),
    }
    for z in np.unique(charge):
        metrics[f"{prefix}spectral_angle_median_charge_{int(z)}"] = float(
            np.median(angles[charge == z])
        )
    return metrics


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True, help="released torch model")
    parser.add_argument(
        "--lmdb-dir",
        type=Path,
        required=True,
        help="directory containing prosit_fragmentation/prosit_fragmentation_test.lmdb",
    )
    parser.add_argument("--holdout-hdf5", type=Path, required=True, help="prediction_hcd_ho.hdf5")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    model = ProteinBertForValuePredictionFragmentationProsit.from_pretrained(str(args.model_dir))
    model = model.to(device).eval()
    dataset = PrositFragmentationDataset(args.lmdb_dir, "test")
    raw_predictions, lmdb_targets = predict(
        model, dataset, args.batch_size, args.num_workers, device
    )

    with h5py.File(args.holdout_hdf5, "r") as handle:
        true = handle["intensities_raw"][...]
        charge = handle["precursor_charge_onehot"][...].argmax(axis=1) + 1
        length = np.count_nonzero(handle["sequence_integer"][...], axis=1)
        prosit_rnn = handle["spectral_angle"][...]
    if len(raw_predictions) != len(true):
        raise RuntimeError(f"LMDB has {len(raw_predictions)} rows, holdout HDF5 has {len(true)}")
    misaligned = ~np.isclose(lmdb_targets, true, atol=1e-6).all(axis=1)
    if misaligned.any():
        raise RuntimeError(f"{int(misaligned.sum())} LMDB rows do not match the holdout HDF5")

    cleaned = np.empty_like(raw_predictions)
    authors, ours = np.empty(len(true)), np.empty(len(true))
    for start in range(0, len(true), CHUNK):
        rows = slice(start, start + CHUNK)
        cleaned[rows] = clean_predictions(raw_predictions[rows], charge[rows], length[rows])
        authors[rows] = authors_spectral_angle(true[rows], cleaned[rows])
        ours[rows] = msdelta_spectral_angle(true[rows], cleaned[rows])

    metrics = {
        "n": int(len(true)),
        **summarize(authors, charge, "prosit_transformer/authors_protocol/"),
        **summarize(ours, charge, "prosit_transformer/msdelta_protocol/"),
        "prosit_transformer/protocol_max_abs_difference": float(np.abs(authors - ours).max()),
        "prosit_rnn/spectral_angle_median": float(np.median(prosit_rnn)),
        "model_dir": str(args.model_dir),
        "lmdb_dir": str(args.lmdb_dir),
        "holdout_hdf5": str(args.holdout_hdf5),
    }
    print(json.dumps(metrics, indent=2))
    (args.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    np.save(args.output_dir / "predictions_cleaned.npy", cleaned.astype(np.float32))
    np.save(args.output_dir / "spectral_angle_msdelta_protocol.npy", ours.astype(np.float32))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
