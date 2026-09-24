"""Convert the Prosit 2020 HCD HDF5 files into uncompressed NumPy arrays.

The figshare files (article 12937092) store every column gzip-compressed in
column-shaped chunks, so random row access costs seconds per batch. This
decompresses each split once, sequentially, into ``.npy`` files that
``msdelta.intensity.PrositIntensityDataset`` memory-maps:

    sequence_integer.npy    int8    (N, 30)   Prosit alphabet, 0-padded
    precursor_charge.npy    int8    (N,)      1-6
    collision_energy.npy    float32 (N,)      collision_energy_aligned_normed
    intensities.npy         float32 (N, 174)  intensities_raw, -1 = impossible ion
    prosit_spectral_angle.npy float32 (N,)    Prosit RNN's own score (reference only)

``masses_raw`` is deliberately dropped: it holds OBSERVED m/z and is 0 wherever
an ion did not fire, so using it as input would leak the target.

    uv run --with h5py python scripts/convert_prosit_hdf5.py \
        --hdf5-dir /mnt/vault-1/k8/prosit-dataset \
        --out-dir /mnt/vault-1/k8/prosit-dataset/arrays
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import h5py
import numpy as np

SPLITS = {
    "train": "prediction_hcd_train.hdf5",
    "val": "prediction_hcd_val.hdf5",
    "holdout": "prediction_hcd_ho.hdf5",
}
COLUMNS = {
    "sequence_integer": ("sequence_integer", np.int8),
    "precursor_charge": ("precursor_charge_onehot", np.int8),
    "collision_energy": ("collision_energy_aligned_normed", np.float32),
    "intensities": ("intensities_raw", np.float32),
    "prosit_spectral_angle": ("spectral_angle", np.float32),
}


def convert_split(source: Path, destination: Path, block_rows: int) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    with h5py.File(source, "r") as handle:
        n_rows = handle["intensities_raw"].shape[0]
        outputs = {}
        for name, (key, dtype) in COLUMNS.items():
            shape = (n_rows,) if name == "precursor_charge" else handle[key].shape
            partial = destination / f"{name}.npy.part"
            outputs[name] = np.lib.format.open_memmap(partial, "w+", dtype=dtype, shape=shape)
        started = time.time()
        for start in range(0, n_rows, block_rows):
            stop = min(start + block_rows, n_rows)
            for name, (key, dtype) in COLUMNS.items():
                block = handle[key][start:stop]
                if name == "precursor_charge":
                    block = block.argmax(axis=1) + 1
                outputs[name][start:stop] = block.reshape(outputs[name][start:stop].shape)
            print(f"{source.name}: {stop}/{n_rows} rows ({time.time() - started:.0f}s)", flush=True)
        for name, array in outputs.items():
            array.flush()
            (destination / f"{name}.npy.part").rename(destination / f"{name}.npy")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hdf5-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--splits", nargs="+", choices=tuple(SPLITS), default=list(SPLITS))
    parser.add_argument("--block-rows", type=int, default=388_200)
    args = parser.parse_args()
    for split in args.splits:
        convert_split(args.hdf5_dir / SPLITS[split], args.out_dir / split, args.block_rows)


if __name__ == "__main__":
    main()
