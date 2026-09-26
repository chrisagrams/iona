# Prosit Transformer evaluation

Evaluates the released Prosit Transformer (Ekvall et al., J. Proteome Res. 2022)
on the Prosit 2020 HCD holdout, the benchmark `msdelta.intensity` reports on, so
MSDelta can be compared against it under one protocol.

The model sees exactly the authors' inputs: their released TAPE LMDB `test`
split (their conversion of `prediction_hcd_ho.hdf5`) read through their TAPE
fork's `PrositFragmentationDataset`, fed to their released torch weights. Their
TensorFlow-based converters are skipped; they only move arrays between LMDB and
HDF5. Predictions are scored twice:

- **authors' protocol**: their `cleanTapeOutput` (clip negatives, base-peak
  normalize, mask impossible ions, masked spectral distance, NaN to 0);
- **MSDelta protocol**: the spectral angle in `msdelta.intensity`.

Applied to the Prosit RNN predictions shipped in the holdout file, the
authors' protocol reproduces the file's `spectral_angle` column to 3e-14 and the
two protocols agree to 2e-8.

## Inputs

| What | Source |
|---|---|
| Torch model (`torch_model.zip`) | figshare 16691269 |
| TAPE LMDB data (`prosit_fragmentation.zip`) | figshare 16688905 |
| Holdout HDF5 (`prediction_hcd_ho.hdf5`) | figshare 12937092 |

On the lab machines these live under `/mnt/vault-1/k8/prosit-dataset/`. Only
`torch_model/{config.json,pytorch_model.bin}` and the `test` LMDB are needed;
the LMDB rows match the holdout HDF5 row for row, and the weights (163.5M
parameters) load with no missing or unexpected keys.

The released `torch_model/args.json` records the authors' training recipe:
up to 200 epochs with early stopping (patience 10), learning rate 1e-4 with
10,000 warmup steps, batch size 1028, fp16.

## Usage

```bash
cd experiments/intensity-prediction
uv sync --python 3.8

uv run prosit-transformer-eval \
  --model-dir /mnt/vault-1/k8/prosit-dataset/prosit-transformer/torch_model \
  --lmdb-dir /mnt/vault-1/k8/prosit-dataset/prosit-transformer/lmdb \
  --holdout-hdf5 /mnt/vault-1/k8/prosit-dataset/prediction_hcd_ho.hdf5 \
  --output-dir outputs/prosit-transformer
```

`--lmdb-dir` is the directory holding
`prosit_fragmentation/prosit_fragmentation_test.lmdb`. Results go to
`metrics.json` in the output directory, next to the cleaned predictions and the
per-spectrum spectral angles.

The environment is Python 3.8 with torch 1.8.1 + CUDA 11.1 (the oldest build
that supports Ampere GPUs) and the authors' TAPE fork pinned to the commit the
model was released with. NVIDIA Apex is not needed; TAPE falls back to a plain
PyTorch LayerNorm.
