# Casanovo denoising baseline

Trains a peak-level signal/noise classifier on top of a **frozen** Casanovo
spectrum encoder, using the same labeled dataset
(`chrisagrams/ms-denoise-100k`), head shape, optimizer defaults, and metrics as
the MSDelta denoising probe (`msdelta/denoising.py`).

## Usage

```bash
cd experiments/denoising/casanovo

uv sync --python 3.12

uv run casanovo-denoising \
  --checkpoint /path/to/casanovo-orbitrap.ckpt \
  --output-dir outputs/orbitrap-seed-0 \
  --device cuda \
  --precision bf16
```

### Multiple GPUs

Training uses [Accelerate](https://huggingface.co/docs/accelerate). To split
one run across GPUs, launch one process per GPU:

```bash
uv run accelerate launch --num_processes 2 -m casanovo_denoising.train \
  --checkpoint /path/to/casanovo-orbitrap.ckpt \
  --output-dir outputs/orbitrap-seed-0 \
  --device cuda \
  --precision bf16
```

- **Training.** Each GPU trains on its own share of the peak-budget batches.
  The number of batches is padded with repeats so every GPU takes the same
  number of steps, and gradients are averaged across GPUs. The effective
  batch is therefore `num_processes` times larger than in a single-GPU run.
- **Evaluation.** Each validation spectrum is evaluated exactly once, and
  predictions are gathered from all GPUs before metrics are computed.
- **Output.** Only the main process writes the output files.

Run `uv run casanovo-denoising --help` to see all options. The defaults match
the MSDelta probe: 1 epoch, AdamW with lr `1e-3` and weight decay `1e-2`, head
hidden size 128, dropout 0.1, and a peak-pair budget of 4,194,304.

The run writes these files to `--output-dir`:

- `head.pt`: the head's `state_dict` only (the encoder is unchanged, so its
  weights are not saved).
- `metrics.json`: validation metrics named `denoise/loss`,
  `denoise/accuracy`, `denoise/balanced_accuracy`, `denoise/precision`,
  `denoise/recall`, `denoise/f1`, `denoise/auroc`, `denoise/auprc`, and
  `denoise/noise_prevalence`. Noise is the positive class.
- `run_config.json`: the CLI arguments, the Casanovo checkpoint's absolute
  path and SHA-256, the resolved device and precision, parameter counts,
  dataset sizes, and package versions.

## Why a separate environment

Casanovo 5.2.0 requires NumPy < 2. The root MSDelta project uses NumPy 2. This
project therefore has its own `pyproject.toml` and virtual environment. **Do not
add Casanovo to the root `pyproject.toml`.**

## Method

- **Frozen encoder.** Only `Spec2Pep.encoder` is kept from the checkpoint. All
  of its parameters have `requires_grad=False`. It stays in eval mode even
  while the wrapper trains, so dropout is off. Casanovo's encoder has about
  18.93M parameters. The trainable head
  (`Linear(512, 128) → GELU → Dropout(0.1) → Linear(128, 1)`) has 65,793.
- **Global token excluded.** Casanovo puts a global spectrum token in front of
  the peak states. That position is dropped, so each label lines up with one
  real peak's hidden state.
- **Loss.** Unweighted binary cross-entropy with logits. It is computed over
  peaks whose label is not `-100` and that are not Casanovo padding.
- **Casanovo preprocessing.** Spectra go through Casanovo's own
  `preprocessing_fn` list, taken from a default `DeNovoDataModule`, so the
  steps and parameters match `casanovo/config.yaml`:
  1. `set_mz_range(50, 2500)`
  2. `remove_precursor_peak(2.0, "Da")`, which also removes the precursor at
     every lower charge state. It uses the dataset's `precursor` and `charge`
     columns.
  3. `scale_intensity("root", 1)`
  4. `filter_intensity(0.01, 150)`. The 1% floor applies after the square
     root, so it is 0.01% of the raw base peak.
  5. Drop spectra with fewer than 20 peaks.
  6. Scale intensities to unit L2 norm.

  Spectra with a charge outside Casanovo's valid range (1–10), or that raise
  the errors Casanovo's parser skips, are dropped. These steps only drop peaks
  and sort them by m/z; they never change m/z values. Each kept peak is
  therefore matched back to its original index by exact m/z to select its
  noise label. A spectrum with duplicate m/z values raises an error rather
  than risk a wrong label. The dataset has none.

## Caveat: not a controlled common-peaks comparison

Casanovo's native preprocessing changes which peaks are evaluated compared
with MSDelta. The m/z window, precursor removal, and top-150 cap drop peaks
that MSDelta keeps, and noise and signal peaks are not dropped at the same
rate. This changes the noise prevalence and the difficulty of the task. Treat
these numbers as an **operational baseline**: each model sees its own native
input. They are not a head-to-head comparison on the same peaks. The logged
`denoise/noise_prevalence` shows how far the evaluated population has shifted.
