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
- **Casanovo preprocessing.** For each spectrum:
  1. Keep peaks with m/z in [50, 2500].
  2. If a `precursor_mz` is available, remove peaks within 2 Da of it.
  3. Remove peaks below 1% of the remaining base peak.
  4. Keep the 150 most intense peaks, restored to their original order.
  5. Apply square-root scaling to intensities, then normalize by the sum.
  6. Drop spectra with fewer than 20 peaks.

  The same peak selection is applied to the noise labels.

## Caveat: not a controlled common-peaks comparison

Casanovo's native preprocessing changes which peaks are evaluated compared
with MSDelta. The m/z window, intensity floor, and top-150 cap remove many
low-intensity peaks, and noise peaks are overrepresented among those. This
changes the noise prevalence and the difficulty of the task. Treat these
numbers as an **operational baseline**: each model sees its own native input.
They are not a head-to-head comparison on the same peaks. The logged
`denoise/noise_prevalence` shows how far the evaluated population has shifted.
