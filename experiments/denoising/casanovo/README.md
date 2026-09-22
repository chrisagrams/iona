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

### Randomly initialized encoder (control)

`--encoder-init random` keeps the checkpoint's architecture but discards its
trained weights. It builds a fresh `Spec2Pep` from the checkpoint's saved
hyperparameters, which is the model Casanovo would start training from. The
encoder stays frozen, so this measures how much of the result comes from
pretraining and how much from the architecture plus a trained head. The
random weights depend on `--seed`.

```bash
uv run casanovo-denoising \
  --checkpoint /path/to/casanovo-orbitrap.ckpt \
  --encoder-init random \
  --output-dir outputs/random-init-seed-0 \
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
- `metrics.json`: two sets of validation metrics, both with noise as the
  positive class (see [Evaluation](#evaluation)):
  - `denoise/*`: `loss`, `accuracy`, `balanced_accuracy`, `precision`,
    `recall`, `f1`, `auroc`, `auprc`, and `noise_prevalence`, over the peaks
    Casanovo keeps.
  - `denoise_full/*`: the same metrics except `loss`, over every original
    peak, plus `num_spectra`, `num_spectra_skipped_by_casanovo`, and
    `head_peak_fraction`.
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
  noise label. A few training spectra (4 of 100,066) contain duplicate m/z
  values. Every such pair shares a noise label, so matching to either copy is
  correct. A duplicate pair with different labels would raise an error rather
  than risk a wrong label.

## Evaluation

Peaks from all validation spectra are pooled into one set and scored
together, not averaged per spectrum. A logit of 0 or above predicts noise.
AUPRC is the area under the precision–recall curve, as in MSDelta. There are
two sets of metrics:

- **`denoise/*`: kept peaks only.** Only peaks that survive Casanovo's
  preprocessing are scored (at most 150 per spectrum), and only in spectra
  Casanovo keeps. This measures the head on Casanovo's own input. It is not
  comparable with MSDelta: the top-150 cap and the other filters mostly drop
  low-intensity peaks, which are mostly noise. The scored set therefore has a
  different noise prevalence (`denoise/noise_prevalence`) and is harder than
  a full spectrum.
- **`denoise_full/*`: every original peak.** Covers every validation spectrum
  with 1 to `--full-max-peaks` peaks (default 1,024), the same spectra
  MSDelta's denoising probe scores. Peaks Casanovo keeps get the head's logit.
  Peaks Casanovo removes, and every peak of a spectrum Casanovo skips, count
  as confident noise predictions. They get a score above every head logit
  and above 0, so preprocessing is treated as part of the denoiser. These
  numbers can be compared directly with MSDelta's `denoise/*` on the same
  peaks. `head_peak_fraction` is the share of those peaks the head actually
  scored.

`denoise_full/*` reflects the whole pipeline. A peak that Casanovo's
preprocessing removes but that is really signal counts as a false positive,
even though the head never saw it.
