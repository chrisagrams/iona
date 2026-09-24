# InstaNovo-FM denoising baseline

Trains a peak-level signal/noise classifier on top of a **frozen** (or, with `--finetune full`, fine-tuned)
[InstaNovo-FM](https://github.com/instadeepai/InstaNovo-FM) spectrum encoder.
It is the counterpart of the [Casanovo baseline](../casanovo/README.md), with
the same labeled dataset (`chrisagrams/ms-denoise-100k`), head shape,
optimizer defaults, batching, and metrics as the MSDelta denoising probe
(`msdelta/denoising.py`).

## Usage

```bash
cd experiments/denoising/instanovo_fm

uv sync --python 3.12

uv run instanovo-fm-denoising \
  --output-dir outputs/instanovo-fm-v0.1.0-seed-0 \
  --device cuda \
  --precision bf16
```

`--checkpoint` takes a checkpoint file or a model ID from InstaNovo-FM's
registry (`instanovo_fm/models.json`). The default is `instanovo-fm-v0.1.0`,
the published model. Registered checkpoints are downloaded from the
InstaNovo-FM GitHub release to `~/.cache/instanovo-fm/`, the same cache
`FoundationModel.from_pretrained` uses. Compute nodes without internet
access need that file copied over first. The other foundational IDs (the
`instanovo-fm-lcfm-*` masking/pairwise-bias ablations and
`instanovo-fm-mcfm-90k-v0.1.0`) work the same way.

### Randomly initialized encoder (control)

`--encoder-init random` keeps the checkpoint's architecture but discards its
trained weights. It builds a fresh `FoundationModel` from the checkpoint's
config, which is the model pretraining starts from. The encoder stays frozen,
so this measures how much of the result comes from pretraining and how much
from the architecture plus a trained head. The random weights depend on
`--seed`.

```bash
uv run instanovo-fm-denoising \
  --encoder-init random \
  --output-dir outputs/random-init-seed-0 \
  --device cuda \
  --precision bf16
```

### Full fine-tuning

`--finetune full` trains the encoder together with the head, instead of
freezing it. The encoder gets its own learning rate, `--encoder-learning-rate`
(default `1e-4`), while the head keeps `--learning-rate`. Encoder dropout is
active during training. Backpropagating through the encoder uses much more
memory than training the head alone, so you may need a lower
`--peak-pair-budget`. The run saves the whole model's `state_dict` to
`model.pt` in place of `head.pt`. `--finetune full` can be combined with
`--encoder-init random` to train the architecture from scratch.

```bash
uv run instanovo-fm-denoising \
  --finetune full \
  --encoder-learning-rate 1e-4 \
  --output-dir outputs/full-finetune \
  --device cuda \
  --precision bf16
```

### Multiple GPUs

Multi-GPU runs work the same way as in the Casanovo baseline (see
[Multiple GPUs](../casanovo/README.md#multiple-gpus)):

```bash
uv run accelerate launch --num_processes 2 -m instanovo_fm_denoising.train \
  --output-dir outputs/instanovo-fm-v0.1.0-seed-0 \
  --device cuda \
  --precision bf16
```

Run `uv run instanovo-fm-denoising --help` to see all options. The defaults
match the MSDelta probe: 1 epoch, AdamW with lr `1e-3` and weight decay
`1e-2`, head hidden size 128, dropout 0.1, and a peak-pair budget of
4,194,304.

The run writes these files to `--output-dir`:

- `head.pt`: the head's `state_dict` only (the encoder is unchanged, so its
  weights are not saved). With `--finetune full`, this is `model.pt` instead,
  holding the encoder and head.
- `metrics.json`: `denoise/*` over the peaks preprocessing keeps and
  `denoise_full/*` over every original peak, as in the
  [Casanovo baseline](../casanovo/README.md#evaluation). The skipped-spectra
  count is named `num_spectra_skipped_by_preprocessing`.
- `run_config.json`: the CLI arguments, the checkpoint's absolute path and
  SHA-256, the preprocessing settings, the resolved device and precision,
  parameter counts, dataset sizes, and package versions.

## Why a separate environment

`instanovo-fm` requires NumPy 2 and Casanovo 5.2.0 requires NumPy < 2, so
the two baselines cannot share an environment. The metrics module and the
batching/collation code are copied from the Casanovo baseline rather than
shared.

## Method

- **Frozen encoder.** The whole `FoundationModel` is kept except its
  masked-reconstruction `prediction_heads`, which are dropped. All of its
  parameters have `requires_grad=False`. It stays in eval mode even while
  the wrapper trains, so dropout is off. For `instanovo-fm-v0.1.0` (768-dim,
  12 layers) the encoder has about 87.39M parameters. The trainable head
  (`Linear(768, 128) → GELU → Dropout(0.1) → Linear(128, 1)`) has 98,561.
- **Peak states.** `InstaNovoFMDenoiser.encode_peaks` follows
  `FoundationModel.encode` step for step, but returns the peak tokens
  instead of the latent token. The latent token InstaNovo-FM puts in front
  of the peaks is dropped, so each label lines up with one real peak's
  hidden state. Padding is masked in every attention softmax, so batches
  are padded to their longest spectrum rather than to a fixed 200. Both
  give identical peak states. The meta token is not supported: it needs
  instrument metadata the dataset lacks, and `instanovo-fm-v0.1.0` has it
  disabled.
- **Loss.** Unweighted binary cross-entropy with logits, over peaks whose
  label is not `-100` and that are not padding.
- **InstaNovo-FM preprocessing.** Spectra go through InstaNovo-FM's own
  `FoundationalDataProcessor`. It is built from the checkpoint's config the
  same way InstaNovo-FM's evaluator builds its validation processor, with
  masking disabled. For `instanovo-fm-v0.1.0`, which was trained with
  `use_spectrum_utils: false`, the steps are:
  1. Keep peaks with 50 ≤ m/z ≤ 2500.
  2. Drop peaks within `remove_precursor_tol` Da of the precursor m/z. The
     tolerance is 0, so only a peak exactly at the precursor m/z is removed.
  3. Keep peaks with raw intensity ≥ 0.01. This floor is absolute, not
     relative to the base peak. The dataset's intensities are raw counts,
     so in practice it removes nothing.
  4. Keep the 200 most intense peaks.
  5. Take the square root of the intensities, then scale them to unit L2
     norm.
  6. Divide m/z by 2500.
  7. Sort peaks by m/z (the collator's `sorted` peak ordering).

  There is no charge filter and no minimum peak count. A spectrum left with
  no peaks, which upstream would replace with a one-peak dummy spectrum, is
  dropped. None of the dataset's spectra are dropped this way. The steps only
  drop and reorder peaks and divide every m/z by the same constant. Each kept
  peak is therefore matched back to its original index by exact float32
  scaled m/z to select its noise label. Duplicate m/z values are handled the
  same way as in the Casanovo baseline.

## Evaluation

Same as the [Casanovo baseline](../casanovo/README.md#evaluation), with
InstaNovo-FM's preprocessing in place of Casanovo's. `denoise/*` scores only
the peaks the encoder sees (at most 200 per spectrum). `denoise_full/*`
scores every original peak of the validation spectra with 1 to
`--full-max-peaks` peaks. It counts peaks that preprocessing removed as
confident noise predictions, so it can be compared directly with MSDelta's
`denoise/*` and Casanovo's `denoise_full/*` on the same peaks. `denoise/*`
cannot be compared across baselines, because each baseline keeps a
different set of peaks (`denoise/noise_prevalence` shows how different).
