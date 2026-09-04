# Configs

Each tier directory is a self-contained, HF-loadable bundle:

```
configs/msdelta-base-100m/
├── config.json                # MSDeltaConfig      → MSDeltaConfig.from_pretrained(dir)
├── preprocessor_config.json   # MSDeltaProcessor   → MSDeltaProcessor.from_pretrained(dir)
└── training.args              # every CLI flag     → msdelta-train --args_file <file>
```

`configs/deepspeed-zero2.json` is shared: ZeRO stage 2, `overlap_comm`, `contiguous_gradients`,
bf16 `torch_autocast`, with the batch fields set to `"auto"` so HF fills them from
`TrainingArguments`.

## Size tiers

| tier | hidden | layers | heads | intermediate | head dim | params |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `msdelta-base-50m` | 640 | 10 | 10 | 2560 | 64 | 49.7M |
| `msdelta-base-100m` | 800 | 13 | 10 | 3200 | 80 | 100.7M |
| `msdelta-base-200m` | 1024 | 16 | 16 | 4096 | 64 | 202.7M |
| `msdelta-base-400m` | 1280 | 20 | 20 | 5120 | 64 | 395.3M |
| `msdelta-base-1b` | 2048 | 20 | 16 | 8192 | 128 | 1011.5M |

All five share identical Fourier/bias settings — `fourier_int_n_freqs=16` over `[1e-2, 1e2]`,
`delta_bias_n_freqs=64` over `[1e-2, 1e3]`, `delta_bias_per_head_hidden=32`, both learnable — and
identical dropout (0.1). So the tiers are a **pure width/depth scaling ladder**, which is what
makes the scaling-law comparison in `TODO.txt` meaningful. The bias module is under 0.01% of
parameters at every tier.

> ⚠ `msdelta-base-400m/config.json` sets `"zero_bias_diagonal": true`. That option was removed
> from the code in commit `22c0a14`; `PretrainedConfig` stores it as an unused attribute. It has
> **no effect**. Either re-implement it or delete the key.

All processors are identical: `intensity_threshold_frac=0.01`, `max_peaks=150`.

## Training arguments

`training.args` files are whitespace-token argument files consumed via HF's `--args_file`. They
mix stock `TrainingArguments` flags with the MSDelta-specific ones below.

### MSDelta-specific

| flag | default | meaning |
| --- | --- | --- |
| `--mask_ratio` | 0.15 | Fraction of real peaks masked per spectrum. **The shipped tiers use 0.50** — an aggressive setting; it changes the loss scale (`kl_div` uses `batchmean`), so don't compare losses across different values. |
| `--validation_batches` | 50 | Eval set size = this × `per_device_eval_batch_size` |
| `--bias_curve_steps` | 5000 | Bias-panel PNG interval (0 disables) |
| `--probe_steps` | 0 | Interval for the linear / Fourier / alignment / retrieval probes (0 disables all four) |
| `--probe_num_spectra` | 3000 | Validation spectra used by the probes |
| `--replicate_retrieval_repo` | `None` | HF repo for the external replicate benchmark; unset disables that callback only |
| `--wandb_project` | `None` | Also seeds `WANDB_PROJECT` / `WANDB_DIR` |

### Denoising probe

Enabled only when `--denoise_steps > 0`. It trains a **fresh head on a frozen encoder** in a
nested `Trainer`, with RNG and gradient state forked and restored around it.

| flag | default |
| --- | --- |
| `--denoise_steps` | 0 (disabled) |
| `--denoise_dataset_repo` | `chrisagrams/ms-denoise-100k` |
| `--denoise_max_peaks` | 1024 — a *separate* processor, far above the pretraining 150 |
| `--denoise_intensity_threshold_frac` | 0.0 — no thresholding; noise peaks are the point |
| `--denoise_peak_pair_budget` | 4194304 — caps `batch × max_len²`, i.e. the padded bias tensor |
| `--denoise_head_hidden_size` / `--denoise_head_dropout` | 128 / 0.1 |
| `--denoise_epochs` / `--denoise_learning_rate` / `--denoise_weight_decay` | 1 / 1e-3 / 1e-2 |
| `--denoise_num_workers` / `--denoise_seed` | 4 / 0 |

### Data

`--dataset_repo_id` (Hub) or `--dataset_root` (local Parquet directory, last
`--num_validation_files` shards held out); `--dataset_train_split` / `--dataset_validation_split`;
`--preprocessing_num_workers`; and optional processor overrides `--intensity_threshold_frac` /
`--max_peaks` that beat `preprocessor_config.json`.

### Model

`--config_name <dir>` plus `--config_overrides "hidden_size=768,num_hidden_layers=12"`, which is
applied through `update_from_string` and re-validated. **This is the fastest way to sweep
architecture hyperparameters without writing new config directories.**

## Adding a tier

```bash
mkdir -p configs/msdelta-base-25m
cp configs/msdelta-base-50m/preprocessor_config.json configs/msdelta-base-25m/
# edit hidden_size / num_hidden_layers / num_attention_heads / intermediate_size in config.json
#   (hidden_size must be divisible by num_attention_heads)
sed 's/msdelta-base-50m/msdelta-base-25m/g' configs/msdelta-base-50m/training.args \
  > configs/msdelta-base-25m/training.args
```

Then re-tune `--learning_rate`, `--warmup_steps`, and `--max_steps`; the shipped tiers do not
follow a single automatic scaling rule.

## Quick debug run

```bash
uv run msdelta-train --args_file configs/msdelta-base-50m/training.args \
  --max_steps 50 --probe_steps 0 --denoise_steps 0 --bias_curve_steps 10 \
  --report_to none --wandb_project "" --output_dir ./runs/debug \
  --dataloader_num_workers 0 --preprocessing_num_workers 4
```
