# API index — everything implemented in `msdelta/`

Complete inventory of the package: 18 implementation modules across four subpackages, ~3,100
lines. Leading-underscore names are private helpers, listed because several of them carry real
logic. Public exports come from `msdelta/__init__.py` and from each subpackage's `__init__.py`;
see [../msdelta/README.md](../msdelta/README.md) for the layout and import conventions.

---

## `model/`

### `model/configuration.py` — config objects

| Symbol | Signature / members | Notes |
| --- | --- | --- |
| `MSDeltaConfig` | `PretrainedConfig`, `model_type="msdelta"` | 15 fields; see the table in [ARCHITECTURE.md §5](ARCHITECTURE.md#5-where-each-config-field-lands) |
| `MSDeltaConfig._validate()` | `→ None` | Checks positivity, `hidden_size % num_attention_heads == 0`, dropouts ∈ [0,1), `0 < f_min < f_max` for both Fourier blocks. **Call manually after `update_from_string`** — `train.main` does. |
| `MSDeltaDenoisingConfig` | `encoder: MSDeltaConfig`, `head_hidden_size=128`, `head_dropout=0.1` | `sub_configs = {"encoder": MSDeltaConfig}`; accepts an `MSDeltaConfig` or a plain dict |
| `MSDeltaDenoisingConfig.initializer_range` | property `→ float` | Proxies `encoder.initializer_range` so `_init_weights` works on the composed model |

Both call `register_for_auto_class()` at import.

### `model/modeling.py` — the model

> Signatures only, below. For diagrams, tensor shapes, measured parameter counts, and the
> change-impact table, see [../msdelta/model/MODULES.md](../msdelta/model/MODULES.md).

| Symbol | Signature | What it does |
| --- | --- | --- |
| `MSDeltaForPreTrainingOutput` | `ModelOutput(loss, logits)` | |
| `MSDeltaForDenoisingOutput` | `ModelOutput(loss, logits)` | |
| `PeakEmbed` | `forward(log_intensity, mask_positions=None) → (B,K,H)` | Fourier(intensity) → `Linear→GELU→Linear`; substitutes the learned `mask_token` at masked positions. **Never sees m/z.** |
| `DeltaMZBias` | `forward(mz) → (B, heads, K, K)` | `Δ = mz_i − mz_j` → Fourier → one MLP per head → concat → permute |
| `DeltaMZBias._curve` | `(feats) → (..., heads)` | Shared by `forward` and `evaluate`; casts features to the parameter dtype |
| `DeltaMZBias.evaluate` | `(delta_mz_grid) → (G, heads) float32` | 1-D curve evaluation for plotting and the alignment test — **the interpretability entry point** |
| `BiasedMHA` | `forward(hidden_states, bias, padding_mask) → (B,K,H)` | Fused QKV, SDPA with the bias as an additive float `attn_mask`, padded keys `-inf`, output projection + dropout |
| `EncoderBlock` | `forward(hidden_states, bias, padding_mask)` | Pre-norm: `x + attn(LN(x))`, then `x + ffn(LN(x))` |
| `MSDeltaPreTrainedModel` | `PreTrainedModel` base | `main_input_name="mz"`, `base_model_prefix="msdelta"`, `_no_split_modules=["EncoderBlock"]`, gradient checkpointing supported |
| `MSDeltaPreTrainedModel._init_weights` | `(module)` | Normal(0, `initializer_range`) for Linear (zero bias), standard LayerNorm init, normal init for `PeakEmbed.mask_token` |
| `MSDeltaModel` | `forward(mz, log_intensity, attention_mask=None, mask_positions=None, return_dict=None) → BaseModelOutput` | Validates shapes, builds the bias once, runs the block stack (with optional checkpointing), final LayerNorm |
| `IntensityHead` | `forward(h) → (B,K) float32` | `Linear(H→1)` |
| `PeakDenoisingHead` | `forward(h) → (B,K) float32` | `Linear→GELU→Dropout→Linear` |
| `MSDeltaForPreTraining` | `forward(..., labels=None)` | Masked relative-intensity KL — see [ARCHITECTURE.md §3](ARCHITECTURE.md#3-the-two-losses) |
| `MSDeltaForDenoising` | `__init__(config, encoder=None, freeze_encoder=False)` | Can wrap an **existing** encoder instance (no re-init) and initialize only the head |
| `MSDeltaForDenoising.freeze_encoder` | `→ None` | `requires_grad_(False)` + `.eval()` |
| `MSDeltaForDenoising.train` | `(mode=True)` | Overridden to keep a frozen encoder in eval mode regardless of the parent's mode |
| `MSDeltaForDenoising.forward` | `(..., labels=None)` | Runs a frozen encoder under `no_grad`; BCE-with-logits over `labels != -100` ∧ `attention_mask` |

`MSDeltaModel` / `MSDeltaForPreTraining` / `MSDeltaForDenoising` register with `AutoModel`,
`AutoModelForPreTraining`, `AutoModelForTokenClassification`.

### `model/fourier.py` — features and frequency health

| Symbol | Signature | What it does |
| --- | --- | --- |
| `FourierFeatures` | `(n_freqs, f_min, f_max, log_spaced=True, learnable=False, clamp_abs=2000.0)` | Frequencies as `nn.Parameter` or a persistent buffer; `out_dim = 2·n_freqs` |
| `FourierFeatures.forward` | `(x) → (..., 2·n_freqs)` | Always fp32; clamps `|x|`; `φ = 2π·x·|f|`; returns `[sin φ ‖ cos φ]`. `abs()` on the frequencies makes the sign of a learned frequency irrelevant. |
| `interp_mae` | `(freqs, x, steps=800, width=96, seed=0) → float` | **Trains a small MLP on the fly** to regress `x` from its own Fourier features (80/20 split) and returns test MAE rescaled to data units — a direct measure of whether the current frequency set can resolve the observed values |
| `dead_freqs` | `(freqs, span) → int` | Counts frequencies completing < ½ cycle across the data span, i.e. contributing nothing |
| `freq_drift` | `(freqs, init_freqs) → float` | Mean `|log10 f − log10 f₀|`, how far training moved the frequencies |

---

## `data/`

### `data/processing.py` — the processor

| Symbol | Signature | What it does |
| --- | --- | --- |
| `_as_spectrum_batch` | `(values, name) → (list[Tensor], was_single)` | Normalizes a scalar list / 1-D / 2-D / ragged nested input into a list of tensors |
| `MSDeltaProcessor` | `(intensity_threshold_frac=0.01, max_peaks=150, padding_value=0.0)` | `FeatureExtractionMixin`; `model_input_names = ["mz","log_intensity","attention_mask"]` |
| `MSDeltaProcessor._process_one` | `(mz, intensity) → (mz, log_intensity, labels, selected_idx)` | Validates finiteness/non-negativity; drops peaks below `frac × base peak`; keeps the top `max_peaks` **by intensity but restored to m/z order**; `log_intensity = log1p(i)/max`; `labels = i/Σi`; returns the surviving indices |
| `MSDeltaProcessor.process_denoising_example` | `(mz, intensity, noise) → dict` | Applies the same filtering while **carrying the noise labels through `selected_idx`** so peak↔label alignment survives |
| `MSDeltaProcessor.pad` | `(encoded_inputs, padding=True, max_length=None, return_tensors=None) → BatchFeature` | Pads m/z, log-intensity, attention mask, and labels (`-100.0` for pads). Used by `DataCollatorWithPadding` in the denoising probe. |
| `MSDeltaProcessor.__call__` | `(mz, intensity, padding=True, truncation=True, max_length=None, pad_to_multiple_of=None, return_tensors=None, return_labels=False) → BatchFeature` | The main entry point; handles single vs. batched input |
| `MSDeltaDataCollatorForPreTraining` | dataclass `(mask_ratio=0.15, min_masked=1, pad_to_multiple_of=None)` | |
| `MSDeltaDataCollatorForPreTraining.__call__` | `(features) → dict[str, Tensor]` | Pads to the longest spectrum and samples `round(len × mask_ratio)` masked positions per row via `randperm` (≥ `min_masked`). Accepts `labels` or the legacy `intensity_prob` key. |

### `data/loading.py` — datasets

| Symbol | Signature | What it does |
| --- | --- | --- |
| `charge_index` | `(peptide_charge) → int` | Parses the trailing `_z`, clipped to `N_CHARGES=8` |
| `precursor_mz` | `(peptide_charge) → float` | Sums residue masses + `[±x]` bracket modifications + water, then `(M + z·proton)/z`; returns `0.0` on any unknown residue |
| `split_paths` | `(root, n_val) → (train, val)` | Last `n_val` sorted Parquet shards become validation |
| `hf_split_paths` | `(repo_id, train_split="train", val_split="val")` | `snapshot_download` of just the matching Parquet patterns |
| `resolve_dataset_paths` | `(*, root, repo_id, train_split, validation_split, num_validation_files)` | Hub if `repo_id` is set, else local root |
| `_preprocess_example` | `(example, processor) → dict` | One row → `mz`, `log_intensity`, `labels`, `charge`, `precursor_mz`, `log_tic`, `peptide_charge`. **Swallows processor `ValueError`s into an empty spectrum**, which the caller then filters. |
| `collate_preprocessed` | `(features) → dict[str, Tensor]` | Padding-only collation, no masking — used by the diagnostics |
| `build_preprocessed_dataset` | `(paths, processor, num_proc=None)` | `load_dataset("parquet") → .map(...) → .filter(non-empty)` |
| `build_pretraining_datasets` | `(train_paths, val_paths, processor, num_proc=None)` | Both splits |
| `build_denoising_datasets` | `(repo_id, processor, num_proc=None) → DatasetDict` | Maps `process_denoising_example` over the labeled signal/noise corpus |

### `data/chemistry.py` — mass references (constants, no functions)

`WATER_MASS`, `PROTON_MASS`, `RESIDUE_MASSES` (20 AA from `pyteomics.mass.std_aa_mass`),
`RESIDUES_AA20` (I/L collapsed to a single `"L/I"` entry — they are isobaric),
`ISOTOPES` (¹³C at charge 1/2/3 and the 2×¹³C variants), `NEUTRAL_LOSSES`
(NH₃, H₂O, CO, CO₂, HPO₃, H₃PO₄, hexose).

---

## `train/`

### `train/cli.py`

| Symbol | Signature | What it does |
| --- | --- | --- |
| `train/cli.py::MSDeltaTrainer.get_decay_parameter_names` | `(model) → list[str]` | Drops every parameter ending in `.freqs` — **learned Fourier frequencies escape weight decay** |
| `main` | `(argv=None) → int` | Parses the three arg dataclasses, builds config/processor/model, applies `config_overrides` and re-validates, sets up W&B, preprocesses under `main_process_first`, slices the eval set to `validation_batches × eval_batch_size`, optionally builds the denoising datasets with a *second* processor, attaches callbacks, trains, then saves `final/` and the last bias panels on rank 0 |

Entry points: `msdelta-train` → `msdelta.train.cli:main`; `python -m msdelta.train` → `msdelta/train/__main__.py`.

### `train/args.py`

* `ModelArguments` — `config_name` (required), `config_overrides` (e.g. `"hidden_size=768,num_hidden_layers=12"`).
* `DataArguments` — `processor_name_or_path` (required), `dataset_root` | `dataset_repo_id`,
  split names, `num_validation_files`, `preprocessing_num_workers=24`, and optional processor
  overrides `intensity_threshold_frac` / `max_peaks`.
* `MSDeltaTrainingArguments(TrainingArguments)` — adds `mask_ratio`, `validation_batches`,
  `bias_curve_steps`, `probe_steps`, `probe_num_spectra`, `replicate_retrieval_repo`,
  `wandb_project`, and eleven `denoise_*` fields. Full table in
  [configs/README.md](../configs/README.md).

### `train/wandb_distributed.py`

`init_wandb_run(*, project, run_name, config) → wandb.Run | None` — one client per **node**
(returns `None` for `LOCAL_RANK != 0`). Multi-node runs use `mode="shared"` with the hostname as
the label and require a preset `WANDB_RUN_ID`; only rank 0 writes the config and the finish state.

---

## `eval/`

### `train/callbacks.py`

| Symbol | What it does |
| --- | --- |
| `_InlineCallback` | Base class: fires in `on_step_end` every `every` steps on the world-zero process; exposes `.encoder` (`module.msdelta`) and `.device`; `_wlog` stamps `train/global_step`; `empty_cache_before` optionally empties the CUDA cache first |
| `LinearProbeCallback` | Runs `probe.run_all_probes` and prints a one-line summary |
| `FourierProbeCallback` | Samples validation intensities and random peak-pair Δ values **once** (`_sample_values`, seeded, ≤1000 spectra × 64 pairs, 8192-value budget), then reports `interp_mae` / `dead_freqs` / `freq_drift` / min / max plus a log₁₀-frequency W&B histogram for both featurizers (`_featurizer_metrics`; skips non-learnable ones) |
| `AlignmentCallback` | Runs `alignment.alignment_metrics` — no dataset needed, it reads the curves directly |
| `RetrievalCallback` | Validation-set replicate retrieval vs. the binned baseline |
| `ReplicateRetrievalCallback` | Same on an external HF benchmark repo |
| `BiasPanelCallback` | Saves fine/coarse bias PNGs to `<out_dir>/figs/` and logs them as `wandb.Image` |
| `DenoisingProbeCallback` | Not an `_InlineCallback`: runs on **all** ranks (nested distributed Trainer), guards against double-firing with `last_step`, logs/prints only on rank 0, and ends with a `torch.distributed.barrier()` |
| `build_callbacks(module, val_dataset, pp, training_args, out_dir, denoising_datasets=None, denoising_processor=None)` | Assembles the enabled set; raises if `denoise_steps` is on without the datasets/processor |

### `eval/probe.py` — linear probes on the frozen encoder

| Symbol | What it does |
| --- | --- |
| `parse_charge`, `precursor_mz` | Parse `peptide_charge`; the probe version returns `None` for non-standard peptides (stricter than `data.precursor_mz`) |
| `_isotope_labels` | For each peak, counts consecutive `¹³C/z` steps present below it (0–3) within 0.01 Da — the isotope-rank target |
| `extract_representations` | One batched pass collecting **spectrum** (mean‖max pooled), **peak** (single token), and **peak-pair** (concatenated token pair) representations, with targets: precursor m/z, peak count, log TIC, charge, max m/z (baseline), isotope rank, fragment m/z, neutral-loss membership. Restores the encoder's training mode. |
| `_split`, `_regression`, `_classification` | Seeded 70/30 split; `StandardScaler` + `Ridge(α=1)` or `LogisticRegression(C=1)`; every metric ships with a baseline (majority class, or `maxmz` for precursor m/z) |
| `run_all_probes(enc, dataset, device, n_spectra=4000, batch_size=128) → dict[str, float]` | Runs them all, prefixed `probe/` |

### `eval/alignment.py` — do bias peaks land on real chemistry?

| Symbol | What it does |
| --- | --- |
| `reference_set(kinds)` | Builds the reference mass table from `isotope` / `loss` / `residue` |
| `RangeSpec` | dataclass: `name, lo, hi, step, kinds, tol, prominence, min_sep` |
| `_eval_curves` | `bias_module.evaluate` over a dense Δ grid |
| `_chance_rate` | **The null model**: the fraction of the Δ axis lying within `tol` of *any* reference mass |
| `analyze_range` | `scipy.signal.find_peaks` per head → nearest reference within `tol` → `binomtest(n_hit, n_peaks, chance, "greater")`, plus enrichment and the per-hit detail |
| `alignment_metrics(enc, fine_tol=0.02, coarse_tol=0.1, prominence=0.3)` | Two ranges — **fine** `[-5,5]` Da @ 1 mDa against isotopes, **coarse** `[2,200]` Da @ 10 mDa against losses+residues — then `align/n_sig05`, `align/n_sig01_bonf` (Bonferroni over all heads × ranges), `align/best_p`, per-range max enrichment and reference coverage |

### `eval/retrieval.py` — replicate-spectrum retrieval

| Symbol | What it does |
| --- | --- |
| `load_benchmark(repo_id, split="test")` | Loads the external benchmark and builds `peptide/charge` labels |
| `_prepped_from_dataset`, `_label_index` | Row iterator; dense first-seen label indices |
| `_collect_capped(prepped, max_peptides, per_peptide, bin_width=1.0, mz_max=2000.0)` | Caps replicate groups, drops singletons, and builds the **1-Da binned intensity-vector baseline** in the same pass |
| `_collect_benchmark(ds, pp, batch_size=512)` | Preprocesses the external benchmark in bounded batches |
| `all_but_top(X, k)` | Center + remove the top `k` principal directions (the standard embedding-whitening trick) |
| `_l2norm` | Row-wise L2 normalization |
| `retrieval_metrics_tm(X, y, device, ks=(5,), pairwise=False, chunk=1024)` | Chunked leave-one-out cosine retrieval via `torchmetrics`: `P@1`, `mAP`, `R@k`, optional pairwise `AUC-PR`. Shifts similarities by +1 so scores stay positive; blanks the diagonal. |
| `_evaluate` | Whiten → normalize → score |
| `retrieval_inline_metrics(enc, dataset, device, max_peptides=100, per_peptide=20, max_scan=150_000, whiten=0)` | Validation-set retrieval **and the binned baseline**, reporting `retrieval/gap_vs_binned` |
| `replicate_retrieval_inline_metrics(enc, repo_id, device, pp, split="test", whiten=16, batch_size=128)` | External benchmark, raw and whitened |

### `eval/denoising.py` — nested denoising probe

| Symbol | What it does |
| --- | --- |
| `PeakBudgetBatchSampler(lengths, peak_pair_budget, seed)` | Length-sorted batching capped by `(batch+1) × longest²` — bounds the **padded attention-bias size**, not the row count; shuffles batch order per epoch (`set_epoch`) |
| `DenoisingProgressCallback` | Labels the nested Trainer's tqdm bars so they don't collide with the outer run |
| `DenoisingTrainer.get_train_dataloader` | Injects the budget sampler and prepares it through `accelerate` |
| `denoising_metrics(EvalPrediction)` | Accuracy, balanced accuracy, precision, recall, F1, AUROC, AUPRC over `labels != -100`, **noise as the positive class** |
| `run_denoising_probe(module, train_ds, val_ds, *, output_dir, processor, peak_pair_budget, epochs, learning_rate, weight_decay, hidden_size, dropout, num_workers, seed, bf16, fp16)` | Wraps the live encoder in a frozen `MSDeltaForDenoising`, trains a fresh head, evaluates, saves the model + metrics. Runs inside `torch.random.fork_rng` and restores Python/NumPy RNG state and every `requires_grad` flag in a `finally` — **so it cannot perturb the outer pretraining run.** |

### `eval/embedding.py`

| Symbol | What it does |
| --- | --- |
| `pool_tokens(tokens, mask) → (B, 2H)` | Concatenated masked mean and masked max pooling |
| `encode_batch(model, mzs, log_intensities, device)` | Pads a list of ragged peak lists, builds the mask, runs the encoder |
| `embed_spectra(enc, specs, device, batch_size=128) → np.ndarray (n, 2H)` | Batched embedding with empty spectra left as zero rows |

### `eval/viz.py`

| Symbol | What it does |
| --- | --- |
| `_references_in_range(lo, hi)` | Reference masses in range, color-coded: isotopes blue, neutral losses orange, residues green |
| `plot_bias_curves(bias_module, dm_lo, dm_hi, step, title) → Figure` | One subplot per head (4 per row), curve + reference vlines, labeled when ≤ 25 references fit |
| `render_bias_panels(bias_module, step) → {"bias/fine": fig, "bias/coarse": fig}` | Fine `[-5,5]` Da @ 1 mDa and coarse `[-200,200]` Da @ 10 mDa |

Uses the `Agg` backend — safe on headless compute nodes.

---

## Not implemented

* **No tests.** No `tests/` directory, no CI workflow, no fixtures.
* **No inference/CLI beyond training.** No batch-embedding script, no checkpoint→embeddings tool;
  `embed_spectra` is the building block if you want one.
* **No `zero_bias_diagonal`** despite the 400M config setting it (removed in `22c0a14`).
* **No `delta_bias_scale`** on `MSDeltaConfig`, despite `model/experimental.py` reading it — so
  that module raises `AttributeError` on model construction.
* **No sparse autoencoder** for embedding interpretability (open item in `TODO.txt`).
* **No FLOP accounting** (open item in `TODO.txt`).
