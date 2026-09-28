> **Status (2026-09-28):** user decisions applied: Fourier m/z-difference features use MASTER's
> defaults (256 frequencies, 1e-3 to 190) for both arms, because master's values come from earlier
> studies; peaks cap 150 is fine for this test (final runs will likely use 512 like the current
> pretrained models); W&B logging on (pbs/aurora-pretrain.pbs now loads the key); the agent's
> unsourced choices are fine for this test (real runs will decide or HP-search them); data =
> Gaolaboratory/MSConsensus-100M, downloading in full to /flare. Still open: K91-P (mass-defect
> encoding) -- the user will decide after studying notes/PAIRFORMER.md.

# P1 Stage 0: one short debug pretraining run per architecture

> **PROPOSED -- needs user approval.** Nothing on this card has been created, downloaded or
> submitted. It supersedes `notes/P1_card_draft.md` for Stage 0 (K85-P, K101-P, K95-P).

## Purpose

Stage 0 answers three questions. Does the ported Pairformer train sanely through the unchanged
pretraining entry point (`python -m msdelta.train` via `pbs/aurora-pretrain.pbs`)? What are its
step time and peak memory? What are the transformer's, under the same data, cap, batch, schedule,
precision and seed?

**Not a goal:** deciding which model is better. 300 steps cannot show that.

**Code:** branch `p1-pairformer` (encoder at 6c03366), submitted from the worktree
`/home/khuss/code/msdelta-p1-pairformer`. `pbs/lib/code_snapshot.sh` freezes the code and records
the commit in `SNAPSHOT.txt`.

**Outputs:** `/lus/flare/projects/UIC-HPC/khuss/msdelta/...` only (K17; never kelhus2).

## Arms

| arm | model config | params | gradient checkpointing |
|---|---|---|---|
| T | `configs/msdelta-base-50m/config.json`, unchanged (640 × 10, 10 heads, FFN 2560, Δm/z bank 256 / 1e-3 / 190) | 49,813,771 | off (standard; that args file has no such flag) |
| P | new `configs/p1-stage0-pairformer/config.json` (below): the source's `pairformer-sweep-50m` mapped onto our fields | 46,112,066 (pair branch 1.17M) | **on**: required, ≈45 GB without it (review #8, K93); the source also had it on (`training.args:24`) |

- Parameter counts come from the formulas in `notes/PAIRFORMER.md` §3. The same formula
  reproduces the meta-device count in the draft card.
- The arms are not parameter-matched (T is 8% larger). Stage 1 covers matched parameters and
  matched compute (ablation #3).

## Pairformer model settings (every field; source = `git show sweep/pairformer-aurora:configs/pairformer-sweep-50m/config.json`, bc2037b)

| our field | value | source field : line |
|---|---|---|
| `architecture` | `"pairformer"` | (source used `model_type` `msdelta-pairformer`, `:19`, and `--model_class`, `training.args:2`) |
| `hidden_size` | 512 | `hidden_size` :15 |
| `num_hidden_layers` | 10 | :22 |
| `num_attention_heads` | 8 | :21 |
| `intermediate_size` | 2048 | :16 |
| `hidden_dropout_prob` | 0.1 | :14 |
| `attention_probs_dropout_prob` | 0.1 | :2 |
| `layer_norm_eps` | 1e-5 | not in the file. Source default `msdelta/model/configuration.py:21` |
| `initializer_range` | 0.02 | not in the file. Source default `configuration.py:22` |
| `delta_bias_n_freqs` | **256** (master default; user 2026-09-28: Pairformer adopts master's Fourier settings) | source had 64 (:6) |
| `delta_bias_f_min` | **1e-3** (master default) | source had 0.01 (:4) |
| `delta_bias_f_max` | **190.0** (master default) | source had 1000 (:3) |
| `delta_bias_per_head_hidden` | 32 (unused by the Pairformer) | :7 |
| `pair_channels` | 64 | `pair_channels` :25 |
| `pair_transition_expansion` | 2 | :27 |
| `pair_tri_channels` | 64 | `tri_channels` :38 |
| `pair_update` | `"triangle"` | :28 |
| `pair_use_triangle_attention` | false | `use_triangle_attention` :40 |
| `pair_tri_attn_heads` | 4 (inactive) | `tri_attn_heads` :37 |
| `pair_tri_attn_dim` | 16 (inactive) | `tri_attn_dim` :36 |
| `pair_tri_attn_chunk` | 32 (inactive) | `tri_attn_chunk` :35 |
| `pair_use_writeback` | true | `use_writeback` :41 |
| `pair_opm_channels` | 16 | `opm_channels` :23 |
| `pair_single_use_mz` | true | `single_use_mz` :34 (K92: fine) |
| `pair_use_intensity` | true | :30 |
| `pair_use_mass_defect` | true | :33 (K91 parked, so the source behaviour is kept) |
| `pair_mass_defect_n_freqs` | 32 | `mass_defect_n_freqs` :18 |
| `pair_use_loss_bank` | true | :32 |
| `pair_loss_bank_sigma_ppm` | 10.0 | `loss_bank_sigma_ppm` :17 (measured choice, source `sweeps/README.md:153-171`) |
| `pair_use_isotope` | true | :31 |
| `pair_dropout` | 0.0 | :26 |
| `pair_bias_scale` | null | :24 |

**Source options our port lacks.** There is no value to set for any of these.

| source option | source value : line | what our port does instead |
|---|---|---|
| `delta_bias_learnable` | true :5 | fixed frequencies |
| `fourier_log_parameterized` | true :12 | fixed frequencies |
| `fourier_int_n_freqs` / `_f_min` / `_f_max` / `_learnable` | 16 / 1.0 / 100.0 / true :8-11 | scalar intensity MLP (`PeakEmbed`) |
| `use_global_cond` / `global_cond_dim` / `n_charges` | true / 128 / 8 :39, :13, :20 | no precursor conditioning; plain affine LayerNorms |
| `pair_use_complementarity` | true :29 | no p4 feature |

Our port has no field the source lacks, apart from `architecture`.

## Shared data and training settings

Standard files:
- T = our `configs/msdelta-base-50m/training.args`.
- S = source `configs/pairformer-sweep-50m/training.args`.

| setting | Stage 0 (both arms) | T standard (line) | S (line) | note |
|---|---|---|---|---|
| data | `Gaolaboratory/MSConsensus-100M` @ 78b3e74 (the repo `chrisagrams/MSConsensus-100M` redirects here). **Subset:** `train-0000{0,1,2,3}-of-00400.parquet` (≈1.0M spectra) + `validation-00000-of-00004.parquet` (≈250k) | `chrisagrams/MSConsensus-100M` :5, full | massive_kb_v1_shuffled :6 | the corpus our 50m used. It is **not cached** in the khuss HF cache (checked), and the full download is 190 GB. The subset is ~2.6 GB. Shard counts and sizes are from the HF API |
| peaks cap | **150** (`--max_peaks 150`; P processor `max_peaks` 150) | 512 (`preprocessor_config.json:3`) | 150 (`preprocessor_config.json:4`) | see *Cap semantics* below |
| intensity threshold | none | none | 1% of base peak (`preprocessor_config.json:3`) | **our processor has no such option** |
| mask ratio | 0.50 | :9 | :10 | same in both |
| optimizer steps | **300** (`--max_steps 300`) | 3 epochs :19 | 56,250 :20 | "few hundred", within the 1 h debug cap |
| global batch | **512** = 1 node × 8 tiles × micro 32 × accum 2 | 512 (pbs default `GLOBAL_BATCH_SIZE`) | 512 (source `pbs/aurora-sweep.pbs:97`) | same in both |
| micro-batch | 32, both arms | 64 (pbs default) | 32 (`training.args:11`) | same micro-batch, so per-forward time and memory compare directly |
| peak lr, schedule | 1.3e-4, cosine to 0 at step 300 | :14, :17 | :15, :18 | same in both |
| warmup | **11 steps** | 2000 :18 | 2000 :19 | 2000 would exceed the run. 11 keeps the source's warmup fraction (2000/56,250 = 3.6%). **Derived, not sourced** |
| Adam β1/β2, wd, grad clip | 0.9 / 0.95, 0.01, 1.0 | :15-16, :20-21 | :16-17, :21-22 | same in both |
| precision | bf16 | :22 | :23 | |
| torch_compile | false | true :23, but the pbs forces false (`aurora-pretrain.pbs:96-102, 235`) | false :25 | |
| DeepSpeed | none (DDP) | ZeRO-2 :45, stripped by the pbs (`:94-98`) | none | |
| seed | 0 | :25 | :27 | |
| eval | the forced final eval at step 300 over the kept validation subset, eval micro-batch 32 | `eval_strategy no`, log-eval from step 500 | every 500, 50 batches | `LogarithmicEvalCallback` always evaluates at `max_steps` (`callbacks.py:48`). Our trainer has no `validation_batches` limit |
| logging | every 10 steps, plus the first step | 50 :26-27 | 50 :28-29 | 30 points on the curve. **Card choice** |
| checkpoints | `--save_strategy no`. `final/` is still saved (`train.py:180`) | every 10,000 | every 1,000 | |
| probes | off: `PROBE_EXECUTION=off`, `--probe_steps 0 --bias_curve_steps 0 --denoise_steps 0 --retrieval_steps 0` | on | on | the step time measures training only. The final bias panel is still drawn (`train.py:181`) |
| dataloader / preprocessing workers | 6 / 24 | :13 / :8 | :14 / :9 | |
| W&B | project `msdelta-pretrain`, runs `p1s0-transformer`, `p1s0-pairformer` (the pbs appends the job id) | :33 | `pairformer_pretrain` | project name taken from T |

**Cap semantics (important).**
- The source *truncates* to the 150 most intense peaks (after its 1% threshold).
- Our processor *drops* any spectrum with more than `max_peaks` peaks. It raises, and
  `build_preprocessed_dataset` filters the empty row.
- MSConsensus validation peak counts (HF datasets-server statistics) are mean 252, median 205,
  max 4181. So **more than half of all spectra are dropped at cap 150**, and the kept set is
  biased toward short spectra.
- Both arms see the identical kept set, so Stage 0's goals are unaffected. The absolute losses are
  not comparable with the production 50m.
- The preprocessing log prints the kept counts (`[data] cache ready: train=…, validation=…`);
  record them.
- Getting the source's semantics would need top-K truncation in the processor, a code change with
  its own card.
- A cap of 512 (our standard) would make the Pairformer's pair tensors ≈12× larger and its
  triangle einsum ≈40× more expensive. That is not feasible at micro 32.

**N in practice:** the collator pads to the batch maximum, so with a cap of 150 most batches will
have N close to 150.

## Files to create after approval (exact contents)

**`configs/p1-stage0-pairformer/config.json`**
```json
{
  "architecture": "pairformer",
  "attention_probs_dropout_prob": 0.1,
  "delta_bias_f_max": 190.0,
  "delta_bias_f_min": 0.001,
  "delta_bias_n_freqs": 256,
  "delta_bias_per_head_hidden": 32,
  "hidden_dropout_prob": 0.1,
  "hidden_size": 512,
  "initializer_range": 0.02,
  "intermediate_size": 2048,
  "layer_norm_eps": 1e-05,
  "model_type": "msdelta",
  "num_attention_heads": 8,
  "num_hidden_layers": 10,
  "pair_bias_scale": null,
  "pair_channels": 64,
  "pair_dropout": 0.0,
  "pair_loss_bank_sigma_ppm": 10.0,
  "pair_mass_defect_n_freqs": 32,
  "pair_opm_channels": 16,
  "pair_single_use_mz": true,
  "pair_transition_expansion": 2,
  "pair_tri_attn_chunk": 32,
  "pair_tri_attn_dim": 16,
  "pair_tri_attn_heads": 4,
  "pair_tri_channels": 64,
  "pair_update": "triangle",
  "pair_use_intensity": true,
  "pair_use_isotope": true,
  "pair_use_loss_bank": true,
  "pair_use_mass_defect": true,
  "pair_use_triangle_attention": false,
  "pair_use_writeback": true
}
```

**`configs/p1-stage0-pairformer/preprocessor_config.json`**:
`{"feature_extractor_type": "MSDeltaProcessor", "max_peaks": 150, "processor_class": "MSDeltaProcessor"}`

**`configs/p1-stage0-pairformer/training.args`**
- HF args files cannot nest, and the pbs has no `MAX_STEPS`/extra-args hook. Each arm therefore
  needs a full file.
- `DATA` = `/lus/flare/projects/UIC-HPC/khuss/msdelta/p1-stage0/msconsensus-subset`.
```
--config_name configs/p1-stage0-pairformer
--processor_name_or_path configs/p1-stage0-pairformer
--max_peaks 150
--output_dir ./runs/p1s0-pairformer
--run_name p1s0-pairformer
--dataset_repo_id /lus/flare/projects/UIC-HPC/khuss/msdelta/p1-stage0/msconsensus-subset
--dataset_train_split train
--dataset_validation_split validation
--preprocessing_num_workers 24
--mask_ratio 0.50
--per_device_train_batch_size 32
--per_device_eval_batch_size 32
--gradient_accumulation_steps 2
--dataloader_num_workers 6
--learning_rate 1.3e-4
--adam_beta1 0.9
--adam_beta2 0.95
--lr_scheduler_type cosine
--warmup_steps 11
--max_steps 300
--weight_decay 0.01
--max_grad_norm 1.0
--bf16 true
--gradient_checkpointing true
--seed 0
--logging_steps 10
--logging_first_step true
--eval_strategy no
--save_strategy no
--remove_unused_columns false
--report_to wandb
--wandb_project msdelta-pretrain
--bias_curve_steps 0
--probe_steps 0
--denoise_steps 0
--retrieval_steps 0
```

**`configs/p1-stage0-transformer/training.args`**: the same file with these changes.
- Lines 1-2: `--config_name configs/msdelta-base-50m` and
  `--processor_name_or_path configs/msdelta-base-50m` (whose `max_peaks` 512 is overridden by
  `--max_peaks 150`).
- `--output_dir ./runs/p1s0-transformer` and `--run_name p1s0-transformer`.
- No `--gradient_checkpointing` line.

`--output_dir` is overridden by the pbs (`$CHECKPOINT_DIR/$RUN_NAME`).

## Exact commands (all after approval; from `/home/khuss/code/msdelta-p1-pairformer`)

**1. Login node: download the subset (~2.6 GB).**
- The dataset is public, so no token is needed.
- This is network-only, but it is more than "light", so it **needs explicit OK**.
- Alternative: do the download inside step 2's job. That needs a pbs change.
```
mkdir -p /lus/flare/projects/UIC-HPC/khuss/msdelta/p1-stage0/{msconsensus-subset,logs}
HF_HOME=/lus/flare/projects/UIC-HPC/khuss/msdelta/huggingface \
/home/khuss/code/msdelta/.venv/bin/hf download Gaolaboratory/MSConsensus-100M \
  train-00000-of-00400.parquet train-00001-of-00400.parquet \
  train-00002-of-00400.parquet train-00003-of-00400.parquet \
  validation-00000-of-00004.parquet \
  --repo-type dataset --revision 78b3e74b90021f233747de6417064c837136a9b9 \
  --local-dir /lus/flare/projects/UIC-HPC/khuss/msdelta/p1-stage0/msconsensus-subset
```
`load_dataset(<dir>)` infers the `train` and `validation` splits from the file names. Do not
download `README.md`: its `test-*` pattern would not match any file.

**2. Debug queue: preprocess once at cap 150** (CPU only; shared by both arms).
```
qsub -A UIC-HPC -q debug -l select=1 -l walltime=00:30:00 -l filesystems=home:flare \
  -j oe -o /lus/flare/projects/UIC-HPC/khuss/msdelta/p1-stage0/logs \
  -v ARGS_FILE=configs/p1-stage0-pairformer/training.args,DATASET_CACHE_DIR=/lus/flare/projects/UIC-HPC/khuss/msdelta/huggingface/datasets,PREPROCESSED_DATASET_DIR=/lus/flare/projects/UIC-HPC/khuss/msdelta/p1-stage0/msconsensus-subset-cap150 \
  pbs/aurora-preprocess.pbs
```

**3. Debug queue: one job per arm.**
- Run them one after the other if the debug queue allows only one running job per user. Check
  `qstat -Qf debug` first.
- Same node type for both. Pairformer first.
```
V=CHECKPOINT_DIR=/lus/flare/projects/UIC-HPC/khuss/msdelta/runs/p1-stage0,HF_HOME=/lus/flare/projects/UIC-HPC/khuss/msdelta/huggingface,PREPROCESSED_DATASET_DIR=/lus/flare/projects/UIC-HPC/khuss/msdelta/p1-stage0/msconsensus-subset-cap150,XPUS_PER_HOST=8,MICRO_BATCH_SIZE=32,GLOBAL_BATCH_SIZE=512,PROBE_EXECUTION=off
qsub -A UIC-HPC -q debug -l select=1 -l place=scatter -l walltime=01:00:00 -l filesystems=home:flare \
  -j oe -o /lus/flare/projects/UIC-HPC/khuss/msdelta/p1-stage0/logs \
  -v ARGS_FILE=configs/p1-stage0-pairformer/training.args,$V pbs/aurora-pretrain.pbs
qsub -A UIC-HPC -q debug -l select=1 -l place=scatter -l walltime=01:00:00 -l filesystems=home:flare \
  -j oe -o /lus/flare/projects/UIC-HPC/khuss/msdelta/p1-stage0/logs \
  -v ARGS_FILE=configs/p1-stage0-transformer/training.args,$V pbs/aurora-pretrain.pbs
```

## PBS script

**`pbs/aurora-pretrain.pbs` runs pretraining.** It launches `python -m msdelta.train` →
`msdelta/pretraining/train.py`.

A short run needs no script change for steps: `--max_steps` lives in the args file, and the micro-
and global batch come from `MICRO_BATCH_SIZE` / `GLOBAL_BATCH_SIZE` (`:81, :148-155`).

It has these gaps:
1. **W&B credentials: needs a 1-line change.** The script does not
   `source "$REPO_DIR/pbs/load_keys.sh"`, unlike `aurora-finetune.pbs:136`. With
   `--report_to wandb`, `WANDB_API_KEY` is not loaded, unless it is configured elsewhere (e.g.
   `~/.netrc`).
   - Proposed: add that line after the proxy exports.
   - Otherwise use `--report_to none`, which also loses the W&B XPU telemetry. The loss would then
     come only from `train-<job>.out`.
2. The script always passes `--preprocessed_dataset_dir` (`:229`), so in-job preprocessing is
   impossible. That is why step 2 exists.
3. It has not been used in this project's recorded runs since the package reorganisation
   (416f0b0) or on the current venv (transformers 5.17; `xccl` is accepted). It is therefore
   **first use**, and the most likely failure mode is plumbing, not the model.

## Expected time and memory

**Pairformer: estimate from the source's measurement.**
- Source: 0.569 optimizer steps/s at micro 32 × accum 1 on real spectra, N ≈ 150
  (`sweeps/README.md:115-117`). Its 27.5 h per 56,250-step arm (`:108`) agrees.
- That is ≈1.76 s per micro-step per tile. With accum 2 this gives ≈3.5 s per step, so 300 steps
  ≈ **18 min**.
- Add start-up (snapshot, telegraf, loading from disk, DDP; unmeasured, a few min) and the final
  eval (≤250k validation spectra before the drop, forward only; ≈3-7 min *est.*).
- Total ≈ 25-35 min, inside 1 h.
- Our port has no AdaLN or learned frequencies. It may run fp32 triangle operands
  (`PAIRFORMER.md` §6.9), so its speed may differ.

**Transformer: no measurement exists.**
- FLOP estimate: ≈2.2 TFLOP per micro-step, against ≈8.2 for the Pairformer (§3). Roughly 5-10 min
  of training *est.*

**Preprocessing: unmeasured.** ≈1.25M spectra with 24 workers, a few minutes *est.* (30 min
walltime).

**Peak memory per tile (64 GB): estimates.**
- Pairformer ≈10-15 GB with checkpointing. The source trained this shape "comfortably" at micro 32
  (`docs/PAIRFORMER.md:73-74`). Without checkpointing it would be ≈45 GB.
- Transformer ≈5 GB at N ≤ 150. `DeltaMZBias` features `(32,150,150,512)` fp32 = 1.5 GB.

## Success criteria (per arm)

1. All 300 steps complete. Every `loss` and `grad_norm` is finite. NaN/inf are visible because the
   pbs sets `--logging_nan_inf_filter false`.
2. The loss decreases: mean train loss over steps 250-300 < mean over steps 1-50. `eval/loss` at
   step 300 is finite. Reference only: the uniform-predictor KL floor at mask 0.5 and 150 peaks is
   ≈1.11 (source README:36-41, synthetic log-normal intensities).
3. Step time is recorded: s/step over steps 21-300, from the `train-<job>.out` timestamps / W&B.
   Excluded: XPU warm-up, the eval, and HF's `train_steps_per_second` (which includes start-up).
4. Peak memory is recorded: max `xpu memory_utilization_percent` × 64 GB (xpu-smi via telegraf,
   10 s sampling).
5. Also recorded:
   - parameter count (the `[model] …M params` line);
   - kept train/validation counts from the preprocessing log;
   - `SNAPSHOT.txt` commit.

**Stop rule / fallback (pre-approved if the card is approved):**
- If P runs out of memory at micro 32: resubmit P and T with `MICRO_BATCH_SIZE=16` (accum 4,
  global 512 unchanged).
- Never lower the cap mid-card.
- Any other failure: stop and report.

## Logged

- Per 10 steps: `train/loss`, `grad_norm`, `learning_rate`.
- Step 300: `eval/loss`, `train_runtime`.
- XPU utilization, power, frequency and memory %.
- `final/` checkpoint and `figs/*_final.png` (bias panels; for P, the Δm/z-only projection).
- Logs: `$RUN_DIR/logs/train-<job>.{out,err}` plus the PBS `-o` file.
- No `pretrain/*_skill` metric: the source's `MaskedIntensityCallback` is not in our repo.

## Settings not taken from a source

- warmup of 11 steps (derived from the source's fraction);
- 300 steps;
- logging every 10 steps;
- save/probe off;
- same micro-batch of 32 for T (T's standard is 64);
- the training-data subset (4 train + 1 validation shard) and its pinned revision;
- the W&B run names;
- the OOM fallback.

Not available in our port (value used = none): intensity threshold 1%, top-K truncation, learned
Fourier frequencies, intensity Fourier token, precursor conditioning, complementarity.
