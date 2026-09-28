# P1 Stage 0 runbook: preprocess at cap 150, then one debug run per arm

Implements `notes/P1_stage0_card.md` (approved 2026-09-28, with the user decisions in its Status
block). **Nothing here has been submitted.** Aurora was in maintenance when this was prepared.

## What is prepared

| item | where |
|---|---|
| raw subset (4 train + 1 validation shard of `Gaolaboratory/MSConsensus-100M` @ 78b3e74, as symlinks into the HF cache; no copy) | `/lus/flare/projects/UIC-HPC/khuss/msdelta/data/stage0-cap150/raw/` |
| preprocessing args (processor `max_peaks` 150) | `configs/stage0/preprocess.args` |
| Pairformer arm: model config, processor config, training args | `configs/stage0/pairformer/{config.json,preprocessor_config.json,training.args}` |
| transformer arm: training args (model = `configs/msdelta-base-50m`, unchanged) | `configs/stage0/transformer/training.args` |
| opt-in per-block timing (K121-P, **off** in both args files) | `msdelta/utils/block_timing.py`, `tests/test_block_timing.py` |

Data layout under `/lus/flare/projects/UIC-HPC/khuss/msdelta/data/stage0-cap150/` (also in its
`README.txt`):
- `raw/`: 5 symlinks, 2.3 GB of parquet, 1,000,000 train + 250,000 validation spectra.
- `datasets/`: the datasets arrow cache written by the preprocessing job (~3 GB est.; can be
  deleted afterwards, following the deletion protocol).
- `preprocessed/`: the finalised `DatasetDict` (`train`, `validation`) both arms read.
- `logs/`: PBS output of the preprocessing job.

Settings the two arms share are identical line for line; `diff` of the two args files shows only
the config/processor path, output dir, run name, and `--gradient_checkpointing true` (Pairformer).

### Verified on the login node (CPU)

- Both args files parse (`HfArgumentParser`, as `train.py` does) and both configs build.
  - transformer: **49,813,771** params (49.81M), unchanged from the card.
  - Pairformer: **46,329,154** params (46.33M), pair stack 1,190,721. The card's 46,112,066 used
    the source's Fourier settings. The +217,088 comes from master's 256-frequency bank and dropping
    the mass-defect feature:
    - `w_c` gains (512 − 128 − 64) × 64 = +20,480;
    - `mz_proj` (single-stream Fourier(m/z)) gains (512 − 128) × 512 = +196,608.
- A real-size forward + backward (B = 2, N ≤ 24) gives a finite loss and nonzero gradients for
  both arms. The Pairformer run used gradient checkpointing.
- Tests: `tests/test_pairformer.py tests/test_configs.py tests/test_training.py tests/test_block_timing.py`
  gave 381 passed, 32 skipped, 4 failed. The 4 failures are the known `TestPBSScripts`
  level-zero ones (ms2r_crossover, rerank_handoff, rerank_psm_r4, rerank_psm_stage2).

### Preprocessing estimates (login node; extrapolated, not measured on a compute node)

- **Drop fraction at cap 150.** 1,000 spectra were sampled: the first 200 of each of the 5 shards.
  The shards are pre-shuffled (seed 42).
  - **72.0 % ± 1.4 %** (1 s.e.) have more than 150 peaks and are dropped. Per shard: 69.5 %,
    69.5 %, 72.0 %, 78.5 % (train) and 70.5 % (validation).
  - Peaks per spectrum: mean 260, median 207. The card's figure for validation from HF statistics
    was mean 252, median 205.
  - No spectrum had 0 peaks.
  - **Expected kept: ≈ 280k train, ≈ 70k validation.** 300 steps × 512 = 153,600 spectra, which
    is less than one epoch of the kept train set.
- **Runtime.**
  - `build_preprocessed_dataset` on 200 rows (1 process) took 0.68–1.34 ms/row. For 1.25M rows
    that is ≈ 15–28 CPU-min, or ≈ 1 min on 24 workers.
  - Not timed: the parquet → arrow conversion by `load_dataset` (single process, ~2.3 GB,
    estimated 1–3 min), `save_to_disk` (< 1 min) and start-up (module load, code snapshot,
    ~1–2 min).
  - **Total ≈ 5–10 min** (est.). A 1 h debug walltime leaves a wide margin.

## Commands

Run everything from the worktree that holds this branch, with a clean tree. The code snapshot
records the commit, and the pbs reads `configs/` from the checkout at job start, so do not edit
`configs/stage0/` while jobs are queued. First check the debug-queue limits (`qstat -Qf debug`:
`max_run` / `max_queued` per user).

```bash
cd /home/khuss/code/msdelta-stage0          # branch stage0-prep
git status --short                           # expect nothing
mkdir -p /lus/flare/projects/UIC-HPC/khuss/msdelta/data/stage0-cap150/logs \
         /lus/flare/projects/UIC-HPC/khuss/msdelta/runs/p1-stage0/pbs-logs
D=/lus/flare/projects/UIC-HPC/khuss/msdelta/data/stage0-cap150

# 1. Preprocess once at cap 150 (CPU; shared by both arms).
PRE=$(qsub -A UIC-HPC -q debug -l select=1 -l walltime=01:00:00 -l filesystems=home:flare \
  -j oe -o $D/logs \
  -v ARGS_FILE=configs/stage0/preprocess.args,DATASET_CACHE_DIR=$D/datasets,PREPROCESSED_DATASET_DIR=$D/preprocessed,HF_HOME=/lus/flare/projects/UIC-HPC/khuss/msdelta/huggingface,HF_HUB_OFFLINE=1,HF_DATASETS_OFFLINE=1 \
  pbs/aurora-preprocess.pbs); echo "$PRE"

# 2. The two arms, 1 node x 8 tiles, micro 32 x accum 2 = global 512, probes off.
#    Pairformer first; the transformer waits for it (afterok), so only one job runs at a time
#    and a Pairformer failure (e.g. OOM -> card fallback MICRO_BATCH_SIZE=16 for BOTH) stops T.
V=CHECKPOINT_DIR=/lus/flare/projects/UIC-HPC/khuss/msdelta/runs/p1-stage0,HF_HOME=/lus/flare/projects/UIC-HPC/khuss/msdelta/huggingface,PREPROCESSED_DATASET_DIR=$D/preprocessed,XPUS_PER_HOST=8,MICRO_BATCH_SIZE=32,GLOBAL_BATCH_SIZE=512,PROBE_EXECUTION=off
P=$(qsub -A UIC-HPC -q debug -l select=1 -l place=scatter -l walltime=01:00:00 -l filesystems=home:flare \
  -j oe -o /lus/flare/projects/UIC-HPC/khuss/msdelta/runs/p1-stage0/pbs-logs -W depend=afterok:$PRE \
  -v ARGS_FILE=configs/stage0/pairformer/training.args,$V pbs/aurora-pretrain.pbs); echo "$P"
T=$(qsub -A UIC-HPC -q debug -l select=1 -l place=scatter -l walltime=01:00:00 -l filesystems=home:flare \
  -j oe -o /lus/flare/projects/UIC-HPC/khuss/msdelta/runs/p1-stage0/pbs-logs -W depend=afterok:$P \
  -v ARGS_FILE=configs/stage0/transformer/training.args,$V pbs/aurora-pretrain.pbs); echo "$T"
```

If the debug queue does not accept held (dependent) jobs, submit each command by hand after the
previous job has finished. Leave out `-W depend=...` in that case.

Each run writes to `/lus/flare/projects/UIC-HPC/khuss/msdelta/runs/p1-stage0/<run>-<jobid>/`:
- `logs/train-<jobid>.{out,err}`;
- `final/`;
- `figs/`;
- `SNAPSHOT.txt` (under `code-snapshots/<jobid>/`).

The W&B project is `msdelta-pretrain`, with runs `p1s0-pairformer-<jobid>` and
`p1s0-transformer-<jobid>`. The pbs loads the W&B key with `pbs/load_keys.sh`. If the
`WANDB_API_KEY not loaded` warning appears in the PBS output, fix it before the arms run.

### Switching on the per-block timing (K121-P, not yet approved)

In both args files set `--block_timing_steps 5`. This times optimizer steps 281–285
(`--block_timing_start_step 281`, chosen so that steps 21–280 stay untimed for the s/step
figure). The result is `<run dir>/block_timing.json`, plus a `[block-timing] …` line in
`train-<jobid>.out`. The JSON reports ms per optimizer step for each block group, split into
`fwd` / `recompute` / `bwd`:
- Pairformer: embed, pair features, z-init, pair layer a–g, single attention h (derived) and
  transition i;
- transformer: embed, Δm/z bias, attention, FFN.

Every timed block is device-synchronised, so the timed steps are slower than the others: use the
proportions, not the absolute numbers. Commit the change before submitting.

## What to check after each job

**Preprocessing** (`$D/logs/<jobid>.*`):
1. Exit 0. The output contains `[data] cache ready: train=…, validation=…` and
   `[data] finalized dataset ready`.
2. Record the kept counts. Expected ≈ 280k / ≈ 70k (72 ± 1.4 % dropped). A count far outside
   roughly 250k–320k train means a wrong cap or the wrong files.
3. `ls $D/preprocessed` shows `dataset_dict.json`, `train/` and `validation/`.

**Each training arm** (`R=/lus/flare/projects/UIC-HPC/khuss/msdelta/runs/p1-stage0/<run>-<jobid>`):
1. Start-up:
   - `grep '\[model\]' $R/logs/train-*.out` gives `49.81M` (T) or `46.33M` (P).
   - The header line shows `micro_batch=32 accumulation=2 total_xpus=8`.
   - `[data] loading finalized dataset from …/stage0-cap150/preprocessed`.
2. All 300 steps finish. Every logged `loss` and `grad_norm` is finite (NaN/inf are not filtered):
   `grep -o "'loss': '[^']*'\|'grad_norm': '[^']*'" $R/logs/train-*.out`. There are about 31
   loss lines (step 1 and every 10 steps).
3. The loss decreases: the mean train loss over steps 250–300 is below the mean over steps
   1–50 (W&B `train/loss`). `eval_loss` at step 300 is finite. For reference only: the
   uniform-predictor floor is ≈ 1.11.
4. Step time: s/step over steps 21–300, taken from W&B `train/global_step` against wall time.
   Do not use HF's `train_steps_per_second`, which includes start-up and eval. Expected (card,
   est.): P ≈ 3.5 s/step (~18 min of training), T well under that.
5. Peak memory: the maximum of W&B system `xpu memory_utilization_percent` × 64 GB (telegraf,
   10 s sampling). Estimates: P ≈ 10–15 GB, T ≈ 5 GB.
6. Record `SNAPSHOT.txt` (commit, `dirty: no`), the walltime used, and `final/` + `figs/*_final.png`.
7. Failure handling (card):
   - P out of memory at micro 32: resubmit **both** arms with `MICRO_BATCH_SIZE=16` (accum 4,
     global 512 unchanged).
   - Never lower the cap.
   - Any other failure: stop and report.
