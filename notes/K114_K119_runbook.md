# K114 / K119 runbook: Pairformer per-block profile and FlexAttention on XPU

Branch `k114-k119-prep` (worktree `/home/khuss/code/msdelta-prof`). Both jobs are one tile
(`ZE_AFFINITY_MASK=0`), debug queue, standalone under `pbs/diag/`, run from a code snapshot
(`pbs/lib/code_snapshot.sh`), bf16 autocast over fp32 weights as in training. **Nothing has been
submitted.** Submit from the worktree root after the maintenance:

```bash
cd /home/khuss/code/msdelta-prof
# K114 (expect ~10-25 min; walltime 45 min)
qsub -q debug -l select=1 -l walltime=00:45:00 -A UIC-HPC -l filesystems=home:flare \
     -v REPO_DIR=$PWD pbs/diag/pairformer_profile.pbs
# K119 (expect ~10-30 min, mostly torch.compile; walltime 60 min)
qsub -q debug -l select=1 -l walltime=01:00:00 -A UIC-HPC -l filesystems=home:flare \
     -v REPO_DIR=$PWD pbs/diag/flexattn_test.pbs
```

Optional narrowing: `-v REPO_DIR=$PWD,PROFILE_ARGS="--models pairformer --peaks 100,150"` (K114),
`-v REPO_DIR=$PWD,FLEX_ARGS="--batches 8 --peaks 100"` (K119). To reproduce the failure with
the venv's upstream triton instead: `-v REPO_DIR=$PWD,MSDELTA_SYSTEM_TRITON=0`.

Outputs: `results/raw/diag/pairformer_profile/<jobid>.{json,log}`,
`results/raw/diag/flexattn/<jobid>.{json,log}`; PBS stdout in `pbs/logs/`.

## K114: `pbs/diag/pairformer_profile.py`

Purpose (user): calibrate how many single blocks take as long as one pair update, at each N.

Models: `pairformer` = `configs/diag/k114-pairformer-stage0/config.json`, a verbatim copy of
`git show stage0-prep:configs/stage0/pairformer/config.json` (commit cde0905; triangle
attention off; a test checks the copy); `pairformer_triattn` = the same with triangle
attention on, `pair_tri_attn_impl="sdpa"`; `transformer` = `configs/msdelta-base-50m`.
Sizes B 8/32 x N 100/150/256/512; random spectra (m/z 100-2000, lengths 0.3N-N, one at N),
padded and masked (ratio 0.5) by the pretraining collator. Full pretraining step
(`MSDeltaForPreTraining` + KL loss), fwd+bwd, no optimizer.

Per (model, B, N), three modes, each OOM-safe:
- `plain_gc_off` / `plain_gc_on`: median step time (no hooks) and peak memory with gradient
  checkpointing off / on (Stage 0 trains with it on).
- `profiled`: checkpointing off, device sync around every block; median over 3 steps of each
  block's forward and backward time, % of step, forward allocated-memory growth.
If both plain modes OOM at N, larger N (and larger B at >= N) are skipped.

Blocks. Pairformer: `token_embed`, `pair_features`, `z_init` (init_state minus features), per
layer `a_writeback`, `b_trimul_out`, `c_trimul_in`, `d_triattn_start`/`e_triattn_end` (triattn
model only), `f_pair_transition`, `g_bias_readout`, `pair_glue` (residuals/dropout/permute),
`h_single_attention` (single block minus its transition: LN, qkv, SDPA, gate, out),
`i_single_transition`, `final_ln`, `head`, `loss` (backward of the loss before the head),
`other`. Transformer: `token_embed`, `dmz_bias`, `attention`, `ffn`, `block_glue`, `final_ln`,
`head`. Times are summed over the 10 layers (the `calls` column says how many).

What it prints: per config a table (block, calls, fwd ms, bwd ms, total, % step, fwd GB),
the line `per layer: pair update X ms, single block Y ms -> R single blocks per pair update`,
and at the end a summary grid: model, B, N, pair ms/layer, single ms/layer, ratio R,
transformer block ms/layer, pair/transformer.

Reading it:
- **R = single blocks per pair update** (fwd+bwd, also fwd-only and bwd-only in the JSON
  `calibration`) is the K114 answer; read how it grows with N (pair work ~N^3 for tri-mult,
  ~N^2 elsewhere; single attention ~N^2 but small constant).
- The per-block backward is attributed from timestamps taken by identity autograd markers at
  every block's inputs and outputs (the engine runs backward in reverse creation order), so
  the blocks tile the whole backward; syncs add overhead: compare `profiled.step_ms` with
  `plain_gc_off.step_ms` (a few % expected). Proportions are the robust part.
- `plain_gc_on.step_ms / plain_gc_off.step_ms` is the checkpointing recompute cost; peak GB
  with GC on is what Stage 0 (B 32, N 150) will see.
- OOM rows: `status: "oom"` in the JSON; skipped rows carry `skipped`.

## K119: `pbs/diag/flexattn_test.py`

Question: does compiled `torch.nn.attention.flex_attention` work on Aurora XPU for triangle
attention, is it correct vs the SDPA path, and is it faster / leaner?

Intel Triton, job-only: the `.venv` has upstream triton 3.8.0 (no `backends/intel`) which
shadows the frameworks' Intel triton 3.6.0. When run as a script, before importing torch,
the test makes a temp dir with symlinks to ONLY `triton` and `triton-*.dist-info` from
`/opt/aurora/26.26.0/frameworks/aurora_frameworks-2025.3.1/lib/python3.12/site-packages`, puts it
at `sys.path[0]` and at the front of `PYTHONPATH` (inductor compile workers inherit it).
Nothing else is shadowed; the venv is untouched. Compile caches are job-local
(`$TMPDIR/msdelta-k119-<job>/`).

Steps (each recorded with status and traceback in the JSON; nothing crashes the job):
- `a_imports`: triton version / real path / backends (must list `intel`), active driver and
  target; `a_triton_kernel`: vector add on XPU.
- `b_make_flex`, `b_trivial_flex`: compiled flex_attention with no mods vs SDPA.
- `correctness`: TriangleAttention (c_z 64, 4 x 16) with the flex core vs the model's SDPA
  path, same weights and inputs (15% padded keys on every other spectrum): outputs and grads
  of z and every parameter; fp32 and bf16 autocast; layouts `rowbatch` (rows i in the batch
  dim, `bias[b // N]`) and `rowhead` (rows in the head dim, `bias[b, h % H]`); masking via
  `block_mask` or inside `score_mod`; starting and ending node; N 37 and 100.
- `bench`: bf16 fwd+bwd of one module, B 8/32 x N 100/150: `sdpa` (model path, chunk 32) vs
  `flex_rowbatch_blockmask` vs `flex_rowhead_blockmask`; median ms, peak GB, first-step
  (compile) seconds.

What it prints: `[flex] triton 3.6.0 from .../site-packages/triton/__init__.py; backends [...]`,
each step's status, a correctness table (worst relative L2 error over output and all grads:
flex/sdpa, flex/naive-fp32, sdpa/naive-fp32) and a bench table like `triattn_bench`.

Reading it:
- If `a_imports` shows triton 3.8.0 or no `intel` backend, the shim failed: nothing after it
  means anything.
- Correct: fp32 flex/sdpa worst rel-L2 ~1e-6 or below; bf16 flex/naive should be of the same
  order as sdpa/naive (~1e-2). A failure listing only `grad.bias.weight` means gradients
  through the captured bias in score_mod are not supported by this build.
- Useful if flex is faster than `sdpa` at B 32 / N 150 or needs clearly less memory; compare
  with the K102 `triattn_bench` numbers. Neither layout needs chunking (logits are never
  materialised).

## Tests

`tests/test_pairformer_profile.py` (18 CPU tests): profiler wrappers leave loss/logits bitwise
equal and gradients equal to float rounding (seen: 1.6e-5 relative on the embed's mask token,
from a different accumulation order), per-block times tile the step, batch masking, the
config copy, the flex score_mod/mask_mod indexing, the rebuilt module with the reference flex
core vs SDPA and naive (both layouts, both maskings, both directions), eager CPU flex_attention
vs SDPA, and the triton shim.
