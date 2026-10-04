# K195b-P: full 25m pretraining on the proposal loss, then downstream vs Chris's 25m

Status: **APPROVED 2026-10-04** as recommended ("Proposal looks good, don't do downstream until this pretrain is
complete. You can delete the intermediate finetunes, then submit the job as described"). S5 waits for S3 to finish.
Progress: S0 done (K196b, 9.11T -> 7.92T); S1 done (8903969); S2 done (8903970/1 + probe fix validated 8904056; layout =
micro 2 x 16); S3 submitted 2026-10-04 as 8904152 (capacity, 110 h). User's request (2026-10-04): "let's schedule a
full training of a 25m model using the same recipe as before but on the full ds. We then finetune it downstream and
compare. Doesn't training also have built in eval probes by default? We could use them as a first eval too. This is the
priority now".

## Question

Does pretraining on the proposal loss (KL to a masked-peak target proportional to intensity^0.5, instead of today's
linear intensity share) give a better 25m spectrum encoder downstream than Chris's 25m (today's loss, same recipe)?

## Why (K195a debug hours, notes/OBSERVATIONS.md "K195a-P")

- Arm A (today's loss) reproduced Chris's 25m curve; arm B (proposal loss) lowered both losses, A only its own.
- The two losses disagree by construction (a perfect today's-loss model scores 0.2415 on the proposal, no better
  than uniform), so the losses cannot pick the winner -- only downstream can.

## Plan

| step | what | where | needs |
|---|---|---|---|
| S0 | free storage (see "Storage") | login | your choice K195b-1 |
| S1 | full data: all 400 MSConsensus-100M train shards, Chris's processor, cap 512 (+ the same validation10k as K195a) | 1 node, capacity, ~1-2 h | S0 |
| S2 | two debug benchmarks, 2 nodes each, k195 20-shard data, ~2,000 steps, probes on every 500 steps: (a) Chris's exact layout, micro 2 x accum 16; (b) micro 32 x accum 1 (same maths) -> step time, memory, probes work and how long they take | debug + debug-scaling, 1 h each | approval of this proposal |
| S3 | the full run (settings below) | capacity, 2 nodes, one job (resume from the last checkpoint if it stops) | S1, S2 |
| S4 | first eval: the built-in probes during the run (below), plus the same probes run once on Chris's 25m final for the reference point | sidecar tiles / 1 debug job | S3 |
| S5 | downstream: K163 consensus contrastive recipe on the final, 3 seeds, scored on all six sets (validation, test, oodval, mouse, human, yeast) with / without precursor filters + library search; compared with Chris's 25m twins already scored (cons-*/s025m_ck540k_*) in the same tables and figures | capacity, 1 node, ~3-5 h + scoring | S3 |

## Settings

Fixed = same as K195a arm B / Chris's 25m production-01 (trainer_state.json):

| setting | value |
|---|---|
| architecture | Chris's 25m config (8 x 512, 8 heads, ffn 2048), fresh init, seed 0 |
| objective | proposal_loss (`--proposal_intensity_power 0.5 --train_on_proposal_loss true`); today's loss still logged as `loss` |
| optimiser | AdamW lr 1.3e-4, betas 0.9 / 0.95, wd 0.01, grad norm 1.0, cosine, warmup 2,000, bf16 |
| masking | mask ratio 0.50 |
| data | MSConsensus-100M train (all 400 shards), cap 512 peaks, random batches (no length grouping, per your K195a instruction) |
| compile | torch.compile, pad to multiple of 64, static shapes (.venv-2026, frameworks/2026.1.0) |
| eval set | validation10k (fixed 10k subset of validation shard 0; same as K195a) |
| W&B | CS_Pharm / pairformer_pretrain, run k195b-25m-B |

Choices for you (my recommendation first):

| ID | choice | options |
|---|---|---|
| K195b-1 | storage for S1 | **(a) K196b first: delete the 81 K188 intermediate fine-tune runs (1.19 TiB; all six sets are scored and plotted, nothing reads them) under the deletion protocol**; (b) preprocess in 4 chunks and delete each chunk's arrow cache after it (peak ~0.55 TB); (c) both |
| K195b-2 | global batch / steps | **512 and 540,423 steps = Chris exactly (3 epochs)**; or 528 (K195a's layout) and 540,423 steps (3.09 epochs) |
| K195b-3 | layout | **decide from S2: micro 32 x accum 1 if its first 2,000 steps match micro 2 and it fits; else Chris's micro 2 x accum 16**; both on 2 nodes x 8 tiles, the other 4 tiles of node 0 run the probes (Chris's launcher design) |
| K195b-4 | probes (first eval) | **Chris's base-config settings: linear probes on frozen features every 5,000 steps (inline, 3,000 spectra), retrieval head every 10,000 (sidecar, frozen encoder, 1 epoch ms-contrastive-100k), denoise head every 10,000 (sidecar, 1 epoch ms-denoise-100k), bias curves every 5,000**; or fewer (e.g. every 50,000) if S2 shows they are slow |
| K195b-5 | eval_loss schedule | **Chris's log-spaced points (500, 1k, 2k, 5k, ... 500k, final) plus every 10,000** (~65 evals, ~20 min total); or every 500 as in K195a (~1,080 evals, ~5 h) |
| K195b-6 | checkpoints | **every 10,000, keep all** (~55 x ~0.3 GB; lets us fine-tune intermediate checkpoints later); or keep the last 2 |
| K195b-7 | control arm | **none: Chris's 25m is the control** (arm A matched his curve to ~0.007 at 2-6k steps); or also run arm A (today's loss) on the identical setup for step-by-step probe curves (2x compute; capacity allows 2 running jobs) |
| K195b-8 | downstream | **contrastive (K163 consensus recipe, 3 seeds) only**; or also denoise fine-tuning (D recipe) |

Scripts ready (not submitted): S1 `pbs/diag/k195b_data.pbs`, S2 `pbs/diag/k195b_bench.pbs` (MICRO=2 / 32). S3 and S5
are written once K195b-2..8 are settled.

## Cost (to be firmed up by S2)

- K195a measured 2.9 steps/s on 24 tiles at micro 2 -> ~3.4 s per 10 steps. Chris's layout on 16 tiles: ~2 steps/s,
  540,423 steps ~75 h on 2 nodes (~150 node-h). Micro 32 should be several times faster (fewer, larger kernels);
  S2 measures it.
- Probes run on spare tiles of node 0 in parallel (no training slowdown unless they fall behind).
- S5: ~1 node x 3-5 h + ~1 h scoring.

## Risks

- Probes have never been run in this project (always off): S2 is the test; if they fail we fix them before S3 or run
  them post hoc on the saved checkpoints (same numbers, later).
- Compile memory at micro 32 x 512 peaks (K189 fitted micro 24 for a 50m transformer; 25m is smaller).
- A capacity job of ~1-3 days; resume from the last 10k checkpoint if it is interrupted.

## After

If B beats Chris's 25m downstream: same comparison at 50m (or 100m), and an arm A control if we want to separate the
loss from code/data-order effects. If not: record and drop (or try alpha = 0.25 / 0.75).
