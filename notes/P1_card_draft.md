# P1-P (b): Pairformer vs transformer, short debug pretraining comparison

> **DRAFT, NOT APPROVED.** Nothing in this card has been submitted. Nothing will be
> submitted until the user approves it or edits it (notes/DECISIONS.md process). Every
> setting below marked **[choose]** is still open. The recommendations are only
> suggestions and are not defaults.

## Question

Can the ported Pairformer (`MSDeltaConfig.architecture="pairformer"`, commit 6c03366 on
`p1-pairformer`) train end-to-end through the unchanged pretraining entry point
(`msdelta/pretraining/train.py`, `pbs/aurora-pretrain.pbs`) on real data? How do its
masked-intensity loss curve and step time compare with the transformer at a similar
parameter count, over a short run?

This is a smoke test and a cost measurement. It does not decide which architecture is
better: a few hundred steps can show that training runs, and give the throughput and memory
figures. They cannot show which model is better at convergence.

## Fixed (as I understand the approval of P1)

- Code: the `p1-pairformer` branch, run from a per-job code snapshot (K29).
- Objective: the masked-intensity KL from `MSDeltaForPreTraining`, with the same collator
  and processor for both arms.
- Both arms use the same data, peak cap, batch, lr schedule, steps, seed and precision. Only
  the encoder differs.
- Outputs go to the khuss scratch, never kelhus2 (K17).
- The run uses no new code paths. It needs two new config directories, created only after
  approval: `configs/p1-debug-transformer/` and `configs/p1-debug-pairformer/`.

## Open choices (need approval)

### 1. Model sizes and how parameter counts are matched **[choose]**

The reference point is the shipped transformer `msdelta-base-50m`: 640 hidden × 10 layers,
10 heads, FFN 2560, `delta_bias_n_freqs` 256. It has **49.8M** params, of which the bias
module is 0.16M. I measured the Pairformer counts below with our port (meta device, no
compute). All of them use pair 64 / tri 64 / OPM 16 and triangle attention off, which is
the source's 50m setting.

| option | Pairformer shape | params | pair branch | matched how |
|---|---|---|---|---|
| A | 512 × 10, 8 heads, FFN 2048 (source's `pairformer-sweep-50m` shape) | 46.3M | 1.19M | the source's own "50m" tier, 7% below T |
| B | 640 × 10, 10 heads, FFN **1664** (SwiGLU at 2/3 width) | 54.5M | 1.26M | same width, depth and heads as T; +9% params |
| C | 640 × 10, 10 heads, FFN **1536** | 52.0M | 1.26M | same width and depth; +4% params |
| D | 640 × 10, 10 heads, FFN 2560 (identical dims to T) | 71.7M | 1.26M | same config numbers, but +44% params. Not a fair match |

The Pairformer single block has more weights per width than `EncoderBlock`: a SwiGLU FFN
has 3 matrices instead of 2, and the attention adds a gate. So "same dims" and "same
params" cannot both hold. Suggestion: **C**, which keeps width, depth and heads equal and
comes within about 4% of the transformer's parameter count. A is the alternative if we want
continuity with the source branch's numbers.

Pair-branch settings, also open: `pair_channels` / `pair_tri_channels` (64 in the source;
smaller values cut memory linearly and triangle compute linearly), `pair_opm_channels`
(16; the write-back activation is B·N²·opm², which the source measured as the largest
activation), `pair_use_writeback` (the source default was on), and
`pair_use_triangle_attention` (suggest **off**, as in the source: it is memory-dominant).
The chemistry priors are all on by default, as in the source.

`delta_bias_n_freqs` / `f_min` / `f_max`: the transformer uses 256 / 1e-3 / 190. The
source Pairformer used 64 / 0.01 / 1000, with learnable frequencies that our port does not
have. Suggestion: use the **transformer's values for both arms**, so that the Δm/z input is
identical.

### 2. Peaks cap (`max_peaks`) **[choose]**

This is the most important setting for cost. Triangle multiplication is cubic in N. The
collator pads to the batch maximum, so in practice N is the cap. The 50m transformer
processor uses 512. The source Pairformer used **150**, and on its data 47% of spectra sat
at the 150 cap. Compared with N=150, the triangle FLOPs at N=512 are about 40× higher, and
each pair tensor about 12× larger.

Options: (i) 150 for both arms (comparable to the source; the transformer runs on a cap it
has not used before); (ii) 512 for both (the transformer's own setting; Pairformer memory
will likely force a very small micro-batch); (iii) a middle value, e.g. 256.
Suggestion: **(i) 150 for both**. Record the fraction of spectra truncated.

### 3. Data **[choose]**

- MSConsensus-100M train/validation. This is what `msdelta-base-*` pretrains on, and the
  cache already exists.
- massive_kb_v1_shuffled, which the source sweep used.

Suggestion: **MSConsensus-100M**, the same as the transformer checkpoints we compare
against.

### 4. Steps, batch, lr **[choose]**

- Steps: suggestion **300 optimizer steps** per arm with logging every 10 steps. Debug
  queue walltime is 1 h and must include start-up and data loading. The alternative is 1000
  steps if the measured rate allows.
- Batch: the global batch should be the same for both arms. The per-tile micro-batch may
  differ, with accumulation making up the difference. The 50m transformer uses micro 64 ×
  accum 4 per tile. The source Pairformer 50m used micro 32 with gradient checkpointing.
  Suggestion: **global 256 = 1 node × 8 tiles (DDP) × micro 32 × accum 1**, with gradient
  checkpointing on for both. If micro 32 runs out of memory for the Pairformer, drop to 16
  × accum 2 for both arms.
- lr: 1.3e-4, cosine, beta2 0.95, wd 0.01 (the `msdelta-base-*` recipe). Warmup must be
  shortened because 2000 steps is longer than the run. Suggestion: **warmup 30 steps and a
  constant lr after it** (no cosine decay inside 300 steps). The alternative is the full
  recipe truncated, which means warmup only.
- Precision: bf16 for both. `torch_compile` **off** for both: the Pairformer's shapes
  change per batch and the source kept compile off, and a compiled transformer against an
  uncompiled Pairformer would distort the step-time comparison.
- DeepSpeed ZeRO-2 (used by the 50m args): off for both at 50M on one node, or on for both.
  Suggestion: off.
- Probes (denoise/retrieval sidecars, bias panels): suggestion **off**, apart from the final
  bias panel, so that the step time measures training only.

### 5. Queue, nodes, expected time **[choose]**

- Queue: `debug` (1 node, 1 h cap). The alternative is `debug-scaling`.
- Jobs: 2, one per arm, which can run at the same time. Or a single job running the arms one
  after the other, which gives the same node and so a cleaner step-time comparison.
  Suggestion: **one job, both arms in sequence**, if both fit in 1 h. Otherwise two jobs.
- Expected time. The source measured about 1.9 steps/s on a 1-node smoke run at micro 32,
  but on synthetic spectra with a batch max of about 107 peaks. At N=150 the triangle work
  is about 2.8× that. A rough range for 300 Pairformer steps is **3-10 min** of training.
  The transformer should be faster. Data start-up on the cached dataset is not measured
  here. These figures are estimates, not measurements, and producing the measurement is one
  of the reasons for this run.

### 6. Memory risk

- Pairformer at N=150, micro 32, pair 64: z is about 0.18 GB per saved fp32 activation per
  layer; bf16 halves this. Triangle multiplication holds several activations of that size.
  The write-back holds B·N²·opm² = B·N²·256 (about 0.37 GB in bf16, from the source's measurement). Gradient
  checkpointing is needed. The source ran this shape at micro 32 with checkpointing.
- At N=512 each of those tensors is about 12× larger. Micro 32 will almost certainly run out
  of memory, which is why the peaks cap has to be chosen first.
- If the job runs out of memory, the fallback order needs approval now so that the run does
  not stall. Suggestion: halve the micro-batch (keeping the global batch through
  accumulation); do not lower the peaks cap mid-card.

## What gets reported

The following is reported for both arms: the loss curve (`train/loss`, and eval loss on 50
validation batches at the end); steps/s and s/step excluding the first 20 steps; peak XPU
memory per tile; parameter counts; the fraction of spectra truncated by the cap; and the
final bias panels from both `bias_module`s. Nothing is claimed about which model is better
beyond "trains stably at the same rate or does not".

## What comes after (not part of this card)

If both arms train cleanly, the next step is to propose a longer, properly sized comparison
card (steps, seeds, `pretrain/*` skill metrics as in the source sweep README). If the
Pairformer is too slow at the chosen cap, the next step is to propose `pair_update` /
write-back / cap ablations, again as a new card.
