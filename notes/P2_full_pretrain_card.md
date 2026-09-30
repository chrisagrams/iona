# P2 card (DRAFT, needs approval — K148-P): first full Pairformer pretraining test run

**Goal (user, 2026-09-30):** a full pretraining run of the Pairformer "to see what it looks like", logged to
W&B `CS_Pharm/pairformer_pretrain`. Baseline = the existing transformer production logs (K141): no new
transformer runs. Comparisons are normalised for FLOPs/time afterwards (K144).

**Gate:** the intensity-leak audit (K147) must pass first. Chris's Pairformer runs reached loss 0.014 / 0.0012
vs 0.061 for the transformer: consistent with the leak his branch had.

**Fixed (from decisions):** current feature set (Fourier Δm/z, loss bank σ 10 ppm, isotope, relative
intensity with masked stand-in, no mass defect); same precursor decision as the transformer; master's
recipe: lr 1.3e-4, warmup 2000, cosine, 3 epochs, mask ratio 0.5, bf16, AdamW (0.9, 0.95), wd 0.01;
MSConsensus-100M rev 78b3e74 (on /flare, 102M spectra); model = the Stage 0 Pairformer (46.3M params,
hidden 512, 10 layers, pair 64, triangle multiplication, write-back).

**Choices (yours):**
| | option | data seen | est. cost (K114 throughput, ±50%) |
|---|---|---|---|
| a | peaks cap 150, spectra above it DROPPED (as Stage 0; keeps ~28% of spectra) | ~28M × 3 epochs | ~80 node-h (8 nodes ≈ 10 h) |
| b | cap 150 by keeping the TOP-150 peaks (all 102M spectra) | 102M × 3 | ~280 node-h (16 nodes ≈ 18 h) |
| c | = (a) or (b) + triangle attention | same | ~2× |
| d | cap 256 (more of the data kept, N³ cost) | — | ~5× (a) |
The transformer baseline saw all spectra at up to 512 peaks (~270 node-h, 45 h): (b) is the closer match.

**Also to set:** nodes (capacity allows 16) and walltime (up to 168 h, with checkpoints + resume);
gradient checkpointing on (7× less memory, +35% time) vs off at a smaller micro-batch; preprocessing of the
chosen data (CPU job, hours for (b)).

**Measured:** loss vs step, vs wall-clock and vs FLOPs; tokens/s; peak memory (telegraf); probes off
(or the master probes at their default steps — your call).

Balance: 9,265 node-h left (UIC-HPC).
