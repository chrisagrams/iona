# P2 card (DRAFT, needs approval — K148-P): first full Pairformer pretraining test run

**Goal (user, 2026-09-30):** a full pretraining run of the Pairformer "to see what it looks like", logged to
W&B `CS_Pharm/pairformer_pretrain`. Baseline = the existing transformer production logs (K141): no new
transformer runs. Comparisons are normalised for FLOPs/time afterwards (K144).

**Gate:** the intensity-leak audit (K147) must pass first. The earlier Pairformer runs (the user's) reached loss 0.014 / 0.0012
vs 0.061 for the transformer: consistent with the leak the source branch had (loss scales differ, see K147 correction).

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

## Update 2026-09-30 (user: (a)+(c), 0.5 epoch, fastest node count, fair comparison with the pretrained transformer)
- Data: shards 0-199 (0.5 epoch, dataset is shuffled), cap 150 drop -> ~13.5M train spectra; validation = the
  Stage 0 shard. Preprocessing job 8879887.
- Peak counts (1 shard, 250k spectra): median 207, p90 461, p99 1012; <=150: 27%, <=256: 66%, <=512: 92%.
- Fastest: 16 nodes per arm (capacity max). 16 x 12 tiles x micro 3 = global 576 = the transformer baseline's
  global batch (2 x 4 accum x 72 ranks). Est.: no tri-attn ~32 node-h (~2 h), tri-attn ~64 node-h (~4 h).
- FAIR COMPARISON (proposal, K150-P): (1) the transformer's LR schedule -- warmup 2000, cosine over its full
  540,423 steps, our run stops at 0.5 epoch (~23k steps) -- so both are compared at the same point of the same
  schedule (a schedule compressed into 23k steps would decay to 0 and flatter us); (2) the transformer's
  checkpoints (50m: 10k, 50k, 120k steps) evaluated on OUR validation set with the same loss; (3) curves vs
  spectra seen, FLOPs and node-hours.
- 512 peaks (the base's cap): ~all spectra kept (3.7x the spectra of cap 150), per-spectrum cost ~10-17x
  (N^3 in the triangle updates; K114: B=8, N=512 only fits with gradient checkpointing, 1.6 spectra/s/tile).
  0.5 epoch ~= 51M spectra -> ~650-750 node-h without tri-attn (~40-45 h on 16 nodes); with tri-attn it
  OOMs at N=512 today (needs chunked checkpointing / kernels, K102/K117). The transformer: ~45 node-h per 0.5
  epoch -> the Pairformer at 512 is ~15x more expensive per spectrum.
