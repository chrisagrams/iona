# K149 card (DRAFT, needs approval): does an intensity leak change training?

**Why:** user (2026-09-30): "train our leak-free copy and his copy and see if that actually affects things;
if it does it might be very serious for our paper." Waits for the K147 audit (which leaks exist, and how big).

**Two separate questions** (only B concerns the paper):

| | question | arms | model |
|---|---|---|---|
| A | How much did the pair-feature leak (source 12a, already fixed) distort Pairformer training? | Pairformer with the stand-in (ours) vs the same with the leak re-opened (the source's negative-control switch) | Pairformer (Stage 0 config) |
| B | Does the max-normalisation leak (K96) change what the TRANSFORMER learns? | current normalisation (max over all peaks, as the paper's models) vs max over VISIBLE peaks only | transformer 50m (master config) |

**Protocol (proposal):** master's recipe (lr 1.3e-4, warmup 2000, mask 0.5), same data/seed per pair, run past
the transformer's plateau (~2,000+ steps; production broke through at ~650), 2 seeds.
Read-outs: (1) training loss on the model's own inputs; (2) the SAME held-out evaluation for both arms,
computed leak-free (visible-only normalisation, stand-in), so a leaky arm can't look good by cheating;
(3) for B, if (2) differs: a short denoise / zero-shot retrieval probe on both, because downstream tasks
never mask peaks -- a pretraining leak matters for the paper only if it changes the learned representation.

**Code needed (default off, no change to existing behaviour):** a processor option to normalise over
visible peaks; a config switch to re-open the Pairformer leak (test-only today).

**Cost (rough):** A ≈ 2 arms × 2 seeds × ~1-2 h on 1 node; B ≈ same at 50m. Debug/capacity, < 30 node-h total.
