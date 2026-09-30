# Open questions

Every question Claude raises in chat is also written here, with enough context to be read
on its own later. When the user decides, the entry moves to `DECISIONS.md` (with the decision)
and is deleted here. IDs: running number + track suffix (-C contrastive / spectrum encoder,
-A alignment / peptide encoder, -D denoise, -R rescoring, -I infrastructure, -P Pairformer,
-S cross-cutting).

Status: **parked** = user said to come back later; **open** = waiting on the user.

---

## Your to-do (user asked to be reminded)

- **K63-I** -- review the job-DAG scheduler design (notes/DAG_SPEC.md) and say whether you agree.
- Study notes/PAIRFORMER.md (mass-defect feature already dropped, K91/K113).

---

## Alignment (peptide encoder) -- parked until the new C models exist

### K86-A: which 20 ppm number the paper cites for the yHydra comparison (parked 2026-09-28)
Context: the alignment evaluation (peptide <-> spectrum retrieval) was upgraded (branch
`a-filtered-eval`, 58c9b28) to report every number without a precursor filter, with a plain 20 ppm
filter and with an isotope-tolerant 20 ppm filter, on filter passes and failures (project rule).
The OLD yHydra comparison script had its own windows, "±1.1 Da" and "20 ppm", which compare
NEUTRAL masses and ignore charge; those numbers are in the paper. The NEW standard 20 ppm filter
compares m/z at the query's charge and requires the charge to match. Both are now computed; they
can differ slightly.
Question: cite the old window (consistent with the published numbers) or switch to the new standard?

### K87-A: what to do when some spectra have no precursor value (parked)
Context: the filtered cross-modal metrics need each query spectrum's measured precursor m/z. Right
now, if ANY row lacks it (0 / NaN), all filtered numbers for that dataset are skipped.
Options: (a) keep skipping; (b) score only rows that have a precursor and report how many were
excluded.

### K88-A: port the mouse-vs-yHydra comparison into the repo (parked)
Context: `pbs/mouse_vs_yhydra.pbs` runs `compare.py` from the shelved `portable_eval` package on
/flare (not in the repo), so it does not get the new with/without-filter split.
Question: port it to the repo's shared filtered-evaluation code (like the yHydra script already was)?

### K89-A: small follow-ups to the filtered alignment evaluation (parked)
(1) `sweeps/package_alignment.py` only reads the old keys; update it so tables/figures can show the
filtered numbers. (2) The during-training validation check (`finetune_align.evaluate_alignment`)
still uses the plain metric; proposal: leave it (it is a health check, not the test evaluation).

---

## Infrastructure / cross-cutting

### K63-I: your review of the job-DAG scheduler (open; reminder requested)
Context: built on branch `i2-dag` (tested only on a simulated batch system). Spec sheet:
`notes/DAG_SPEC.md`. First live trial approved (K84-I). You said you would read it and say whether
you agree with the design.

### K108-A: database search to compete with MSFragger (open, 2026-09-28)
Context: today we only RESCORE the candidate lists MSFragger produces (reranking). To compete with
MSFragger directly we need our own database search: spectrum -> candidate peptide SEQUENCES.
Pieces: (1) a protein database: public (UniProt human reference proteome UP000005640; Swiss-Prot
~20k proteins, more with isoforms) -- ideally the SAME FASTA + contaminants the lab used for the
HEK/HCT116 MSFragger search (ask); (2) in-silico digestion with MSFragger-matching settings (enzyme,
missed cleavages, length range, fixed/variable modifications, charges) -> millions of peptide+charge
entries (pyteomics is installed); (3) embed every entry with the peptide encoder, plus decoys
(reversed sequences) for target-decoy FDR; (4) per spectrum: candidates within the precursor window
MSFragger used, ranked by cosine to the spectrum encoder's embedding; (5) PSMs at 1% FDR vs
MSFragger on the same spectra (psm-rerank-hek-hct116 contains MSFragger's results). Depends on the
peptide encoder (alignment track), which waits for the new C models; the peptide encoder is weak on
unseen data (nine-species open Hit@1 0.39). Proposal: a new PLAN thread + a design card when
alignment resumes; meanwhile optionally prepare the digestion/index code.

### K134-C..K137-C: before "winning C recipe on every pretraining checkpoint" (open, 2026-09-29)
Context: K66-C per-scale search = 4 settings x 3 seeds at the final checkpoint (540,423). 100m/200m scored;
400m (8876832) and 25m (8876833) training, scoring automatic afterwards (yeast ~3 h more). The all-checkpoint
run needs its own card; these must be settled first:
- K134-C selection rule: pick per scale by validation (lr 4e-4 P170xK2 leads at 100m/200m by 0.002-0.004,
  borderline at 200m), or one recipe for all scales? OOD/yeast disagree with validation.
- K135-C consensus in training (C20): not in the K66 grid. It leaves experimental MAP@R unchanged but lifts
  library Hit@1 by ~0.2 (K100). Include it in the final recipe (then it is untested at the K66 settings)?
- K136-C decided: run it (8878032). K137-C decided: no 25m for now.
- K135-C cost (answered 2026-09-29): per run, consensus costs about the same (steps are set by groups, batch
  size unchanged; C20's job ran 7.5 h vs 6.1 h for its reference, different packing). The cost is extra runs:
  (1) consensus variant at the final checkpoint of each scale (50/100/200/400m x 3 seeds = 12 runs, ~1-2
  capacity jobs, <= 14 h, + scoring); (2) consensus on every checkpoint = the all-checkpoint run twice
  (~90 more runs, ~8 capacity jobs, +2-3 days). Proposal: decide the consensus variant with C27 at 50m
  first, then run every checkpoint ONCE in the chosen variant; the K66-C finals stay the experimental-only
  version (final checkpoint only).
Size (rough): ~31 checkpoints x 3 seeds = ~93 runs; 12 runs per node-job -> ~8 capacity jobs of 10-14 h,
2 at a time -> ~2-3 days of queue, plus scoring.

### K139-C follow-up: decision rule and weighting method (open, 2026-09-29)
User: "(c) isn't library search the main metric?" -- Yes: in the approved rule library Hit@1 (validation) is
THE selection metric; experimental-only MAP@R is only a guard (reject an arm whose MAP@R falls by more than
the seed spread). Question: keep the guard, or select on library Hit@1 alone and just report MAP@R?
(b) sampler oversampling vs loss weighting: pros/cons given in chat 2026-09-29; approved card uses the sampler.

### K143-I: bigger allocations / queues (open, 2026-09-29)
Checked `qstat -Qf` + ALCF "Running jobs on Aurora":
- capacity: 1-16 NODES per job, walltime up to 168 h (7 days), 2 running + 5 queued per user, 512 nodes total.
  We have been submitting select=1 with 10-14 h walltimes. aurora-finetune-sweep.pbs already spreads arms over
  several nodes (12 arms per node), so e.g. the all-checkpoint C run (~90 arms) fits in ONE 8-node capacity job.
- prod (routes to small/medium/large): minimum 256 nodes; 10 running per project. UIC-HPC balance 9,265
  node-hours (2,521 used); one 256-node hour = 256 node-hours, so a 24 h prod job would use 2/3 of it.
- legacy / legacy-reg (the new queue): runs the OLD node image (bkc compute_aurora_legacy_20251010), open to
  all, up to 2,413 nodes, 24 h; right now no nodes carry the legacy label (like next-eval after the rollout).
  Useful only as a fallback to the pre-update stack; our env now works on the new image.
Proposal: no prod allocation for current work; use multi-node capacity jobs and longer walltimes. Prod
makes sense only for production-scale pretraining (e.g. Pairformer at 100m+/full MSConsensus, MassIVE-KB),
which would need a larger allocation request. K142-C: 400m 8878459 will likely hit its 14 h walltime --
future capacity jobs should ask for more (limit is 168 h).

### K148-P: full Pairformer pretraining test run -- choices (open, 2026-09-30)
User picked (a)+(c): cap 150 (drop), with and without triangle attention, ~0.25 or 0.5 epoch. Decided: 0.5 epoch, fastest node count (16/arm). Preprocessing 8879887 submitted.
DECIDED 2026-09-30 (K150 approved; runs wait for decoupled streams + validation). Was open (K150-P): fair-comparison design -- transformer's LR schedule (cosine over 540k, stop at ~23k), global
batch 576 (16 nodes x 12 x micro 3), transformer checkpoints evaluated on our validation set; and whether to
run 512 peaks (~650-750 node-h without tri-attn; tri-attn OOMs at 512 today). See the card's update.
Card notes/P2_full_pretrain_card.md. Pick data/cap (a: cap 150 drop, ~80 node-h; b: top-150 peaks all spectra,
~280 node-h; c: + triangle attention ~2x; d: cap 256 ~5x), nodes/walltime, grad checkpointing, probes.
Gated on the K147 leak audit. Dataset MSConsensus-100M (rev 78b3e74) is on /flare.

### K151-P: making the pair update cheaper -- which knobs to screen (open, 2026-09-30)
User: "the pair update needs to be cheaper; what is every knob?" Measured (8880138): one pair update ~= 108 ms
(no tri-attn) / ~364 ms (tri-attn) vs ~131 ms for all 10 single blocks, B=32 N=150. Per-update time split
(K114): triangle mult. out ~35% + in ~33%, write-back ~8%, pair transition ~8%, glue ~3%; tri-attn adds ~45%.
Existing knobs: pair_update_every / pair_bias_lag (how many updates); pair_channels c_z (64; projections and
transition scale ~c_z^2); pair_tri_channels c_t (64); pair_transition_expansion (2); pair_opm_channels c_o
(16; write-back output ~c_o^2 * c_z); pair_use_writeback; pair_update triangle|transition|static;
pair_use_triangle_attention (+ heads, dim); max_peaks N (N^2 everywhere, N^3 in the triangle einsums);
delta_bias_n_freqs (pair-feature init, once). Not yet in code: fewer peaks in the pair stream only (top-k
peaks), sparse pairs, low-rank pair, shared pair weights across updates. Proposal: a speed-only profiling
screen first (no training), then a short training card for the cheapest settings.

### K157-P: is the P2 validation shard held out from the production transformers? (open, 2026-09-30)
The P2 validation set is validation-00000-of-00004 of Gaolaboratory/MSConsensus-100M (rev 78b3e74). The
Pairformer comparison assumes the production transformers never trained on it (they trained on the same
dataset's train split, per the K132 check of master). Confirm with Chris that the validation split was held out
in the production runs; if not, the transformer's numbers on it are optimistic.

### K110-S: checkpoint cleanup on /flare (f, g, a-lite DELETED 2026-09-30; rest open)
Confirm: delete the remaining checkpoint WEIGHTS of the K110a runs too (only finals kept)? K110b/c/d/h/i/j open.
Decide per tier (K110a..K110k in the card). Biggest: K110a old-run intermediate checkpoints 1,055 GiB (or
K110a-lite, optimizer/rng states only, 781 GiB); K110b recent-run checkpoints 417 GiB; K110f crashed FT19
runs 328 GiB. Also: runs/quarantine/checkpoint-13800 (origin unknown); K110c / part of K110d wait for
C20 scoring. Scratch measures 3.57 TiB (not 3.2 TB). Nothing deleted; deletion protocol applies.
Project disk: 8.6 / 10 TB used (UIC-HPC); ours 3.2 TB, of which runs/ = 2.6 TB. User: see how many
checkpoints we have and delete the ones that aren't useful. Proposal: an inventory first (per sweep:
arms, checkpoint dirs, size, whether its results are scored/committed, whether it is a released or
reference model), then a deletion card for approval, executed under the deletion protocol (dry run,
test on a copy, staged).

### C18-C: MassIVE-KB prep (green light 2026-09-29: dry run queued with the card's defaults)
Merged (data/prepare_massive_kb.py, pbs/prepare_massive_kb.pbs, notes/C18_prepare_card.md). Dry run
(2 shards, card defaults) queued 2026-09-29. Before the FULL run, confirm the open choices below
(the dry run does not commit us to them).
Dry run 8877152 (2026-09-29) passed its checks: 184,712 rows in 155,352 peptide groups (~1.19 spectra per group).
K133-C (withdrawn): that group-size figure is meaningless -- the source (massive_kb_v1_shuffled) is shuffled and
the dry run read 2 shards per split = 0.78% of train rows, so a peptide's spectra are mostly outside the
sample. Real group sizes come from the full run's manifest histogram; 40,027 spectra (18,476 sequences) removed for overlap with eval sets.
Finding: in a 2,000-spectrum sample, 18.6% have a peptide sequence that is in one of our evaluation
sets (ms-con-100k val 94, HEK 90, ms-con-100k test 68, human 59, HCT116 50, mouse 27, replicate
corpus 22, OOD 12, yeast 2 of 2,000) -- the prep script removes them. Maybe ms-contrastive-100k is
derived from MassIVE-KB (ask Chris). Open choices: split policy (re-split peptide-disjoint by
default, or keep source splits); oversize spectra (>512 peaks) dropped (default, 4%) or top-512;
cap very large groups?; exclude more sources (full psm-rerank, other nine-species splits)?; the
contrastive trainer needs a new dataset format to read the output (separate card).

### K114-P: decoupled pair / single streams (user's architecture suggestion, 2026-09-28)
Idea (user): run several single-stream attention blocks per pair update, calibrated so they take about
as long as one (triangle-attention) pair update, and run the two concurrently so the pair stack is not
a bottleneck. Partly covered by review ablation #7 (ratio only); the parallel execution is new. Details
and open points: notes/PAIRFORMER_REVIEW.md ablation #12. Proposal: test the ratio first (sequential,
compute-matched); build the parallel version only if a larger ratio doesn't hurt quality. Needs a card
(and code) when Pairformer ablations start.
Update 2026-09-30: code on branch decoupled-streams (`pair_update_every`, `pair_bias_lag`; sequential
only; see notes/PAIRFORMER.md §5). Design choices awaiting confirmation: (a) the update runs on the
FIRST layer of each round (i % k == 0), not the last; (b) skipped layers keep their own bias readout
and do not write back; (c) k that does not divide L (e.g. 3 with L=10) leaves a short last round
(updates at 0,3,6,9) -- allowed, not rejected; (d) at lag 1 the last round's update is not built (one
update fewer than lag 0) and lag 1 needs >= 2 rounds; (e) interface is the ratio k, not a total count.
Rough gain from the K114 profile (gc off, sequential, pair ops a-f + glue removed on skipped layers):
step time x0.54-0.63 at k=2, x0.45-0.56 at k=3, x0.27-0.41 at k=5 (larger N / triattn -> larger gain).

### K116-P / K117-P: which SDPA call layout for triangle attention (K117 bench approved 2026-09-29, being written; K116 open)
Context: triangle attention shares one bias beta_jk across every row i. The 4-D SDPA call (fused
kernel) needs the bias COPIED per row in a chunk (+ summing the copies' gradients back in backward);
the 5-D call passes it broadcast (no copy) but Intel's fused kernel rejects 5-D, so it runs PyTorch's
plain math path (stores the chunk's attention weights). Benchmark 8875808 (fwd+bwd, bf16), 4-D vs 5-D:
B8N100 6.3 vs 6.2 ms; B8N150 14.3 vs 11.0; B32N100 18.2 vs 15.0; B32N150 57.5 vs 44.6 (5-D ~20% faster,
growing with size up to N=150; unknown beyond). Fastest is best only if accuracy (K115 bf16 check,
running) and memory are fine. K117-P proposal (one ~30 min debug job): (1) profiler breakdown -- time of
the mask copy and its backward vs the attention kernel; (2) a copy-free FUSED variant: 4-D mask as a
strided expand() view without contiguous(); (3) 4-D with the mask pre-built outside the timed region
(isolates the copy cost); (4) size sweep N = 200, 256, 512 at small batch (final pretraining likely 512).

### K118-S: the venv's Triton shadows Intel's Triton (open, 2026-09-28)
Context: Aurora's frameworks module ships Intel's Triton 3.6.0 (with the `intel` GPU backend,
/opt/aurora/26.26.0/frameworks/aurora_frameworks-2025.3.1/.../site-packages/triton); our .venv has an
upstream Triton 3.8.0 (nvidia + amd backends only), and the venv comes first on sys.path. So any
Triton kernel, torch.compile or FlexAttention on the XPU cannot work from the venv as is. Options:
(a) job-only override: put the system Triton first on the path for those jobs (venv untouched,
reversible); (b) remove/replace Triton in the venv after checking what needs 3.8; (c) a separate small
venv for kernel work. Suggestion: (a).

### K119-P: our own fused triangle attention (open, 2026-09-28)
Cheapest first: (1) FlexAttention (torch.nn.attention.flex_attention) with a score_mod adding
beta[b,h,j,k] -> compiled fused kernel, fwd+bwd, no bias copy, no stored weights (needs Intel Triton,
K118; XPU support in this PyTorch uncertain) ~1-2 days; (2) torch.compile of the 5-D math path ~1 day;
(3) hand-written Triton flash-attention kernel with pair bias + gating ~1-2 weeks; (4) SYCL kernel,
several weeks. Proposal: after K115 (bf16 check) and K117 (copy cost / size sweep), try K118(a) then (1)
and (2), one short debug job each; (3) only if they fall short.

### C26-C: what is needed for the spectrum-only database search card (open)
(1) Which intensity predictor: the user's "predict all intensities from m/z" model (does one exist yet?
is it Chris's msdelta-intensity work?), our pretrained model with every peak masked, or Prosit as a
stand-in; (2) the FASTA (ideally the one the lab gave MSFragger for HEK/HCT116); (3) which runs and
MSFragger's settings (precursor tolerance, enzyme, modifications) for the comparison.

### K96-S: the pretraining loss, and the input-normalisation leak (parked; user wants to explore other losses)
The pretraining task (msdelta/models/processing_msdelta.py + modeling_msdelta.py):
- Per spectrum, a random subset of peaks is masked: round(mask_ratio x peaks), at least 1
  (mask_ratio 0.50 in the production configs, e.g. configs/msdelta-base-50m; code default 0.15).
- Input per peak: its m/z (always visible) and an intensity feature x = log(1+I) / max over ALL peaks
  of log(1+I) (in [0,1]). For masked peaks the intensity token is replaced by a learned mask token;
  their m/z stays visible.
- Target: the intensity distribution p = I / sum(I) over all peaks, restricted to the masked peaks
  and renormalised to sum to 1 over them.
- Prediction: one logit per peak from the intensity head; softmax over the MASKED peaks only -> q.
- Loss: KL(p || q) = sum over masked peaks of p log(p / q), averaged over the batch. So the model
  learns the RELATIVE intensities of the masked peaks (how the missing intensity is shared among
  them), not their absolute values.
The leak: the input feature's max is taken over all peaks BEFORE masking; if the tallest peak is
masked, no visible peak has x = 1.0, which hints that a masked peak is the tallest.
Options for the leak: (a) caveat; (b) measure it; (c) normalise over visible peaks for new runs only.
Other losses to explore later (user): e.g. regression on log intensity per masked peak, KL on the
full spectrum with visible peaks given, ranking losses, or predicting absolute intensity share.
---

## Contrastive (spectrum encoder)

### K85-P: the first Pairformer vs transformer comparison run (open; see K91-K95 and the review)
Context: `notes/P1_card_draft.md` (branch p1-pairformer) drafts a short debug pretraining run
comparing the ported Pairformer with our transformer at ~50M parameters (loss curves, step time).
It needs choices: size match (options A-D), pair settings, Fourier frequencies, peaks cap, data,
steps/batch/lr. Review: `notes/PAIRFORMER_REVIEW.md`. User asked (2026-09-28): defaults first, or
HP search first? -- answered in chat; decision pending.

### K91-P: mass-defect features are poorly encoded (parked 2026-09-28)
Context: the Pairformer's pair features include the fractional mass of each peak-pair difference
(mass defect), encoded with non-integer log-spaced Fourier frequencies. So a defect of −1 mDa and
+1 mDa look unrelated to the model (feature distance 5.5 vs 0.36 for a 2 mDa step), splitting losses
just below an integer mass (CO, CO2, O) from those just above (H2O, NH3). Inherited from the source.
Options: fix before any comparison (integer frequencies, or feed the signed defect) or run as-is
and test the fix as an ablation.

### K102-P: add per-chunk gradient checkpointing to triangle attention before trying it (open)
Context: triangle attention stores its attention weights for the backward pass: about
3.2 x batch x peaks^3 x heads x 4 bytes, ~5.5 GB per module at 32 spectra x 150 peaks x 4 heads
(two modules per layer). The code's "chunking" only reduces memory without gradients (inference).
User wants to find how to make triangle attention win (ablation #8); that needs this memory fix
first (recompute each chunk in the backward pass; slower, but memory ~ one chunk). Small code
change + a test.

### K95-P: every Pairformer setting must be listed explicitly on the K85-P card (open)
Context: the new pair_* config fields default to tiny test-sized values (pair channels 16, triangle
hidden 16, outer-product 8, triangle attention 2 heads x 8, 16 mass-defect frequencies, loss-dictionary
tolerance 20 ppm). A run config that omits one silently gets the tiny value. The source's experiments
used different values (tolerance 10 ppm, 32 mass-defect frequencies, Fourier 64/0.01/1000).
Proposal: the card lists and the user picks every pair_* value; optionally change the defaults to the
source's 50m values.

### K93-P: memory (explained 2026-09-28; no decision beyond K102-P)
Triangle multiplication keeps ~12 pair-sized tensors per module for the backward pass: at 32 spectra
x 150 peaks x 64 channels that is ~2.3 GB per module, ~45 GB for 10 layers -- more than a tile
comfortably holds with everything else. So gradient checkpointing (recompute in backward, ~30% slower)
is required for the Pairformer, not optional.
