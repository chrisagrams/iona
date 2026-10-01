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

### K161-C (parked: user "add that as an option but don't pursue for now") -- scale BOTH sets to every checkpoint?
Option: after K160-C (consensus twins of the existing models) is compared, run with- AND without-consensus on
every pretraining checkpoint x scale. Without consensus: resume K155 (RESUME_JOB=8880712; 48 arms have
checkpoints, 21 50m arms not started). With consensus: the remaining ~69 twins (~80 node-h). The user expects to
go forward with consensus ("we probably will then go forward with the consensus runs"), so the likely alternative
is consensus-only on the remaining checkpoints and K155 left paused.

### K189-P (open) -- K187 "full schedule": on which data, against which transformer?
transformer-50m (Chris, configs/msdelta-base-50m/training.args): MSConsensus-100M (all shards), 3 epochs = 540,423
steps, no 150-peak cap. P2 data (all our Pairformer runs): shards 0-199 at cap 150 (drop), 13,470,623 spectra = one
pass in 23,387 steps; 540,423 steps on it = ~23 passes. So "full schedule" on P2 data repeats data ~8x more than the
transformer did, on a different (capped) subset: the 0.055 comparison would be confounded. Options:
(A) P2 data, 540k steps (cheapest; repetition + data mismatch);
(B) a cap-150 build of all shards (preprocessing job; still drops >150-peak spectra the transformer saw);
User follow-up: "If we do MSConsensus-100M, all shards, with a peak cap of 512 like chris' ..., for 3 epochs, how much time
... All the trained models should be trained like this for a fair comparison. ... why not just use chris' pretraining data?"
-> agreed: on Chris's exact recipe his transformer-50m IS the baseline (no twin). Facts: only ~27% of spectra have
<=150 peaks (p2-cap150-half README), so at cap 512 most spectra are long; triangle multiplication scales ~N^3, pair
readouts ~N^2. Rough estimate for 10 x 640, 1 update, 540k steps: ~0.8-1.8 s/step on 2 nodes -> ~5-11 days, ~250-550
node-h per model (needs a debug cost probe at cap 512: s/step + memory + peak-count histogram).
(C) (superseded) also a transformer-50m twin trained identically (same data/cap/steps/compile) -- the clean baseline; ~half
    the Pairformer's cost. Recommendation: (C) with (A) or (B). K187 not submitted until decided.

### K188-C (card PROPOSED 2026-10-01, awaiting approval) -- all-checkpoint scaling WITH consensus
User: "K184-C: So we should do the all-checkpoint scaling using the with consensus runs. Have already finished that
scaling for non-consensus? How much time would it take to finish with consensus?"
Status of K155 (no consensus): paused, 27/78 done (100m 220k/330k/430k, all 18 200m). In-flight when paused:
50m 18 arms at 17-33% (ck 265/530 of 1,593), 100m 9 arms at 67-83%, 400m 21 arms at 67%; resumable (RESUME_JOB=8880712).
Remaining ~16 node-h (~4-5 h on 4 nodes) by the measured per-arm costs; none of the 78 is scored yet.
Proposal: 78 consensus twins (50m/100m/200m/400m x 26 non-final checkpoints x 3 seeds) = each K155 arm's training.args
+ ONLY --include_consensus true --consensus_weight 1 (as K160/K163, which the finals used); K168-C cache fix included.
Est. ~65 node-h training (K155's ~60 x 1.075 for consensus's extra steps), one 4-node capacity job ~16 h, after a debug
smoke; scoring on the six sets ~3-4 h per set (78 arms; 15 arms took 27-48 min per set).
Needs: (i) approve; (ii) also resume and score the no-consensus K155 remainder (~16 node-h + scoring)? (iii) priority:
capacity runs only 2 of my jobs at a time -- C (16 h job) vs P (K185/K186 five ~1.5 h jobs, K187 two ~20 h jobs).
### K187-P (shape still open)
User: "The regular transformer model uses 10x640? why?" The transformer ladder (configs/msdelta-base-*, Chris's) keeps
width ~= 64 x depth with 64-dim heads: 8x512 (25m), 10x640 (50m), 13x800 (100m), 16x1024 (200m), 20x1280 (400m); 1b
breaks it (20x2048). P2's 10x512 (45M) was our own choice, below the 50m rung. Options: Pairformer 10x640 (the same
single stream as transformer-50m + the pair stream; cleanest comparison) or 10x512.

### K185-P / K186-P / K187-P (cards PROPOSED 2026-10-01, awaiting approval) -- how useful are pair updates?
User (K180-P reply): "start looking at option b ... try dropping them ... go to 20 layers, and test 1 vs 2 vs 4 pair
updates ... It's slower but we need the data ... later on the models with pair updates [may] generalize better ... train
two 50m models to 'completion', one with k=1 and one with k=2/4". All on the K180 setup (P2 recipe/data/masks, lag 0,
compiled, 2 nodes x micro 24, global 576, stop 23,387, eval_mlm digest 89e46099fdba) unless stated.
K185-P no pair updates (pair_update=static: initial pair state, per-layer bias readouts, never updated):
  L10 static (vs L10 k10 0.0839) and L20 static (anchor for K186). ~1.3 + ~2.5 node-h.
K186-P 20 layers (hidden 512, ~90M params), 1 / 2 / 4 pair updates = pair_update_every 20 / 10 / 5 (+ L20 static
  from K185 as 0 updates). ~2.5 node-h each, ~7.5 total. Reading: "k=1,2,4" = number of pair updates, not
  pair_update_every (k=1 there would be 20 updates) -- confirm.
K187-P two "50m" models to completion, 1 update vs 2 or 4 (picked from K186). Needs: (i) shape -- P2's L10 x 512
  (45M) or the transformer-50m shape L10 x 640 (~50M); (ii) "completion" = the transformer-50m's full 540,423-step
  schedule on the same data (~11.5x P2's length; est. 15-25 h on 2 nodes, ~30-50 node-h each, needs resume across
  jobs) or something shorter; (iii) compare against transformer-50m final (P2 validation eval_mlm 0.055).
Still open from K182-P: (a) seed replicates; (d) transformer baselines under this exact setup.

### K182-P (open) -- one pair update beats more: what to test next?
K180: L10 k10 (one pair update) 0.0839 beats L10 k3 0.0903 and k5 0.0937 (same setup) and L14 k7 0.0856, at the lowest
cost (39 min, 2 nodes). The lead is steady from step ~6k (not noise). Candidate follow-ups (each ~1-1.5 node-h, 2 nodes,
compiled, P2 recipe): (a) seed replicate of k10 and k5 (is the ranking stable?); (b) k = inf: no pair update at all, only
the initial pair state + per-layer bias readouts (does the update matter?); (c) L14 k14 (one update, deeper);
(d) a 14-layer transformer baseline (K166) and a transformer under this exact setup (the ~0.096 baseline came from a
different setup). Needs: which of these.

### K172-P (open) -- make the two streams actually run in parallel
User (K169): "the entire point we do this is to run the two streams in parallel." Current state: the code has
pair_update_every (k) and pair_bias_lag (K114), and lag 1 removes the dependency so the pair update COULD overlap the
single blocks, but concurrent execution is NOT implemented (pairformer.py "What true concurrency would still
need"); every P2 run used lag 0 and ran the two streams one after the other. Proposal: (1) implement same-tile
two-stream execution at lag 1 (side XPU stream per round, events, record_stream; check XPU overlaps kernels incl.
backward), CPU/equivalence tests on login; (2) debug-node timing: k5 lag 1 overlapped vs sequential, micro 24;
(3) if it overlaps, a P2-length k5 lag-1 run to measure the loss cost of the one-round-stale bias. Balance target:
pair update time ~= k single layers (k ~7 at large per-tile batch, K169). Needs: go-ahead.

### K166-P (open) -- more pair updates / deeper Pairformer: which arms?
User: "here we're only doing 2 pairwise updates, I want to see what happens when there are more" and "Can we try an
intermediate model with 14 layers where k=7 and so we have 2 pair updates?" Done so far (P2 recipe, 10 layers,
hidden 512): k5 = 2 updates 0.0926, k1 = 10 updates 0.0907. Proposal (P2 recipe otherwise: c_z 32, outgoing,
factored write-back, no tri-attn, 23,387 steps, same data/masks, eval_mlm; 2 nodes x micro 24; torch.compile if
K167-I validates): A = 14 layers k7 (2 updates, the user's); B = 15 layers k5 (3 updates); C = 20 layers k5 (4 updates).
A isolates "more single layers at the same pair cost"; B/C add pair updates at the k5 spacing. ~2-4 node-h each.
Needs: which arms, and whether a 14-layer transformer baseline is wanted for matched comparison.

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
