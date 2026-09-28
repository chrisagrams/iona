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
- **C18-C** -- review the MassIVE-KB prep (merged, not run; 18.6% overlap finding; open choices below).
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

### K110-S: checkpoint cleanup on /flare (open)
Project disk: 8.6 / 10 TB used (UIC-HPC); ours 3.2 TB, of which runs/ = 2.6 TB. User: see how many
checkpoints we have and delete the ones that aren't useful. Proposal: an inventory first (per sweep:
arms, checkpoint dirs, size, whether its results are scored/committed, whether it is a released or
reference model), then a deletion card for approval, executed under the deletion protocol (dry run,
test on a copy, staged).

### C18-C: MassIVE-KB prep -- review later (parked; reminder requested)
Merged (data/prepare_massive_kb.py, pbs/prepare_massive_kb.pbs, notes/C18_prepare_card.md). NOT run.
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

### K116-P / K117-P: which SDPA call layout for triangle attention (open, 2026-09-28)
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

### K100-C: run the library-search evaluation on C20 now? (open; terminology clarified 2026-09-28)
Three different search settings (all "identify the peptide behind a spectrum"):
- replicate retrieval (our current contrastive metric): query spectrum vs other EXPERIMENTAL spectra;
  several correct answers (the same peptide's other replicates).
- spectral LIBRARY search (C25): query spectrum vs a library of CONSENSUS spectra, one per
  peptide+charge; exactly one correct answer; the peptide is read off the matched entry.
- DATABASE search (what search engines like MSFragger do): query spectrum vs candidate PEPTIDE
  SEQUENCES (from a protein database), scored against theoretical/predicted spectra; our analogue is
  the alignment cross-modal evaluation (spectrum encoder vs peptide encoder).
Library search is close to replicate retrieval but not the same: the gallery is one clean consensus
per peptide instead of several noisy replicates, and there is exactly one correct entry (so Hit@k
and rank statistics, not MAP@R). Code is ready (branch c25-library-search). Question: approve
scoring the C20 models (validation, test) with it now, running from that branch (K90-S mechanism,
once built) without merging?
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
