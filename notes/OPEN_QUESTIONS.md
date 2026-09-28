# Open questions

Every question Claude raises in chat is also written here, with enough context to be read
on its own later. When the user decides, the entry moves to `DECISIONS.md` (with the decision)
and is deleted here. IDs: running number + track suffix (-C contrastive / spectrum encoder,
-A alignment / peptide encoder, -D denoise, -R rescoring, -I infrastructure, -P Pairformer,
-S cross-cutting).

Status: **parked** = user said to come back later; **open** = waiting on the user.

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

### K106-S: stop branching? (open, 2026-09-28)
Context: the extra branches (reorg, p1-pairformer, i2-dag, a-filtered-eval, c25-library-search,
k94-strict-load, k90-code-ref, c18-massivekb-prep) were created by Claude, one per background
agent, in separate git worktrees. Two reasons: (1) before per-job code snapshots existed, jobs ran
code straight from the main checkout, so editing it could break running jobs; (2) several agents
editing the same checkout at once would overwrite each other's files and mix unrelated half-done
work into commits. (1) is solved (snapshots); (2) still holds when agents run in parallel.
Options: (a) keep one branch per parallel agent but MERGE into dev_finetune_02 as soon as its
tests pass and you've seen the report (no waiting); (b) no branches: agents work one at a time
directly on dev_finetune_02 (slower, no parallelism); (c) current practice (branches wait for
explicit approval to merge).

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

### K107-P: approve the Stage 0 card (open, 2026-09-28) -- notes/P1_stage0_card.md
Context: Stage 0 (user: "we should run it") = one short debug pretraining run of each architecture,
only to check the Pairformer port trains and to measure step time / memory. Full card:
notes/P1_stage0_card.md; architecture: notes/PAIRFORMER.md. Arms: Pairformer 512x10, 8 heads, FFN 2048,
pair/triangle 64, write-back 16, triangle attention off (46.1M params; every pair setting taken
from the source's pairformer-sweep-50m config, cited by line) vs our msdelta-base-50m transformer
(49.8M) unchanged. Both: peaks cap 150, 300 steps, global batch 512, lr 1.3e-4 cosine, warmup 11,
bf16, mask ratio 0.5, seed 0; gradient checkpointing on for the Pairformer. Debug queue, ~25-35 min
per Pairformer job (estimate). Decisions needed:
(1) DATA: our pretraining corpus MSConsensus-100M (190 GB) is not in our HF cache (Chris's cache is
    not readable). Proposal: download a small pinned subset (~2.6 GB: 4 train + 1 validation shard)
    on the login node, then one debug preprocessing job at cap 150.
(2) PEAKS CAP 150 DROPS SPECTRA: our processor drops any spectrum with more than 150 peaks (the
    source instead kept the 150 most intense after a 1% threshold); median spectrum has ~205 peaks,
    so over half are dropped and the kept set skews to short spectra. Same for both arms, so Stage 0
    is still fair; matching the source needs a code change (own card). Accept for Stage 0?
(3) W&B: pbs/aurora-pretrain.pbs does not load the W&B key; add the one standard line
    (pbs/load_keys.sh) or run with --report_to none.
(4) Settings the agent had to choose (not in the source): warmup 11 (keeps the source's 3.6%
    warmup fraction), 300 steps, logging every 10 steps, saves/probes off, transformer micro-batch 32
    (its standard is 64), which data shards, OOM fallback micro 16 x accum 4. Approve or change.
(5) K91-P: run with the source's mass-defect encoding (as carded) or fix it first.
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
