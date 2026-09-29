# Narrative (story so far → where we're going)

Keep this under a page. Append a line per chapter change; details live in OBSERVATIONS / DECISIONS.

## 1. Up to the paper (→ 2026-09-26, branch `dev_finetune`, frozen)
- **Denoise (D):** scales with size, pretraining buys ~+0.05 AUROC everywhere, gains saturate by ~330k steps. Done.
- **Contrastive (C):** pretraining buys everything (random init = chance). Recipe then: two-stage, replicate
  corpus → ms-contrastive-100k (C7); 400m best (test MAP@R 0.868).
- **Alignment (A):** a peptide encoder trained to match the C encoder retrieves the right peptide
  ~92% of the time; beats yHydra.
- **Rescoring (R):** encoders add a little on top of strong rescoring; benchmark issues unresolved.

## 2. After submission: rebuild the contrastive recipe (2026-09-26 → 28, branch `dev_finetune_02`)
- Simpler **single-dataset recipe** (ms-contrastive-100k only, 3 epochs): SupCon, **same-mass batches**
  (C19, GLEAMS-inspired), temperature 0.002, KL 10; new best **lr 4e-4, 128 groups × 2 spectra** at 50m.
- Tested and rejected: sigmoid loss (C8), batch mixes / two mass regions (C21).
- **Consensus spectra in training (C20):** experimental retrieval unchanged, but library search
  (query vs consensus library) jumps 0.71 → 0.94 Hit@1 (K100).
- **Evaluation upgraded:** every number with/without precursor filter (none / 20 ppm / isotope-tolerant),
  on filter passes and failures; library search; unseen species (mouse, human, yeast, 8-species OOD).
- **Infra:** per-job code snapshots, run-from-commit, job DAG, strict checkpoint loading, package reorg.
- **Pairformer (P1):** ported, reviewed, profiled (triangle multiplications dominate; SDPA default).

## 3. The per-scale search, and the Aurora update (2026-09-28 → 29)
- **K66-C:** 4 settings × 3 seeds at every scale, final checkpoint. 100m/200m: lr 4e-4 + 170×2 leads on
  validation, but OOD/yeast disagree. 25m done 2026-09-29.
- **Aurora's 2026-09 update broke every multi-process launch** (env modules, CCL MPI init, a 109-char TMPDIR).
  Jobs sat "running" but never trained — ~a day of 400m/25m/50m compute lost before it was caught.
  All three causes fixed and guarded by tests; lesson: check loss lines, not the queue.
- **Stage 0 (Pairformer vs transformer, 300 steps):** Pairformer learns (0.90 → 0.32); transformer arm
  flat at 0.83 — suspicious, being checked (K141).

## 4. Now (2026-09-29 evening)
- Running: K66-C **400m** (may hit its 14 h walltime → resume ready), **C27** (consensus weighted
  more heavily, 50m, 12 arms), **K136** (missing 50m cell) queued.
- Found: capacity allows **16 nodes / 7 days per job** — the next big runs go as single multi-node jobs.

## 5. Where we're going
1. **Pick the C recipe** (you): per-scale winner (K134), consensus or not (K135, informed by C27).
2. **Scaling curve for C:** winning recipe on every pretraining checkpoint × every scale (no 25m) —
   one multi-node capacity job, ~14 h. → answers "does C scale with size and pretraining?"
3. **Alignment (A)** resumes on the new C models (caveats list waiting).
4. **Pairformer (P2):** matched comparison with the transformer once Stage 0 is understood.
5. Later: MassIVE-KB training data (C18), spectrum-only database search (C26), rescoring (R3-R5),
   move to the new PyTorch stack between phases (I3).

## Branch map
`master` (Chris, untouched) · `dev_finetune` (paper, frozen) · `dev_finetune_02` (all current work; side
branches `stage0-prep`, `k114-k119-prep`, `k117-prep`, `c27-prep` are merged into it).
