# Project status

The schedule for the denoise fine-tune and the reranking pipeline. Hand-kept, so it says
what is *true right now*; `TODO.md` says what is *wrong and open*, and the two are meant
to be read together.

Regenerate the job table with `pbs/job_history.sh`, which reads the logs rather than
anyone's memory. Last updated: 2026-09-19.

## Where things stand

```
INFRASTRUCTURE ─────────────────────────────────────────────────── all green
  denoise, 1 tile ................. OK   8840190  test AUROC 0.8628
  denoise, 12 tiles DeepSpeed ..... OK   8840264  21.4x one tile
  alignment, 1 tile ............... OK   8840304  full pipeline, saved
  alignment, 12 tiles DeepSpeed ... NO   8840356  GPU fault, see FT9
  grid, 72 arms, 1 tile ........... OK   8840232  72/72
  grid, 216 arms, full pipeline ... ??   8840345  RUNNING  <- gates capacity

SCIENCE ───────────────────────────────────────── nothing real run yet
  216-arm HP grid ................. blocked on 8840323  ~84 node-hrs / ~5 h
  alignment, full 10 epochs ....... ready               ~35 min on 1 tile
  denoise re-runs: 100m, scratch,
    seeds ......................... unblocked, unsubmitted

BUGS (detail in TODO.md) ──────────────────────────────────────────────────
  FT7  GPU fault is DDP ........... worked around with DeepSpeed, NOT fixed
  FT4  32 stray labels ............ DDP gather only; absent under ZeRO-2
  FT3  failure reports "finished" . open
  FT1  >1024 peaks ................ open
  FT5  seeds on the winner ........ waiting on the grid
  FT6  from-scratch control ....... 8839946 is a v1 run, must be redone
  FT8  encoder warm-up freeze ..... deferred, set to 0 everywhere
  FT2  ............................ closed, not a bug
```

## Reranking: the causal chain, validated end to end

The alignment tower was reaching only 2.3x chance at hit@1. The separation eval said why
with a measurement rather than a theory: the frozen pretrained encoder does not separate
peptides. Replicate spectra of one peptide sat at cosine 0.963 and spectra of DIFFERENT
peptides at 0.950 -- an out/in distance ratio of 1.34, essentially no structure. Nothing
in a masked-peak objective ever asked for peptide identity, so the student was imitating
that space faithfully and there was nothing there to imitate.

Fixing the TEACHER fixed retrieval, with no change to the student at all:

| | pretrained teacher | contrastive teacher |
| --- | --- | --- |
| teacher out/in ratio | 1.34 | **7.49** |
| cross-modal hit@1 | 0.0249 (2.3x chance) | **0.0446 (4.2x)** |
| hit@5 | 0.1157 | **0.2288** |
| MRR | 0.0991 | **0.1594** |

`eval_loss` went UP, 0.055 to 0.243, which is the right direction: the old target space
was easy to fit precisely because it was nearly collapsed. The loss was never the metric.

Not yet a usable reranker -- 4.2x chance over 94 candidates is not a reranker -- but the
mechanism is established and the levers are known:

1. **GradCache.** The contrastive encoder trains at batch 4, which is 4 negatives per
   step; a contrastive objective is largely a function of how many negatives it sees.
   Decoupling that from memory is the single biggest lever and the ratio is what drives
   the downstream number.
2. **The tail.** `clean` is still 0.010: some replicate pairs of one peptide stay far
   apart (`worst_in` 0.24 -> 0.84) even as the averages separate. Worth looking at
   whether those spectra genuinely resemble each other before trying to force them
   together.
3. **The student.** Worth tuning now, since the teacher no longer bottlenecks it.

Two findings from the sweep worth keeping. `kl_weight=10` beats `kl_weight=0` on the full
corpus while all six `kl=0` arms won at 740 steps -- the regulariser earns its place only
once there is enough training to overfit, and a smoke test would have locked in the wrong
answer. And rank arms by RATIO, not margin: margin is a difference and rises when a model
merely inflates the space, which one arm did.

## The embedding direction has exhausted its identified levers

Every lever tried, and what it did to the separation ratio (pretrained = 1.34):

| lever | result |
| --- | --- |
| contrastive + KL, batch 4 | 1.34 -> **6.94**, the best figure reached |
| more steps at batch 4 | 6.94 -> 4.82 -> 4.46 across 3, 10 and 50 epochs |
| more negatives (GradCache, batch 64) | best 5.70, BELOW the batch-4 best |
| more steps at batch 64 | 5.70 -> 5.35 -> 4.37 |
| intensity-weighted pooling, frozen | 1.43 -> 1.44, nothing |
| denoiser P(signal) pooling, frozen | 1.34 -> **1.46**, the best readout available, and still nothing next to 6.94 |
| reading layer 8 instead of the output | 1.43 -> **1.53**; the stack peaks at 8 and DROPS at 9 |
| `mean` instead of `mean+max`, frozen | 1.34 -> 1.43, free but small |
| contrastive teacher -> alignment | cross-modal hit@1 +79% |
| that embedding -> reranker | **-0.109 hit@1**, five paired seeds |

Two of these deserve care, because the obvious reading of each is wrong.

**Longer training degrading the ratio is not overfitting.** At batch 4 the contrastive
loss reached 0.4% of chance, so "the task is solved and further steps overfit it" was a
natural explanation, and it is the one recorded earlier. GradCache disproves it: at batch
64 the loss is at 30.6% of chance and still falling -- nowhere near saturated -- and the
ratio falls monotonically anyway. Something about optimising in-batch discrimination
rearranges the space in a way the global separation metric dislikes, independent of
whether the training task has been learned.

**A feature can be strong alone and harmful in a model.** `embedding_cosine` separates
true from decoy pairs at AUROC 0.846, yet adding it costs eleven points of reranking
hit@1 while leaving pooled AUROC untouched. AUROC pools all pairs; hit@1 ranks within a
spectrum; the two diverge when a feature's errors are correlated within a spectrum, and
this one is structurally so -- every candidate for a spectrum is scored against the SAME
cached teacher vector.

**What works instead.** A hand-built feature rescorer reaches hit@1 0.889 on fragment
coverage, mass error and spectrum quality, with no neural embedding at all.

**Read from layer 8, not the output, if this is ever revisited.** Across the ten
encoder blocks the separation ratio climbs monotonically -- 1.37, 1.38, 1.38, 1.41, 1.45,
1.51, 1.51, 1.52, **1.53** -- and then FALLS to 1.43 at the final block. That is the
classic signature of a last layer specialised for its pretraining head: predicting masked
peak intensities is not the same objective as representing peptide identity, and layer 9
has been optimised for the former. Every embedding measured in this repo has been read
from the output. It is a free 7%, and it compounds with training rather than competing
with it. Not acted on now because switching layers invalidates every embedding measured
today for a gain that does not reach the reranker.

Worth noting the whole readout axis together: layer choice, pooling mode and peak
weighting combined move the frozen ratio from 1.34 to about 1.53, roughly +14%.
Contrastive training moves it to 6.94, +418%. Every extraction trick available is about
3% of what training buys, which is the clearest statement of where the information is
not.

**What has not been tried.** The correlated-error structure is a property of the
two-tower formulation, not of embedding quality, so no amount of better embedding fixes
it. A cross-encoder scoring (spectrum, candidate) jointly has no shared per-spectrum
vector and is the natural next formulation -- reranking only scores the top-k candidates,
so it never needed a shared retrieval space in the first place.

## The embedding does not help the reranker, and hurts it

Measured, five paired seeds, both arms sharing each seed's split and initialisation:

| | with embedding | without | contribution |
| --- | --- | --- | --- |
| pairwise AUROC | 0.9061 | 0.9055 | **+0.0006 +- 0.0010** |
| hit@1 | 0.7803 | **0.8893** | **-0.1090 +- 0.0271** |

A hand-built feature rescorer reaches hit@1 0.889 on its own. Adding `embedding_cosine`
leaves pooled AUROC untouched and costs eleven points of hit@1, in every seed.

`embedding_cosine` is not weak on its own -- 0.846 AUROC separating true from decoy pairs.
But AUROC pools all 5,541 pairs while hit@1 ranks WITHIN each spectrum, and the two come
apart when a feature's errors are correlated within a spectrum. That is exactly this
feature's shape: every candidate for one spectrum is scored against the SAME cached
teacher vector, so when that vector is poor it misleads all of that spectrum's candidates
together. Fragment features are computed against the observed peaks per candidate, and
their errors do not line up that way.

So the embedding injects spectrum-level noise into precisely the comparison reranking
depends on. Whether the alignment tower can be rebuilt to avoid that is open; what is
settled is that its current output should not be a reranker feature.

**What this costs.** The contrastive work raised the separation ratio 1.34 -> 7.49 and
cross-modal hit@1 by 79%, and none of it reaches the reranker. The feature classifier
was already in the repo on `sweep/pairformer-aurora`, with `separation.py` stating the
prerequisite question and `negatives.py` the hard-negative problem, before any of today's
embedding work started.

## The one real number so far

`test AUROC 0.8628` from a 700-step denoise run (8840190), against a free raw-intensity
baseline of about 0.75. Everything else this session was infrastructure. **No tuned model
and no reranking result exists yet.**

## What has to happen, in order

1. **8840323 passes** -> submit the 216-arm grid to capacity. ~84 node-hours, ~5 h on 16
   nodes. Axes: `learning_rate` x `encoder_lr_scale` x `num_train_epochs` x
   `head_hidden_size` x effective batch.
2. **Full alignment training, one tile.** Twelve tiles faults under both backends
   (FT9), so one tile at batch 4 is the only proven path: ~35 min, which fits the debug
   hour if `eval_steps` goes from 200 to 2000. At the current 200 the 143 evals would eat
   28 of those minutes. The real fix is to precompute the frozen teacher's embeddings,
   which is cheaper anyway -- see FT9.
3. **Grid winner** -> FT5 (seeds) and FT8 (freeze), both of which need it.
4. **Denoise re-runs** on the fixed pipeline: 100m, from-scratch control, seeds.
5. **Reranking end to end**: features + embedding distance -> classifier, which needs a
   trained alignment tower first.

## Conventions worth not rediscovering

- **Runs are prefixed `v2_`.** Anything without it predates the DDP, dtype and eval fixes
  and is not comparable. `RUN_PREFIX` in both launchers and the generator.
- **Validate on debug before capacity**, and validate with `MAX_SAMPLES`, not `MAX_STEPS`.
  Capping steps skips saving, `save_total_limit`, `load_best_model_at_end`, the test split
  and the final save -- which is exactly where the alignment bugs were hiding.
- **DDP is broken on this stack.** One tile, or DeepSpeed ZeRO-2 on twelve. The sweep
  launcher refuses multi-tile arms that name no deepspeed config.
- **Every experiment carries a description**: auto-derived from settings into W&B notes
  and `RUN.md`, plus a hand-written `DESCRIPTION.md` beside each args file for intent.
- Jobs read configs from a snapshot taken at job start, so the working tree can be edited
  while a sweep runs.

## Job history

Regenerate with `pbs/job_history.sh`.

```
JOB       TASK     PARALLELISM   OUTCOME                    NOTE
8840154   denoise  12 tiles DDP  GPU FAULT at 168/700
8840190   denoise  1 tile        COMPLETE 700/700           test AUROC 0.8628
8840223   align    12 tiles DDP  GPU FAULT at 3/200
8840232   grid     1 tile/arm    72/72 arms ok
8840238   align    1 tile        GPU FAULT at 73/200        batch 16
8840257   align    1 tile        ERROR in eval              bf16 vs fp32 fused kernel
8840264   denoise  12 tiles DS   COMPLETE 300/300           21.4x, 0 faults
8840277   align    1 tile        ERROR after eval           no eval_loss
8840291   grid     12 tiles DS   refused: stale arms        guard worked
8840304   align    1 tile        COMPLETE 400/400           eval_loss 0.048
8840313   grid     12 tiles DS   refused: stale arms        guard worked
8840323   grid     12 tiles DS   RUNNING                    216-arm full pipeline
8840336   align    12 tiles DS   RUNNING                    does ZeRO-2 work here
```
