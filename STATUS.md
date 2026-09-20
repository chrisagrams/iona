# Project status

The schedule for the denoise fine-tune and the reranking pipeline. Hand-kept, so it says
what is *true right now*; `TODO.md` says what is *wrong and open*, and the two are meant
to be read together.

Regenerate the job table with `pbs/job_history.sh`, which reads the logs rather than
anyone's memory. Last updated: 2026-09-20.

## Where things stand

```
INFRASTRUCTURE ────────────────────────────────────────── green except FT9
  denoise, 1 tile ................. OK   8840190  test AUROC 0.8628
  denoise, 12 tiles DeepSpeed ..... OK   8840264  21.4x one tile
  denoise 200m, 12 tiles .......... OK   8841966  peak 7.66 of 68.7 GB, 0 faults
  alignment, 1 tile ............... OK   8840304  full pipeline, saved
  alignment, 12 tiles DeepSpeed ... NO   8840603  GPU fault, see FT9
  grid launcher, full pipeline .... OK   8841345  12/12 arms incl. test split

SCIENCE ─────────────────────────────── denoise delivering, embedding does not
  50m  grid, 216 arms ............. DONE            best test AUROC 0.9320
  100m grid, 12 arms .............. DONE            best test AUROC 0.9403
  200m grid, 12 arms .............. QUEUED 8841992  memtest passed, ~4 h
  from-scratch ablation, 12 arms .. QUEUED 8841984  isolates pretraining's worth
  layer/scale readout probe ....... DONE   8841973  39 configs, ceiling 1.53
  feature rescorer ................ DONE            hit@1 0.889, no embedding
  neural embedding -> rescorer .... DEAD            -0.109 hit@1, 5 paired seeds

BUGS (detail in TODO.md) ──────────────────────────────────────────────────
  FT9  12-tile alignment fault .... 8 hypotheses dead, parked for ALCF
  FT7  GPU fault is DDP ........... worked around with DeepSpeed, NOT fixed
  FT4  32 stray labels ............ DDP gather only; absent under ZeRO-2
  FT3  failure reports "finished" . open
  FT1  >1024 peaks ................ open
  FT5  seeds on the winner ........ ready, grid has a winner now
  FT6  from-scratch control ....... running as 8841984
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

## Denoise: the grid is finished and 100m beats 50m

216 arms at 50m, then 12 at 100m narrowed by what the first grid measured.

| | best test AUROC |
| --- | --- |
| free raw-intensity baseline | ~0.75 |
| first 700-step run | 0.8628 |
| 50m grid, 212 arms | 0.9320 (`lr2e4_es05_ep4_h512_b12`) |
| **100m grid, 12 arms** | **0.9403** (`lr2e4_es05_b12`) |

What the 50m grid established, by mean AUROC across arms:

| axis | effect |
| --- | --- |
| `encoder_lr_scale` | 0.760 frozen -> 0.872 at full rate. The largest factor by far |
| `learning_rate` | 0.776 -> 0.885 at 2e-4, monotone |
| batch | b144 0.813, b48 0.831, b12 0.849 |
| `num_train_epochs` | 0.824 -> 0.838 |
| `head_hidden_size` | 0.828 / 0.832 / 0.834 -- nothing |

**Fine-tuning the encoder is the whole game.** Frozen arms average 0.760, barely above the
free baseline, so what the pretrained model contributes comes from ADAPTING it rather
than from its features as they stand -- the same conclusion the reranking work reached
from the opposite direction.

Three things the 100m grid settled that the 50m grid could not:

- **2e-4 is a real interior optimum.** It was the best value AND the boundary at 50m, so
  the 100m grid extended to 5e-4, which is worse (0.9332 against 0.9392). Extending was
  the point of including it.
- **Keeping 5e-5 was sound reasoning and a wrong hypothesis.** Larger models often prefer
  lower rates; this one does not.
- **The b12 advantage is 50m-specific.** It beat b48 by 0.018 at 50m and loses to it by
  0.001 at 100m, so "more optimizer steps win" was about to become folklore on one data
  point.

The whole 100m grid spans 0.019 against the 50m grid's 0.26. Once the encoder is unfrozen
at a sane rate, the larger model barely cares about these hyperparameters, which is the
strongest argument that narrowing the grid was right.

## The embedding direction has exhausted its identified levers

Every lever tried, and what it did to the separation ratio (pretrained = 1.34):

| lever | result |
| --- | --- |
| contrastive + KL, batch 4 | 1.34 -> **6.94**, the best figure reached |
| more steps at batch 4 | 6.94 -> 4.82 -> 4.46 across 3, 10 and 50 epochs |
| more negatives (GradCache, batch 64) | best 5.70, BELOW the batch-4 best |
| more steps at batch 64, to 17,893 | 5.70 -> 5.35 -> 4.37 -> 4.34 -> **3.35**, monotone |
| intensity-weighted pooling, frozen | 1.43 -> 1.44, nothing |
| denoiser P(signal) pooling, frozen | 1.34 -> **1.46**, the best readout available, and still nothing next to 6.94 |
| reading layer 8 instead of the output | 1.43 -> **1.53**; the stack peaks at 8 and DROPS at 9 |
| scaling the encoder 50m -> 100m -> 200m, frozen | output 1.43 -> 1.36 -> 1.35; best-block 1.53 -> 1.41 -> 1.44. No gain, but all three checkpoints are at ~1 epoch of a 3-epoch schedule, so this is confounded with undertraining |
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

**The readout axis, measured exhaustively: 39 configurations, 3 scales.** Every
encoder block of the 50m, 100m and 200m pretrained checkpoints, plus four pooling modes
on each output, all frozen and training-free. The probe validates itself: at every scale
the last block's ratio equals the `mean`-pooled output ratio, as it must.

| scale | blocks | output `mean` | best block | best ratio | depth of best |
| --- | --- | --- | --- | --- | --- |
| 50m  | 10 | 1.43 | 8  | **1.53** | 89% |
| 100m | 13 | 1.36 | 11 | 1.41 | 92% |
| 200m | 16 | 1.35 | 6  | 1.44 | 40% |

Two things fall out, and the second is the one that matters.

**The mid-stack hump is real and universal.** No scale peaks at its output. The ratio
climbs through the stack and then drops in the last one to three blocks -- 50m
1.37/1.38/1.38/1.41/1.45/1.51/1.51/1.52/**1.53**/1.43, 200m rising to **1.44** at block 6
and sagging to 1.35 by block 15. That is the signature of a last layer specialised for
its pretraining head: predicting masked peak intensities is not the same objective as
representing peptide identity, and the final blocks have been optimised for the former.
Reading one block earlier is free.

**Scale does not help the frozen embedding, and MAY hurt it -- confounded.** The
output-layer ratio falls monotonically with capacity (1.43, 1.36, 1.35) and the best
block over the whole stack belongs to the smallest model. The tempting reading is "scale
hurts". It is not supported yet, because all three checkpoints are `production-01`, and
reading their pretraining state says why:

| scale | step | of schedule | epoch | pretrain loss | still falling? |
| --- | --- | --- | --- | --- | --- |
| 50m  | 180,000 | 33.3% | 1.00 | 0.0705 | yes, -0.0012 / 10% |
| 100m | 190,000 | 35.2% | 1.06 | 0.0639 | yes, -0.0013 / 10% |
| 200m | 192,799 | 35.7% | 1.07 | 0.0558 | yes, -0.0012 / 10% |

None of them is converged -- every one is about a third of the way through a 540,423-step
schedule and still descending at the same rate. And the pretraining loss orders correctly
by capacity, so the larger encoders are straightforwardly better at the objective they
were trained on. A bigger model can be better at its objective while its frozen
replicate-identity structure, which nothing ever optimised, has simply not emerged yet.
Matched schedule FRACTION is not matched convergence, and larger models generally need
more data to reach a given representation quality.

So the honest statement is narrower: **at ~1 epoch of pretraining, scale buys nothing for
the frozen embedding.** Whether it would at 3 epochs is untested.

Two things settle it, and both are cheap:

  The checkpoint sweep. Every scale kept checkpoints from 10,000 to ~190,000, so the
  same training-free probe can be run along the pretraining trajectory. If the ratio is
  still climbing at the last checkpoint -- especially if the 200m is climbing faster --
  the number above is an artefact of reading too early. If it is flat, it is not.

  The 200m denoise grid (8841992, queued). Fine-tuning is the control the frozen probe
  lacks. The 100m checkpoint already fine-tunes BETTER than the 50m (0.9403 vs 0.9320)
  despite a worse frozen ratio, which alone shows the frozen readout is not measuring
  encoder quality. If the 200m fine-tunes better again, the encoders are fine and only
  the frozen readout degrades; if it fine-tunes worse, undertraining is real and reaches
  the fine-tuned numbers too.

`clean` -- the fraction of groups whose every replicate is nearer to its own group than
to anything outside it -- sits at 0.010 for essentially all 39 configurations, meaning 1
of 99 groups, and at 0.000 for the earliest blocks. No layer of no model at no scale
separates replicates.

So the whole readout axis -- layer choice, pooling mode, peak weighting, and now model
scale -- spans 1.29 to 1.53. Contrastive training reaches 6.94. Every extraction trick
available, across three model sizes, is about 3% of what training buys, which is the
clearest available statement of where the information is not.

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

Regenerate with `pbs/job_history.sh`. Full probe results in `results/`.

```
JOB       TASK     PARALLELISM OUTCOME                    NOTE
8816450   denoise  8 tiles     ran to 40/40               
8821044   denoise  ?           no training                
8821246   denoise  8 tiles     ERROR: OSError: libmkl_intel_lp64.so.2: cannot  
8821247   denoise  8 tiles     ERROR: OSError: libmkl_intel_lp64.so.2: cannot  
8821248   denoise  8 tiles     ERROR: OSError: libmkl_intel_lp64.so.2: cannot  
8821249   denoise  8 tiles     ERROR: OSError: libmkl_intel_lp64.so.2: cannot  
8821250   denoise  8 tiles     ERROR: OSError: libmkl_intel_lp64.so.2: cannot  
8821251   denoise  8 tiles     ERROR: OSError: libmkl_intel_lp64.so.2: cannot  
8821272   denoise  8 tiles     ERROR: OSError: libmkl_intel_lp64.so.2: cannot  
8821273   denoise  8 tiles     ERROR: OSError: libmkl_intel_lp64.so.2: cannot  
8821275   denoise  8 tiles     ERROR: OSError: libmkl_intel_lp64.so.2: cannot  
8821276   denoise  8 tiles     ERROR: OSError: libmkl_intel_lp64.so.2: cannot  
8821277   denoise  8 tiles     ERROR: OSError: libmkl_intel_lp64.so.2: cannot  
8821278   denoise  8 tiles     ERROR: OSError: libmkl_intel_lp64.so.2: cannot  
8821285   denoise  8 tiles     ERROR: OSError: libmkl_intel_lp64.so.2: cannot  
8821292   denoise  8 tiles     ERROR: OSError: libmkl_intel_lp64.so.2: cannot  
8821293   denoise  8 tiles     ERROR: OSError: libmkl_intel_lp64.so.2: cannot  
8821295   denoise  8 tiles     ERROR: OSError: libmkl_intel_lp64.so.2: cannot  
8821296   denoise  8 tiles     ERROR: OSError: libmkl_intel_lp64.so.2: cannot  
8821297   denoise  8 tiles     ERROR: OSError: libmkl_intel_lp64.so.2: cannot  
8821309   denoise  8 tiles     ERROR: OSError: libmkl_intel_lp64.so.2: cannot  
8821311   denoise  8 tiles     ERROR: OSError: libmkl_intel_lp64.so.2: cannot  
8821312   denoise  8 tiles     ERROR: OSError: libmkl_intel_lp64.so.2: cannot  
8821314   denoise  8 tiles     ERROR: OSError: libmkl_intel_lp64.so.2: cannot  
8821315   denoise  8 tiles     ERROR: OSError: libmkl_intel_lp64.so.2: cannot  
8821316   denoise  8 tiles     ERROR: OSError: libmkl_intel_lp64.so.2: cannot  
8821326   denoise  8 tiles     ERROR: OSError: libmkl_intel_lp64.so.2: cannot  
8821327   denoise  8 tiles     ERROR: OSError: libmkl_intel_lp64.so.2: cannot  
8821328   denoise  8 tiles     ERROR: OSError: libmkl_intel_lp64.so.2: cannot  
8821329   denoise  8 tiles     ERROR: OSError: libmkl_intel_lp64.so.2: cannot  
8821330   denoise  8 tiles     ERROR: OSError: libmkl_intel_lp64.so.2: cannot  
8821331   denoise  8 tiles     ERROR: OSError: libmkl_intel_lp64.so.2: cannot  
8821345   denoise  8 tiles     ERROR: OSError: libmkl_intel_lp64.so.2: cannot  
8821347   denoise  8 tiles     ERROR: OSError: libmkl_intel_lp64.so.2: cannot  
8821349   denoise  8 tiles     ERROR: OSError: libmkl_intel_lp64.so.2: cannot  
8821351   denoise  8 tiles     ERROR: OSError: libmkl_intel_lp64.so.2: cannot  
8821352   denoise  8 tiles     ERROR: OSError: libmkl_intel_lp64.so.2: cannot  
8821353   denoise  8 tiles     ERROR: OSError: libmkl_intel_lp64.so.2: cannot  
8821363   denoise  8 tiles     ERROR: OSError: libmkl_intel_lp64.so.2: cannot  
8821364   denoise  8 tiles     ERROR: OSError: libmkl_intel_lp64.so.2: cannot  
8821365   denoise  8 tiles     ERROR: OSError: libmkl_intel_lp64.so.2: cannot  
8821366   denoise  8 tiles     ERROR: OSError: libmkl_intel_lp64.so.2: cannot  
8821368   denoise  8 tiles     ERROR: OSError: libmkl_intel_lp64.so.2: cannot  
8821369   denoise  8 tiles     ERROR: OSError: libmkl_intel_lp64.so.2: cannot  
8821376   denoise  8 tiles     ERROR: OSError: libmkl_intel_lp64.so.2: cannot  
8821377   denoise  8 tiles     ERROR: OSError: libmkl_intel_lp64.so.2: cannot  
8821378   denoise  8 tiles     ERROR: OSError: libmkl_intel_lp64.so.2: cannot  
8821380   denoise  8 tiles     ERROR: OSError: libmkl_intel_lp64.so.2: cannot  
8821381   denoise  8 tiles     ERROR: OSError: libmkl_intel_lp64.so.2: cannot  
8821382   denoise  8 tiles     ERROR: OSError: libmkl_intel_lp64.so.2: cannot  
8821403   denoise  8 tiles     ERROR: OSError: libmkl_intel_lp64.so.2: cannot  
8821404   denoise  8 tiles     ERROR: OSError: libmkl_intel_lp64.so.2: cannot  
8821405   denoise  8 tiles     ERROR: OSError: libmkl_intel_lp64.so.2: cannot  
8821406   denoise  8 tiles     ERROR: OSError: libmkl_intel_lp64.so.2: cannot  
8821407   denoise  8 tiles     ERROR: OSError: libmkl_intel_lp64.so.2: cannot  
8821408   denoise  8 tiles     ERROR: OSError: libmkl_intel_lp64.so.2: cannot  
8821428   denoise  ?           no training                
8821434   denoise  ?           no training                
8821438   denoise  ?           no training                
8821439   denoise  ?           no training                
8821469   denoise  ?           no training                
8821504   denoise  ?           no training                
8821931   denoise  ?           no training                
8822208   denoise  ?           no training                
8822209   denoise  ?           no training                
8838696   denoise  ?           no training                
8838813   denoise  ?           no training                
8838858   denoise  ?           no training                
8838985   denoise  ?           no training                
8839072   denoise  ?           no training                
8839122   denoise  ?           no training                
8839139   denoise  8 tiles     no training                
8839150   denoise  12 tiles    ERROR: RuntimeError: The device index is out of 
8839166   denoise  12 tiles    GPU FAULT at 19/60         
8839203   denoise  12 tiles    GPU FAULT at 2/103         
8839579   denoise  12 tiles    ERROR: ValueError: multiclass format is not sup 
8839683   denoise  12 tiles    COMPLETE 1/1               'test_auroc': 0.9354260932338447
8839842   denoise  12 tiles    COMPLETE 1/1               'test_auroc': 0.9354279541485844
8839881   denoise  12 tiles    COMPLETE 1/1               'test_auroc': 0.9401584363884974
8839890   denoise  12 tiles    COMPLETE 1/1               'test_auroc': 0.9355479921240085
8839920   denoise  ?           no training                
8839937   grid                 4/4 arms ok                
8839946   denoise  12 tiles    ERROR: RuntimeError: Expected to have finished  
8839957   grid                 4/4 arms ok                
8839997   grid     1 tile(s) each 4/4 arms ok                
8840007   denoise  12 tiles    GPU FAULT at 107/2018      
8840008   denoise  12 tiles    GPU FAULT at 26/2018       
8840010   denoise  12 tiles    GPU FAULT at 107/2018      
8840011   denoise  12 tiles    GPU FAULT at 29/2018       
8840057   denoise  12 tiles    no training                
8840058   denoise  12 tiles    no training                
8840095   denoise  12 tiles    GPU FAULT at 29/60         
8840120   denoise  12 tiles    GPU FAULT at 29/600        
8840154   denoise  12 tiles    GPU FAULT at 168/700       
8840190   denoise  1 tiles     COMPLETE 1/1               'test_auroc': 0.8628252912383914
8840223   align    12 tiles    GPU FAULT at 3/200         
8840232   grid     1 tile(s) each 72/72 arms ok              
8840238   align    1 tiles     GPU FAULT at 73/200        
8840257   align    1 tiles     ERROR: RuntimeError: expected scalar type BFloa 
8840264   denoise  12 tiles DS COMPLETE 1/1               'test_auroc': 0.7296013323843181
8840277   align    1 tiles     ERROR: KeyError: 'eval_loss' 
8840291   denoise  ?           refused: stale arms        
8840304   align    1 tiles     COMPLETE 400/400           'eval_loss': '0.04813'
8840313   denoise  ?           refused: stale arms        
8840323   denoise  ?           refused: stale arms        
8840336   align    12 tiles DS ERROR: RuntimeError: expected scalar type Float 
8840345   grid     12 tile(s) each 216/216 arms ok            'test_auroc': 0.5783115944609307
8840356   align    12 tiles DS GPU FAULT at 56/300        'eval_loss': '0.6521'
8840378   align    12 tiles    GPU FAULT at 1167/1167     
8840403   align    12 tiles    GPU FAULT at 0/300         
8840408   grid     12 tile(s) each 212/216 arms ok            'test_auroc': 0.8015630560943791
8840431   denoise  ?           no training                
8840444   denoise  ?           no training                
8840469   align    12 tiles    GPU FAULT at 120/300       'eval_loss': '0.05669'
8840487   denoise  ?           no training                
8840498   align    12 tiles    GPU FAULT at 22/2390       
8840516   align    12 tiles    GPU FAULT at 206/2390      
8840529   align    1 tiles     COMPLETE 28630/28630       'eval_loss': '0.05461'
8840537   align    12 tiles    GPU FAULT at 26/2390       
8840581   denoise  1 tiles     ERROR:                     
8840587   denoise  ?           no training                
8840597   denoise  1 tiles     ERROR: AttributeError: 'MSDeltaForContrastive'  
8840603   align    12 tiles    GPU FAULT at 802/2390      
8840611   denoise  1 tiles     ERROR:                     
8840621   denoise  1 tiles     ERROR:                     
8840632   denoise  1 tiles     ERROR:                     
8840651   denoise  1 tiles     COMPLETE 1/1               
8840656   denoise  1 tiles     COMPLETE 1/1               
8840665   denoise  ?           12/12 arms ok              
8840693   denoise  ?           0/12 arms ok               
8840715   denoise  ?           12/12 arms ok              
8840739   align    1 tiles     COMPLETE 28630/28630       'eval_loss': '0.2432'
8840792   denoise  ?           4/4 arms ok                
8840898   denoise  ?           no training                
8840907   denoise  ?           no training                
8840927   denoise  ?           no training                
8840941   denoise  ?           no training                
8841066   denoise  ?           3/3 arms ok                
8841165   denoise  ?           2/2 arms ok                
8841255   denoise  ?           ERROR: TypeError: MSDeltaModel.forward() missin 
8841270   denoise  ?           ran to 170/170             
8841285   denoise  ?           ran to 172/172             
8841310   denoise  ?           ran to 170/170             
8841345   denoise  ?           12/12 arms ok              'test_auroc': 0.9360134693643527
8841966   denoise  12 tiles    COMPLETE 1/1               'test_auroc': 0.8429291281565523
8841973   denoise  ?           ran to 266/266
```
