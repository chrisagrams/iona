# Test suite

What this exists for: to say, in minutes rather than a capacity job's forty, whether a
change has broken something that would otherwise be discovered forty minutes into one.

Every test here traces to a capability we rely on, and the ones marked **(regression)**
trace to a specific failure that has already cost real compute. Those are the ones not to
delete when they become inconvenient.

## Running

```bash
PYTHONPATH=. .venv/bin/python -m pytest tests -q            # CPU suite (see timings below)
PYTHONPATH=. .venv/bin/python -m pytest tests -q -m ""      # include slow checkpoint tests
PYTHONPATH=. .venv/bin/python -m pytest tests -q --legacy   # + opt-in legacy-approach tests
qsub -q debug -l select=1 -l walltime=00:30:00 pbs/run_tests.pbs   # CPU + device suites
qsub -q debug -l select=1 -l walltime=01:00:00 -A UIC-HPC -l filesystems=home:flare \
     -v REPO_DIR=$PWD pbs/run_e2e.pbs                       # opt-in e2e + golden
```

**It is not a seconds-long suite.** Measured 2026-09-27: the default CPU run is 647 tests
(641 pass, 6 known environment failures -- 4 level-zero PBS checks, 2 paths absent in a fresh worktree -- and 29 skipped) and took **10 min 20 s** on a loaded
login node (an earlier run on the same tree: ~6 min) and 4 min 49 s on a debug compute
node through `pbs/run_tests.pbs` (job 8873488, where the device suite then took 19 s); `tests/test_imports.py` alone is
~40 s because it starts child interpreters. On a login node run only the files you
touched; run the whole suite through `pbs/run_tests.pbs`.

### Opt-in markers

Three markers are **deselected by default** (they neither run nor count as skipped) and
selected by a flag or by any `-m` expression that names them (`tests/conftest.py`):

| marker | flag | what | where it runs |
| --- | --- | --- | --- |
| `legacy` | `--legacy` or `-m legacy` | tests of approaches no longer in the recipe (table below) | login, CPU, ~30 s for the 48 |
| `e2e` | `--e2e` | `tests/e2e/`: every entry point as a subprocess on tiny synthetic data | debug node, `pbs/run_e2e.pbs` |
| `golden` | `--golden` | `tests/golden/`: frozen checkpoints on frozen inputs vs stored references | debug node, `pbs/run_e2e.pbs`; skips without /flare |

The split is not arbitrary. A login node has no XPU, so anything about tile binding,
collectives, bf16 kernels or ZeRO-2 sharding is **unprovable** there, and a suite that
pretends otherwise is worse than one that admits the gap. `tests/gpu/` holds exactly the
cases that need a device, and `pbs/run_tests.pbs` runs them.

## Capabilities and where each is covered

### Primitives — `tests/test_primitives.py` (login)
| capability | why it matters |
| --- | --- |
| `FourierFeatures` width, finiteness, determinism | every m/z and modification mass goes through it |
| `DeltaMZBias` shape and antisymmetry | the dominant memory term; a wrong shape is an OOM at scale |
| `pool_sequence` ignores padding | padded peaks must not enter a mean or a max |
| `pooled_width` matches reality | the student's output width is derived from it |
| `parse_peptide` residues, mods, unknown characters | an unmapped residue is an unchecked GPU read **(regression)** |
| `PeptideCollator` padding and truncation | a peptide longer than the position table is the same read **(regression)** |
| `AlignmentCollator` spectra padding and `target` passthrough | |
| probes fire, and are off by default | a probe that is always on costs throughput |

### Models — `tests/test_models.py` (login, tiny configs)
| capability | why it matters |
| --- | --- |
| `MSDeltaModel` / `ForPreTraining` / `ForDenoising` forward shapes | one logit per peak, not two |
| `PeptideEncoder` under eval + autocast, fp32 AND bf16 weights | the fused-kernel dtype bug, both directions **(regression)** |
| teacher frozen *and* in eval mode | dropout would move the target every epoch **(regression)** |
| precomputed target equals the live one | the whole precompute path rests on this **(regression)** |
| teacher detaches cleanly (`spectrum_model=None`) | what removes it from the DDP/ZeRO graph |

### Metrics — `tests/test_metrics.py` (login)
| capability | why it matters |
| --- | --- |
| `denoise_metrics` ignores -100 and survives stray labels | a metrics crash killed a 4 h job **(regression)** |
| `denoise_metrics` on perfect and inverted predictions | AUROC 1.0 and 0.0, so orientation is pinned |
| `cross_modal_metrics` deduplicates candidates | |
| aligned embeddings rank perfectly | |

### Training mechanics — `tests/test_training.py` (login, CPU)
| capability | why it matters |
| --- | --- |
| the student's L2 actually falls | |
| `encoder_lr_scale` puts the right rate on each param group | |
| the scheduler writes only `param_group["lr"]` | DDP-safe; the alternative desynced ranks **(regression)** |
| Trainer can produce `eval_loss` | `metric_for_best_model` raised without it **(regression)** |
| `subset_splits` caps rows and leaves the rest alone | |

### Configuration and orchestration — `tests/test_configs.py` (login)
| capability | why it matters |
| --- | --- |
| every args file parses into its dataclasses | |
| no comments in args files | `HfArgumentParser` splits on whitespace **(regression)** |
| no multi-word values in args files | same cause, different symptom |
| every config directory has a `DESCRIPTION.md` | |
| W&B destination is `CS_Pharm/msdelta-finetune` | |
| run names carry the `v2_` prefix | old runs are not comparable |
| `peak_pair_budget` >= `max_peaks`² | below it, batching cannot make progress **(regression)** |
| grid arms match their template and recorded stage | a stale grid would have OOMed 72 arms **(regression)** |
| every PBS script is valid bash | |
| PBS scripts reference only paths that exist | |
| sweep partitions every arm exactly once | a dropped arm is a silent hole in the grid |
| sweep refuses multi-tile arms without DeepSpeed | that is the DDP fault, FT7 **(regression)** |

### Needs a device — `tests/gpu/` (debug queue)
| capability | why it cannot run on a login node |
| --- | --- |
| tile count and `select_device` under both `ZE_AFFINITY_MASK` conventions | no XPU |
| a real checkpoint forward on device | |
| eval under bf16 autocast on XPU | CPU does not take the same fused path — this is how FT-dtype escaped **(regression)** |
| bf16 weights, as ZeRO-2 supplies them | **(regression)** |
| a checkpoint save/load round trip | |
| the distributed gather emits no stray labels | FT4 **(regression)** |

## Opt-in (legacy approaches)

The default run covers the recipe as it stands: contrastive = SupCon + KL anchor +
same-mass batches, mean+max pooling, no projection head; alignment = MSE student onto
frozen teacher embeddings, mean+max readout. Tests of rejected or superseded approaches
are kept, marked `legacy`, and run only on request -- the library code they test still
exists and a parked retry (layer mix) may come back:

```bash
PYTHONPATH=. .venv/bin/python -m pytest tests -q --legacy      # everything, legacy included
PYTHONPATH=. .venv/bin/python -m pytest tests -q -m legacy     # only the legacy tests
```

48 tests; all pass (2026-09-27). Tests that were moved into the default suite when these
were dropped (7ed4e19) stay where they were moved and are not duplicated here.

| opt-in (`legacy`) | decision | what it tests |
| --- | --- | --- |
| `tests/test_projection_head.py` (whole file) | C9 (no projection head) | `--projection_dim` head: width, readout switch, head gradients, GradCache exact through the head |
| `test_contrastive.py::TestSigmoidLoss` (`test_rejects_an_unknown_loss` is in the default `TestContrastiveModel`) | C8 (keep SupCon) | `sigmoid_contrastive_loss` hand value, learnable scale/bias, GradCache exactness with the sigmoid loss |
| `test_contrastive.py::TestPairSamplerAndLoss` | FT17 (pair loss superseded) | `PairBatchSampler`, `pair_contrastive_loss` |
| `test_contrastive.py::TestLayerMixPooler` | C9 design decision / FT11 (layer mix dropped; PLAN lists it as a *parked retry*) | `LayerMixPooler`, `encoder_layer_states`, `pooling=layer_mix` |
| `tests/test_align_contrastive.py` (whole file; the plain-MSE default is checked by the default `test_models.py::TestAlignmentModel::test_default_loss_is_plain_mse`) | A4 / A6 (LiT student, hard negatives) | `hard_negatives`, `lit_contrastive_loss`, `AlignmentCollator(hard_negatives=...)`, `loss="lit"` gradient regression |
| `tests/test_mass_aware.py` (whole file; `test_peptide_neutral_mass` is in the default `test_contrastive.py::TestSameMassBatches`, since it feeds C19's `group_masses`) | A8 (mass-aware student) | `MassNegativePool`, `MassBatchSampler`, `AlignmentCollator(neg_source="mass")` |
| `test_student_readout.py::test_shapes_and_unit_norm`, `::test_padding_does_not_leak` | A3 (cls/attn readouts rejected, keep mean+max) | cls/attn forward shapes and padding. `student_readout()` detection is in the default run: the loaders use it on legacy A3 dirs |

## End-to-end — `tests/e2e/` (opt-in `--e2e`, debug node)

Each entry point runs as `python -m msdelta.<path>` in a subprocess, exactly as a PBS
script launches it, on synthetic data generated deterministically by
`tests/e2e/synth.py` (b/y ions of real peptides, jittered per replicate, plus noise
peaks; ~300 KB, nothing stored in the repo). Entry points that load a Hub dataset by
name take the synthetic directory of `<split>.parquet` files as the same argument
(`datasets.load_dataset(<dir>)`), so no code change was needed. The model is 2 layers,
hidden 32; `pbs/run_e2e.pbs` runs them on one tile. Measured on job 8873354 (one debug
node, XPU): 6 passed in 4 min 20 s.

| test | drives | checks | time |
| --- | --- | --- | --- |
| `test_pretraining` | `pretraining.train`, 20 steps, 64 spectra, `--preprocessed_dataset_dir` | loss finite and falling, eval_loss, `final/` == `checkpoint-20`, processor saved | 91 s |
| `test_contrastive_and_resume` | `finetuning.contrastive.finetune_contrastive`, current recipe (SupCon, same-mass batches, GradCache chunk 4 + trim, KL 10, no grad checkpointing), 30 analytes x 3, 10 steps; then `--resume_from_checkpoint checkpoint-5` | loadable `final/` and `checkpoint-5/encoder`, step sequence 1..10 once, resumed run did not restart | 36 s |
| `test_denoise_and_resume` | `finetuning.denoise.finetune_denoise`, 10 steps + eval + test split; resume | eval/test AUROC, F1, AUPRC finite; resume as above | 30 s |
| `test_alignment_precompute_sharded_and_train` | `precompute_align` prepare / 2 shards / merge, and unsharded; `finetune_align` 10 steps | sharded targets == unsharded, `final/peptide_encoder` loads with `PeptideEncoderModel` (model_type `msdelta-peptide-encoder`) and embeds | 57 s |
| `test_grouped_retrieval_eval` | `eval.eval_grouped_retrieval prepare` + `score` (pretrained, contrastive, binned) | row count, MAP@R in [0, 1] for `all` and `experimental` | 17 s |
| `test_psm_rerank_cli` | `rescoring.psm_rerank score` / `train` / `score --mode global`, `rescoring.rerank_psm_fdr` on `test_rerank_r4`'s synthetic runs | one PSM per spectrum, q in [0, 1], the strong synthetic signal survives | 28 s |

## Golden outputs — `tests/golden/` (opt-in `--golden`, debug node)

Frozen checkpoints on frozen inputs, compared against what the code computed when the
references were written. Catches the silent class of regression the other tests cannot:
the same weights and the same spectra now producing different numbers.

- **Frozen:** the 25M pretrained checkpoint (step 540,423), a 50M contrastive `final/`
  (sweep-cont050m_ep01_seed1), and the peptide encoder Hub release `Gaolaboratory/iona-peptide-embedder-400m`
  from the local HF cache; rows 0-199 of the prepared ms-contrastive-100k validation split
  and the first 50 distinct (peptide, charge) pairs in them. All read-only.
- **Compared:** pooled mean+max spectrum embeddings (both models), the pretraining head's
  per-peak log-probabilities (32 spectra), grouped-retrieval metrics on the 200 rows
  (`all` / `experimental`, the eval's own `_variants`), peptide embeddings.
- **References:** `tests/golden/reference/golden.npz` (1.1 MB; unit-norm embeddings as
  float16, with the float16 rounding added to the tolerance) and `MANIFEST.json` (paths,
  sha256 of every weight file, row indices, peptides, code commit, torch version,
  tolerances, the measured bf16 deviation). Generated on CPU in fp32 by debug job 8873406
  at commit 1b8c0b4.
- **Tolerances:** CPU fp32 recompute: 1e-4 absolute. XPU bf16 autocast against the fp32
  reference: embeddings 1e-2, head log-probs 0.25, metrics 0.05 (measured max deviation
  5.4e-3, 0.087 and 7.2e-3).
- Checks that the checkpoints' sha256 still match first: a replaced checkpoint makes
  every other failure meaningless. Skips with the list of missing paths where /flare or
  the HF cache is not readable.

**Regenerating is a deliberate act.** A failing golden test means the code now computes
something different. If that is a bug, fix the code. Only when the change is intended and
understood, regenerate (`qsub ... -v REPO_DIR=$PWD,SUITE=regenerate-golden
pbs/run_e2e.pbs`, see `tests/golden/regenerate.py`) and commit the new references with the
change that moved them, saying why in the message.

## Imports and metric fixtures

- `tests/test_imports.py` — every flat shim `msdelta/<old>.py` is the same module object as
  its new home, `from msdelta.<old> import X` works, `python -m msdelta.<old> --help`
  reaches the new main for the entry points PBS uses, the `msdelta.data` /
  `msdelta.rescoring` PEP 562 fall-through, and every module imports with faiss blocked
  (`tests/_nofaiss/`). One child interpreter does the expensive part (~40 s).
- `tests/test_metric_fixtures.py` — MAP@R, R-Precision, Hit@1, MAP@100, R@5 on hand-worked
  cases (perfect, worst, R=1/R=2 mix with a singleton, MAP@R != R-Precision), topk vs
  exact, the grouped eval's `all` / `experimental` variants, `denoise_metrics`
  (F1, AUROC, per-spectrum AUROC), `cross_modal_metrics` (peptide -> spectrum Hit@1, MRR).
  Each expected value is derived in the test's docstring.

## Deliberate gaps

- **Multi-tile behaviour is not covered here.** FT7 and FT9 need 12 ranks; `pbs/run_tests.pbs`
  requests one node and the collective paths are exercised by the smoke sweeps instead.
- **No test asserts a model is *good*.** These check that things run and that arithmetic is
  right. Whether AUROC 0.86 is worth anything is a question for the sweep, not the suite.
