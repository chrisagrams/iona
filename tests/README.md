# Test suite

What this exists for: to say, in under a minute on a login node, whether a change has
broken something that would otherwise be discovered forty minutes into a capacity job.

Every test here traces to a capability we rely on, and the ones marked **(regression)**
trace to a specific failure that has already cost real compute. Those are the ones not to
delete when they become inconvenient.

## Running

```bash
PYTHONPATH=. .venv/bin/python -m pytest tests -q            # login: CPU only, seconds
PYTHONPATH=. .venv/bin/python -m pytest tests -q -m ""      # include slow checkpoint tests
qsub -q debug -l select=1 -l walltime=00:30:00 pbs/run_tests.pbs   # the device half
```

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

## Removed (approach no longer used)

The suite covers the recipe as it stands: contrastive = SupCon + KL anchor + same-mass
batches, mean+max pooling, no projection head; alignment = MSE student onto frozen
teacher embeddings, mean+max readout. Tests of rejected or superseded approaches were
removed; the LIBRARY CODE they tested was not. To restore any of them:
`git show fa90d44:tests/<file>` (the last commit that has them all).

| removed | decision | what it tested |
| --- | --- | --- |
| `tests/test_projection_head.py` (whole file) | C9 (no projection head) | `--projection_dim` head: width, readout switch, head gradients, GradCache exact through the head |
| `test_contrastive.py::TestSigmoidLoss` (except `test_rejects_an_unknown_loss`, moved to `TestContrastiveModel`) | C8 (keep SupCon) | `sigmoid_contrastive_loss` hand value, learnable scale/bias, GradCache exactness with the sigmoid loss |
| `test_contrastive.py::TestPairSamplerAndLoss` | FT17 (pair loss superseded) | `PairBatchSampler`, `pair_contrastive_loss` |
| `test_contrastive.py::TestLayerMixPooler` | C9 design decision / FT11 (layer mix dropped; PLAN lists it as a *parked retry*, restore with it) | `LayerMixPooler`, `encoder_layer_states`, `pooling=layer_mix` |
| `tests/test_align_contrastive.py` (whole file; `test_default_model_loss_is_mse` moved to `test_models.py::TestAlignmentModel::test_default_loss_is_plain_mse`) | A4 / A6 (LiT student, hard negatives) | `hard_negatives`, `lit_contrastive_loss`, `AlignmentCollator(hard_negatives=...)`, `loss="lit"` gradient regression |
| `tests/test_mass_aware.py` (whole file; `test_peptide_neutral_mass` moved to `test_contrastive.py::TestSameMassBatches`, since it feeds C19's `group_masses`) | A8 (mass-aware student) | `MassNegativePool`, `MassBatchSampler`, `AlignmentCollator(neg_source="mass")` |
| `test_student_readout.py::test_shapes_and_unit_norm`, `::test_padding_does_not_leak` | A3 (cls/attn readouts rejected, keep mean+max) | cls/attn forward shapes and padding. `student_readout()` detection stays: the loaders use it on legacy A3 dirs |

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
