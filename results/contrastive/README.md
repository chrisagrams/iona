# Contrastive spectrum embeddings (PLAN C2, C4, C7, C10; K163/K188)

Retrieval on the **ms-contrastive-100k test split** (~25k experimental-spectrum queries, ~9.9k peptides): experimental
MAP@R and Hit@1, 3 seeds per point (mean ± sd). Every panel draws binned cosine (0.1 and 1 Da) and the frozen encoders'
final layer as reference lines. Current figures are at the top; `superseded/` keeps older ones for the record (an
invalid metric, an old recipe or an earlier eval, so don't cite them).

**Paper recipe** (`sweeps/plot_contrastive_100k.py`):

| figure | what it shows |
|---|---|
| `c100k_scale.png` | **C2**: model size at checkpoint 220k (frozen C1 recipe, replicate-corpus training); dashed line = the same encoders frozen, best layer ("before" training) |
| `c100k_checkpoint.png` | **C4**: pretraining checkpoint, 50m and 100m |
| `c100k_c7.png` | **C7**: continuing the best models on ms-contrastive-100k; x = steps into the epoch |
| `c100k_zeroshot_layers.png` | **C10**: frozen encoders, retrieval at every block (left) and best block vs final layer (right) |
| `c100k_zeroshot_abtt.png` | frozen encoders with all-but-the-top (mean + top-D principal directions fitted on TRAIN, removed before cosine): raw vs ABTT per encoder, and best-block MAP@R vs D |
| `c100k_transfer.png` | small replicate-corpus eval vs this test, all 69 Stage-1 models (Spearman 0.78) |

**Since submission: consensus recipe** (K163/K188: ms-contrastive-100k + consensus spectra, lr 4e-4, P170 x K2, every
pretraining checkpoint; the same 25,137 test queries and reference lines; `sweeps/plot_contrastive_cons.py`):

| figure | what it shows |
|---|---|
| `c100k_cons_scale.png` | model size at checkpoint 220k (as `c100k_scale.png`) plus the 540k finals incl. 25m |
| `c100k_cons_checkpoint.png` | pretraining grad steps, one line per scale (as `c100k_checkpoint.png`); bottom row zoomed |
| `c100k_cons_filters.png` | experimental MAP@R and library-search Hit@1 with no / 20 ppm / iso-20 ppm precursor filter (no query fails 20 ppm on this split) |
| `c100k_old_vs_new.png` | new recipe (solid) vs the paper's replicate-corpus C2/C4 curves (dashed) and C7 (stars) |

**All evaluation sets vs baselines** (`sweeps/plot_contrastive_datasets.py`; validation, test, oodval, mouse 20k,
human 20k, yeast full 86k; baselines on the same queries: binned cosine 0.1 / 1 Da, GLEAMS, the paper's C7 as stars):

| figure | what it shows |
|---|---|
| `datasets_scale.png` | MAP@R and Hit@1 vs model size at 540k, one column per set, open search |
| `datasets_checkpoint.png` | MAP@R vs pretraining steps per set (K188), one line per scale |
| `datasets_filters.png` | mouse / human: no filter vs 20 ppm vs iso-20 ppm, all / F (fail 20 ppm) / Fbar queries, vs binned 0.1 Da |

`c1_recipe/` (`sweeps/plot_c1.py`, small replicate-corpus eval): `c1_width_temperature.png` batch width x temperature
x epochs grid; `c1_scale.png` 50m vs 400m in that grid.

`superseded/`: `contrastive_ladder.png` old recipe (t 0.07; `sweeps/plot_contrastive_ladder.py`);
`contrastive_scaling.png`, `contrastive_ablation.png`, `contrastive_breadth.png` separation-ratio era, a metric C0
showed doesn't predict retrieval (`sweeps/plot_contrastive.py`); `retrieval_vs_separation.png` the C0 evidence itself
(`sweeps/plot_retrieval.py $MSDELTA_EVAL/contrastive/retrieval_vs_separation_8848049.json`).

## How it was made

| figures | command | inputs | built |
|---|---|---|---|
| `c100k_*` (paper), `c1_recipe/` | `qsub -q debug -l select=1 -l walltime=00:20:00 pbs/make_figures.pbs` (walks run dirs: compute node) | `$MSDELTA_EVAL/contrastive/{grouped100k-test,zeroshot-layers,zeroshot-layers-abtt}/`, run dirs in `$MSDELTA_RUNS` | before 2026-09-27 (file dates are the reorg merge b4f1f7a6) |
| `c100k_cons_*`, `c100k_old_vs_new.png` | `.venv/bin/python sweeps/plot_contrastive_cons.py` (login: result JSONs only) | `$MSDELTA_EVAL/contrastive/{cons-test,cons-allck-test,grouped100k-test}/` | 2026-10-04 01:50 UTC |
| `datasets_*` | `.venv/bin/python sweeps/plot_contrastive_datasets.py` (login) | `$MSDELTA_EVAL/contrastive/{cons-<set>,cons-allck-<set>,<set dir>,gleams,filter-failure}/`, `$MSDELTA_DERIVED/summary/{C_benchmarks,C_transfer}.csv` | 2026-10-04 03:37 UTC |
| `superseded/` | the scripts named above | `$MSDELTA_EVAL/contrastive/`, run dirs | before 2026-09-27 |

The ABTT numbers behind `c100k_zeroshot_abtt.png` are written to `$MSDELTA_DERIVED/tables/zeroshot-layers-abtt/summary.csv`.
