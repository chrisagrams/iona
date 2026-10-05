# Summary figures and tables (D / C / A / R, paper submission, 2026-09-25)

The figures and tables of the submitted results, one prefix per track: D denoise, C contrastive spectrum embeddings,
A peptide embeddings (alignment), R PSM reranking. `tables.md` holds the summary tables. The CSV behind each figure
(exactly the plotted rows) is in `$MSDELTA_DERIVED/summary/<figure>.csv`; longer tables in `$MSDELTA_DERIVED/tables/`.

| figures | command (login: result JSONs and recorded numbers only) | inputs |
|---|---|---|
| `D_denoise.png`, `A_peptide.png`, `R_reranking.png`, `tables.md` | `.venv/bin/python sweeps/plot_summary.py` | numbers recorded in notes/OBSERVATIONS.md (job ids in the script's comments); denoise cells from run dirs via `sweeps/plot_ladder.py` -> `$MSDELTA_DERIVED/tables/denoise_scaling_pretraining.csv` |
| `C_benchmarks.png`, `C_pretraining_scaling.png`, `C_pretraining_ablation.png`, `C_zeroshot.png`, `C_transfer.png`, `C_transfer_ours.png` | `.venv/bin/python sweeps/package_contrastive.py` | `$MSDELTA_EVAL/contrastive/<benchmark>/`, GLEAMS metrics in `$MSDELTA_STORAGE/baselines/<benchmark>/`, `$MSDELTA_DERIVED/tables/zeroshot-layers-abtt/summary.csv`; also refreshes `paper/experiments/spectrum_embedding/0_shot/` |
| `A_windows.png`, `A_windows_A2_preview.png`, `A_teacher_student.png` + `.md`, `A_ablations.png` | `.venv/bin/python sweeps/package_alignment.py` | `$MSDELTA_EVAL/align/` (student evals, `mouse_yhydra/`), `$MSDELTA_STORAGE/baselines/<dataset>/xmodal/` |
| `R_embedding_gain.png`, `R_benchmark.png` | `.venv/bin/python sweeps/package_rerank.py` | `$MSDELTA_EVAL/rerank/psm/*.json`, `baselines_wip/results_ms2rescore*.json` |
| `D_hp_parallel.png` | `.venv/bin/python sweeps/plot_hp_parallel.py` | `$MSDELTA_DERIVED/denoise/grid_denoise_50m.txt` (job 8840408, 216 arms) |
| `D_denoise_loss_scaling.png` | `.venv/bin/python sweeps/plot_denoise_loss_scaling.py` | run dirs via `sweeps/plot_ladder.py`; drawn in `paper/experiments/denoise/loss_scaling/` and copied here |
| `C_transfer_with_frozen_PREVIEW.png` | none in the tree (an ad hoc preview, kept for the record) | -- |

Built: before 2026-09-27 (file dates are the reorg merge b4f1f7a6); the numbers are the submitted ones
(notes/OBSERVATIONS.md). Newer contrastive results are in `results/contrastive/` and the `k*` report folders.
