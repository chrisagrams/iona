# Denoise: per-peak noise classification (PLAN D1-D3)

Test split of `ms-denoise-100k`, 3 seeds per point, mean ± sd. The x axis is pretraining grad steps, not compute.
Current figures are at the top; `superseded/` keeps older ones for the record (don't cite them).

| figure | what it shows |
|---|---|
| `ladder_compute.png` | **Headline.** AUROC, F1, AUPRC and test loss vs pretraining steps, one line per model size |
| `ladder_by_scale_{auroc,f1,auprc,loss}.png` | one panel per model size: metric vs pretraining checkpoint |
| `ladder_by_checkpoint_{auroc,f1,auprc,loss}.png` | one panel per checkpoint: metric vs model size |

`superseded/` (`sweeps/plot_denoise.py`): `scaling.png`, `pretrain_ablation.png` first scale / scratch plots, made
before the checkpoint ladder existed; `encoder_lr_scale.png`, `probe_heatmap.png`, `top_cluster.png` HP-search
diagnostics.

## How it was made

| figures | command | inputs | built |
|---|---|---|---|
| `ladder_*` | `qsub -q debug -l select=1 -l walltime=00:20:00 pbs/make_figures.pbs` (`sweeps/plot_ladder.py`; walks run dirs: compute node) | denoise sweep run dirs in `$MSDELTA_RUNS` (`all_results.json`), arm configs `configs/sweep-*` | before 2026-09-27 (file dates are the reorg merge b4f1f7a6) |
| `superseded/` | `.venv/bin/python sweeps/plot_denoise.py --runs $MSDELTA_RUNS` | run dirs; `$MSDELTA_DERIVED/denoise/denoise_scale_seeds.txt` | before 2026-09-27 |

The grid tables (`sweeps/summarise_denoise.py`) are in `$MSDELTA_DERIVED/denoise/`. They are rebuilt from whatever run
dirs the script finds, so check that every run a table lists is still on /flare before rebuilding over it.
