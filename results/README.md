# Results

Only what people read: one folder per report, each with Markdown and plots and a README.md that says how the report
was made (command, inputs, when). Machine-readable data is not here (notes/AGENT_PLAYBOOK_2.md C7); it lives on /flare
in the homes defined once in `configs/homes.env` (shell: `source pbs/lib/homes.sh`; Python: `sweeps/homes.py`):

| home | what | path |
|---|---|---|
| primary | training runs, `<run>-<job>/` | `$MSDELTA_RUNS` = `/lus/flare/projects/UIC-HPC/khuss/msdelta/runs` |
| primary | evaluation outputs, `<track>/<eval run>/` (per-model JSONs, logs) | `$MSDELTA_EVAL` = `.../msdelta/eval` |
| primary | diagnostic job outputs, `<name>/` | `$MSDELTA_DIAG` = `.../msdelta/diag` |
| preprocessed | datasets as the code consumes them | `.../msdelta/{data,eval-data,baselines/*/prepared}` |
| derived | tables computed from the above (CSV behind each figure, grid tables); rebuildable, never hand-edited | `$MSDELTA_DERIVED` = `.../msdelta/_derived` |
| results | Markdown + plots | this folder |

| report | what |
|---|---|
| `summary/` | the submitted D / C / A / R figures and summary tables (2026-09-25) |
| `contrastive/` | spectrum-embedding retrieval: paper recipe (C2/C4/C7/C10), consensus recipe (K163/K188), all evaluation sets vs baselines |
| `denoise/` | denoise checkpoint ladder (D1-D3) |
| `pretrain/` | masked-peak pretraining losses (K195) |
| `k163_cons/` | K163-C consensus twins vs no-consensus partners, per set |
| `k188_allck_cons/` | K188-C consensus fine-tunes on every pretraining checkpoint, per set |
| `k197_binned_edge/` | K197-C where binned cosine beats our encoders |

To change a number: rerun the evaluation (it writes to `$MSDELTA_EVAL` / `$MSDELTA_DIAG`), then the report's command
from its README. Until K198b is decided, `raw/` and `processed/` still hold the originals of the copies on /flare
(K198a, verified by md5: `$MSDELTA_STORAGE/k198-logs/`); nothing reads them any more.
