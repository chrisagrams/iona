# Primer (read this first, 3 minutes)

**Project.** msdelta (a.k.a. Iona) is a pretrained transformer over MS/MS spectra (sizes 25m/50m/100m/200m/400m,
pretraining checkpoints 10k → 540k steps; pretraining is Chris's). We fine-tune it for downstream tasks and ask
two things: **does it scale** (size, pretraining amount) and **how much does pretraining buy**.

**Tasks** (track letter = ID suffix in chat, e.g. K136-C):
| track | task | metric | state |
|---|---|---|---|
| D | denoise: per-peak noise classification | AUROC/F1 | done (D1-D4) |
| C | contrastive: spectrum encoder for retrieval | MAP@R (experimental), library Hit@1, ± precursor filter | **active** |
| A | alignment: peptide encoder matched to the spectrum encoder | Hit@1 spectrum→peptide | parked until new C models |
| R | rescoring PSMs with the encoders | PSMs at 1% FDR | parked |
| P | Pairformer architecture (pair representation) vs transformer | loss, speed, downstream | Stage 0 done |
| I / S | infrastructure / cross-cutting | — | ongoing |

**Branches.** `master` = Chris's, never touched. `dev_finetune` = frozen at paper submission (2026-09-26).
`dev_finetune_02` = everything since; main checkout `~/code/msdelta`. Side work goes on a branch/worktree,
is merged via a scratch worktree after tests pass, then `pbs/checkout_ff`.

**Code map** (`msdelta/`):
- `models/` — encoder (`modeling_msdelta.py`), `pairformer.py`, peptide encoder, strict `loading.py`
- `pretraining/` — master's `train.py` (don't modify)
- `finetuning/{contrastive,denoise,alignment}/` — the fine-tuning trainers (contrastive: SupCon, GroupBatchSampler)
- `eval/` — grouped retrieval (MAP@R), `filtered_retrieval.py`, `library_search.py`
- `data/` — grouped datasets (ms-contrastive-100k: 3 experimental + 1 consensus spectrum per peptide group)
- `rescoring/` — PSM reranking

**How experiments run.**
- `sweeps/make_*.py` writes a grid to `configs/sweep-*/<arm>/training.args` (docstring = why the grid exists).
- `pbs/aurora-finetune-sweep.pbs` runs one arm per GPU tile (12 per node), any number of nodes;
  `pbs/eval_grouped_retrieval.pbs` scores finals. Every job snapshots its code (`pbs/lib/code_snapshot.sh`)
  and loads the env via `pbs/lib/load_frameworks.sh` (required since the 2026-09 Aurora update).
- `pbs/tools/feeder.sh <plan>` submits approved jobs as queue slots free (plans in `pbs/tools/feeder_plans/`).
- Outputs: runs on `/lus/flare/projects/UIC-HPC/khuss/msdelta/runs/`, results in `results/raw/…`.

**Notes** (`notes/`) — where to look:
| question | file |
|---|---|
| what's happening right now / jobs | `STATUS.md` |
| the story so far, where we're going | `NARRATIVE.md` |
| what needs my decision | `OPEN_QUESTIONS.md` (top: my to-do) |
| what I decided | `DECISIONS.md` |
| results | `OBSERVATIONS.md` |
| all research questions + status | `PLAN.md` |

**Rules I set.** Nothing is submitted without my approved card; every decision logged; every question Claude
raises goes to OPEN_QUESTIONS with a K-ID; report retrieval with/without precursor filter; login node = light
work only; never touch master; deletions follow the protocol.

**Check status yourself:** `qstat -u khuss` · `tail $S/logs/feeder_*.log` · `git log --oneline -10` ·
`sbank-list-allocations -p UIC-HPC` (node-hours left).
