# Project notes

Five files, one job each. If something doesn't fit exactly one of them, it goes nowhere
until it does.

| file | answers | contains | does NOT contain |
|---|---|---|---|
| `PLAN.md` | **What are we trying to find out?** | goals, the numbered questions (D1–D4 denoise, C0–C8 contrastive, A0–A3 alignment, R0–R2 reranking) with status, order of work, rules | results, job logs |
| `STATUS.md` | **Where does everything stand right now?** | the status diagram: live jobs (each tagged with its question id), what's done/running/blocked, decisions needed | reasoning, history — it is overwritten, not appended |
| `OBSERVATIONS.md` | **What have we learned?** | one entry per result: numbers, n, p-values, strength, what would overturn it, what is not yet measured. Append-only; corrections are new entries | plans, defects |
| `TODO.md` | **What is broken or dangerous?** | defects and operational hazards (FT-numbered), each with status | research questions — those are PLAN.md |
| `status-history/` | **What did STATUS look like before?** | every past version of the status diagram, one file per commit, dated | — |

Reading order for someone new: `PLAN.md` → `STATUS.md` → `OBSERVATIONS.md`.

Outside this directory: `results/raw/finetune/README.md` indexes the result tables and
figures, and each sweep generator's docstring records why that grid exists.

When the status diagram changes, copy the old block into `status-history/` before
overwriting it.
