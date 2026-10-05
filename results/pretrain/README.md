# Masked-peak pretraining (K195)

| figure | what it shows |
|---|---|
| `k195a_losses.png` | K195a-P: today's loss vs proposal_loss (KL to intensity^0.5), 25m debug hours; arm A trains on today's loss, arm B on the proposal, vs Chris's 25m curve; dotted = each loss for a model that fits the other target perfectly |

## How it was made

- Command: `.venv/bin/python sweeps/k195_compare.py` (login: logs and JSONs only).
- Inputs: arm runs `$MSDELTA_RUNS/k195/k195-25m-{A-8903620,B-8903621}/` (trainer_state.json, logs); Chris's 25m curve
  `$MSDELTA_DERIVED/k195/chris25m_log.json`; the floors `$MSDELTA_DERIVED/k195/k195a_floors.json`. Writes the parsed
  arm logs to `$MSDELTA_DERIVED/k195/k195a_<arm>_log.json`.
- Built: 2026-10-04 02:33 UTC. Notes: notes/OBSERVATIONS.md "K195a-P".
