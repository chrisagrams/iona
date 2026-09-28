# pbs/dag -- a thin job DAG + scheduler for Aurora PBS (PLAN I2)

One DAG for every job: pipelines are declared as Python specs, a scheduler tick submits
ready nodes onto whichever queue gets them finished soonest within the per-user limits,
and every job reports its own end through a notification file and, on success, a
`_SUCCESS.json` manifest. Standard library only, so it runs on the login node in well
under a second per tick (`pbs/dagctl` uses the repo venv's Python 3.12; 3.8+ works).

**Safety defaults:** dry-run is ON (`config.json` must say `"dry_run": false` AND the tick
must get `--live`), force-deleting hung jobs is OFF, cancelling orphans is OFF, and a node
without an approved card ID that is also logged in `notes/DECISIONS.md` is never submitted.

## Why here, why Python specs

* `pbs/dag/`, not `msdelta/`: `import msdelta` pulls in transformers/torch (tens of
  seconds on a login node), and this is batch-system plumbing next to the PBS scripts it
  drives. It is copied into each job's code snapshot like the rest of `pbs/`, harmlessly.
* Specs are Python, not YAML: real pipelines are generated (K66-C = 4 scales x 5 scoring
  sets), some nodes need a login-node step before qsub (write a models file from the
  training job's run dirs), and PyYAML is not in the stock interpreter.

## Layout

| file | what |
|---|---|
| `spec.py` | `Node`, `Sweep`, `Calibration`, `Prepare`, `Pipeline`; loading all `pipelines/*.py` into one graph; `validate` |
| `config.py` | queue limits, safety switches, paths; `<home>/config.json` overrides |
| `contract.py` | notification / manifest schema, `verify_success`, scheduler alerts, mixed-commit check |
| `backend.py` | `PBSBackend` (qstat -u + `qstat -f -F json`, qsub, qdel) and `FakeBackend` (simulated queues, limits, outcomes) |
| `history.py` | runtime estimates: spec, calibration, notification history, per-arm times from `pbs/logs/*.OU` |
| `scheduler.py` | the tick: observe, block, ready, pack, submit, heartbeat; `TickLock` |
| `cli.py` / `../dagctl` | `validate`, `plan`, `tick`, `status`, `commits`, `simulate` |
| `pipelines/` | the LIVE registry (empty until the first approved pipeline is added) |
| `examples/k66c.py` | today's K66-C pipeline as a spec (kept out of `pipelines/`: it is already running by hand) |
| `../lib/job_finish.sh` | sourced by PBS scripts: EXIT trap -> notification (always) + manifest (success) |

State lives in `--home` (default `$DAG_HOME` or `<scratch>/dag`): `state.json`,
`events.log`, `heartbeat.json`, `tick.lock`, `config.json`, `history_cache.json`.

## Usage

```bash
pbs/dagctl validate
pbs/dagctl plan                        # dry run against real qstat (read-only)
pbs/dagctl plan --backend fake         # against the fake backend
pbs/dagctl status                      # nodes, next actions, budget, heartbeat age
pbs/dagctl tick                        # one round, DRY-RUN
pbs/dagctl tick --live                 # only acts if config.json has "dry_run": false
pbs/dagctl commits k66c/score_400m_test k66c/score_200m_test
pbs/dagctl --pipelines-dir pbs/dag/examples --home /tmp/x simulate --backend fake
```

A tick loop (cron or `while sleep 600`) is NOT installed; that is part of going live.

## The user's decisions (PLAN I2) and where they live

1. **Spec** -- `spec.Node`: id, `script` + `vars` or `sweep=Sweep(root, arms_file)`, `deps`,
   `nodes`, `runtime_min` (None = unknown), `outputs` (globs; `{job}`, `{job:<dep>}`,
   `{scratch}`, `{repo}` templating), `queues`, `approved` (card ID). The scheduler refuses
   any node without a card ID, and (default) any card not in the DECISIONS.md log.
2. **Success contract** -- `lib/job_finish.sh` writes `<scratch>/manifests/<job>/_SUCCESS.json`
   LAST, atomically, only when the exit status is 0, every `jf_output` exists, every
   `jf_check` passed and (sweeps) `JF_ARMS_OK == JF_ARMS_TOTAL`; it records job id, snapshot
   commit/branch/dirty from `$MSDELTA_CODE_DIR/SNAPSHOT.txt`, outputs, checks, arms.
   `contract.verify_success` additionally checks the job id, the arm count against the
   ARMS_FILE (N/N) and the spec's own output globs.
3. **Notifications** -- the same EXIT trap ALWAYS writes
   `<scratch>/notifications/<UTC>_<job>_<node>_<status>.json` (ok / failed / partial /
   walltime / killed) with runtime, outputs, summary. SIGTERM is trapped only to record
   it (exit 143 as before). A job gone from qstat with no notification for
   `vanish_grace_min` is failed. Scheduler alerts go to `notifications/scheduler/`.
4. **Runtime + queues** -- estimates: spec > calibration > notification history (median
   per kind) > per-arm times from old sweep logs packed onto slots > unknown. Unknown +
   `Calibration` -> a generated `<id>~calib` debug node (the same command with e.g.
   `MAX_STEPS`), which needs its own card ID. A node fits debug / debug-scaling if
   estimate <= 50 min and nodes <= 2, otherwise capacity (<= 16 nodes) with walltime =
   1.3 x estimate (rounded up to 5 / 15 min). Each tick packs ready nodes by longest
   remaining critical path, preferring a queue where the job can start now, then debug
   before capacity, within max running + max queued per queue AND the generic Q limit
   (counting every job of the user, DAG or not, held ones included). A qsub "would
   exceed" is treated as a full queue, not a failure.
5. **Partial failures** -- a sweep that ends partial / walltime / killed / hung, or failed
   with some arms done, is resubmitted with `RESUME_JOB=<first job>` (the job whose run
   dirs hold the arms), up to `max_resumes`. Scripts that skip finished work
   (eval_grouped_retrieval) can declare `rerun_on=("walltime",)`.
6. **Hung jobs (FT27)** -- R with no `resources_used` after `hung_minutes` (10), or no
   write to the node's `progress` globs for `stall_minutes`: flagged + alerted once.
   `qdel -W force` + resubmission only if `allow_force_delete` is on, capped by
   `max_retries` / `max_resumes`.
7. **Runaway protection** -- deterministic job names `dag-<pipeline>-<node>[-r<n>]`; a live
   job with that name is adopted instead of submitting a second one; `flock` so ticks
   never overlap; dry-run default; `budget_node_hours` per pipeline (walltime reserved
   while live, actual runtime after); `max_submissions_per_tick`.
8. **Orphans** -- children are submitted only after their parents succeeded, except an
   explicit `submit_with_parent=True` pair (qsub `-W depend=afterok:<parent>`). A failed
   node blocks all descendants and alerts; a live orphan is cancelled only if
   `allow_cancel_orphans` is on (otherwise an alert says to cancel it by hand).
9. **One DAG, no code pin** -- every node records its manifest's snapshot commit;
   `status`/`plan` warn when a node consumes results from different commits, and
   `dagctl commits <nodes>` flags any comparison across commits.
10. **Health** -- each tick writes `heartbeat.json`; `status` shows the DAG, per-node state
    and view, next actions, budget use, and warns when the heartbeat is older than
    `heartbeat_stale_min`.
11. **Light** -- stdlib; one `qstat -u` + one `qstat -f -F json` per tick; log parsing is
    cached. `plan`/`simulate` of the 25-node K66-C example take ~0.1 / ~0.4 s.

## Adding job_finish.sh to another PBS script

```bash
SCRATCH_ROOT=${SCRATCH_ROOT:-...}
source "$REPO_DIR/pbs/lib/job_finish.sh"      # early: early exits get reported too
...
jf_output "$OUT_DIR"                           # must exist at exit
jf_check "every model scored" my_check_fn      # recorded pass / fail
JF_ARMS_OK=$ok JF_ARMS_TOTAL=$n                # sweeps
```

It is a no-op when `DRY_RUN` is set or `PBS_JOBID` is unset, and never changes the exit
status. Wired today into `aurora-finetune-sweep.pbs` and `eval_grouped_retrieval.pbs`.

## Not done / needs a live trial

* Nothing has been submitted by this code. `PBSBackend.submit/delete` are exercised only
  through the fake; qsub output parsing (`<id>.aurora-pbs-...`), the "would exceed" text
  and `qstat -f -F json` fields are taken from real output but not from a real submission.
* The walltime-vs-qdel distinction relies on `DAG_WALLTIME_SEC` (passed by the scheduler);
  a job submitted by hand ends as `killed`, which the scheduler treats like walltime.
* Whether debug-queue Q jobs count toward the generic Q limit is unknown; the default
  counts them (conservative). Tune `generic_q_limit` / `queues` in `config.json`.
* No tick loop / cron is installed; the first live trial should run ticks by hand.
