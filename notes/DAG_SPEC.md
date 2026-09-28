# Job DAG + scheduler: specification (I2)

Status: **built, never run live** (branch `i2-dag`: `pbs/dag/`, `pbs/dagctl`, `pbs/lib/job_finish.sh`;
50 tests on a fake PBS backend). Awaiting the user's review (K63-I) before merge or any live trial.
Implementation notes: `pbs/dag/README.md` on that branch. Decisions: PLAN.md row I2, DECISIONS.md.

This sheet describes *what the system must do* independently of how `pbs/dag` does it, so it
can later be generalised into a standalone feature (other projects, other PBS/Slurm sites).

## 1. Purpose

Run every experiment as nodes of one dependency graph. A node is queued automatically when its
prerequisites have succeeded, is placed on whichever queue finishes the graph soonest within the
site's per-user limits, reports its own outcome, and publishes outputs its children can consume.
Humans approve every experiment; the scheduler never invents work.

## 2. Concepts

| term | meaning |
|---|---|
| **pipeline** | a named set of nodes (an experiment and its evaluation), declared in a spec file |
| **node** | one batch job: a command (script + variables, or a sweep = config grid + arm list), dependencies, resources (nodes), expected runtime (or unknown), declared outputs, allowed queues, approval (card ID) |
| **arm** | one training run inside a sweep node; a sweep succeeds only when all its arms do |
| **card** | the user-approved experiment description; its ID must appear in `notes/DECISIONS.md` |
| **tick** | one scheduling round: observe -> decide -> submit -> record; seconds long, idempotent |
| **manifest** | `_SUCCESS.json`, written last by a job that succeeded; the only proof of success |
| **notification** | a file every job writes when it ends, whatever the outcome |

## 3. Requirements (user decisions, 2026-09-27)

R1 **Approval gate** -- a node without an approved card ID (logged in DECISIONS.md) is never submitted.
R2 **Dependencies** -- a node is ready only when every parent has a valid success manifest.
R3 **Success contract** -- success = exit 0 AND every declared output exists AND every declared check passed AND (sweeps) N/N arms ok; recorded in a manifest written LAST and atomically (tmp + rename). PBS exit codes alone are never trusted.
R4 **Notifications** -- every job writes `<notifications>/<UTC>_<job>_<node>_<status>.json` at exit (ok / failed / partial / walltime / killed) with runtime, outputs, summary. A job that vanished without one is treated as failed.
R5 **Expected runtime** -- from the spec, else from history (past runs of the same kind), else a **calibration node** (short debug run, extrapolated) that itself needs approval.
R6 **Queue choice + multi-queue allocation** -- each node lists the queues it fits; each tick fills free slots across all queues, highest priority = longest remaining downstream path; short jobs use debug slots instead of waiting for capacity.
R7 **Pacing within limits** -- never exceed any per-user limit (running per queue, queued per queue, the site's generic queued limit which counts held jobs).
R8 **Partial failures** -- sweeps with missing arms resume (`RESUME_JOB`), rerunning only what is missing; capped retries.
R9 **Hung jobs** -- detect jobs that run without progress (e.g. no resources used, logs not growing); alert; force-delete only if explicitly enabled.
R10 **Runaway protection** -- deterministic job names (adopt, never duplicate), a lock so ticks never overlap, dry-run by default, per-pipeline node-hour budget, max submissions per tick.
R11 **Orphans** -- children are submitted only after parents succeed (no speculative `afterok` chains except explicit smoke->run pairs); a failed node blocks its descendants and alerts.
R12 **Provenance** -- each node records the code snapshot commit it ran; comparisons across different commits are flagged. One DAG for all jobs, so no per-pipeline code pin.
R13 **Health** -- each tick writes a heartbeat; a stale heartbeat is reported; `status` shows every node, the next actions and budget use.
R14 **Light** -- runs on a login node in seconds; no heavy validation.
R15 **Secrets / site rules** -- never pass secrets through the batch system's variables; smoke on debug before capacity; respect login-node discipline.

## 4. Contracts (interfaces other code must honour)

**Job side** (any batch script): source one helper early; declare outputs/checks; for sweeps set arms ok/total. The helper's EXIT trap writes the notification (always) and the manifest (success only), never changes the exit status, and is a no-op outside a batch job.

**Manifest** `_SUCCESS.json`: job id, node id, pipeline, snapshot commit/branch/dirty, start/end time, outputs (paths), checks (name -> pass), arms (ok/total).

**Notification**: job id, node id, status, exit code, runtime, outputs, one-line summary.

**Spec** (per node): id, command, deps, nodes, runtime_min | None, outputs (templated globs), queues, approved card ID, optional: calibration recipe, progress globs (for hang detection), rerun_on (statuses that allow a plain rerun), submit_with_parent.

**State** (scheduler home): per-node state machine
`blocked-on-approval -> waiting -> ready -> submitted -> running -> {succeeded | failed | partial | walltime | hung | vanished} -> (resume/retry -> submitted) | blocked-descendants`, plus an append-only event log, heartbeat, lock, config.

**CLI**: `validate` (specs), `plan` (what would be submitted where and why; read-only), `tick` (one round; dry-run unless explicitly live), `status`, `commits` (cross-commit check), `simulate` (fake backend).

## 5. Policies (defaults, all configurable)

| policy | default |
|---|---|
| debug eligibility | estimate <= 50 min and <= 2 nodes |
| capacity walltime | 1.3 x estimate, rounded up |
| priority | longest remaining critical path; ties: start-now queue first, debug before capacity |
| retries / resumes | 2 |
| hung | no resources used after 10 min, or progress files unchanged for a set time: alert |
| vanished | gone from the queue with no notification for 5 min: failed |
| dry run / force delete / cancel orphans | on / off / off |

## 6. What could go wrong (and the mitigation)

| risk | mitigation |
|---|---|
| queued-job limits (held jobs count) | pace: submit only ready nodes; count every job of the user |
| orphans held forever after a parent fails | no speculative chains; block + alert descendants |
| exit codes that lie (partial sweeps) | manifest-based success; per-arm accounting |
| wrong runtime estimates, walltime kills | resumable jobs; estimates learn from history |
| unrepresentative calibration (node contention) | record packing density with the estimate |
| hung jobs holding a slot | detection + alert; opt-in force delete |
| duplicate / runaway submissions burning allocation | deterministic names, lock, dry run, budget, per-tick cap |
| results from different code versions compared | commit in every manifest; cross-commit warnings |
| decision points (pick the best HP, then continue) | a node cannot exist without an approved card; the next stage is a new card |
| partial writes / filesystem hiccups | atomic writes, manifest last |
| the tick silently stops (session ends, reboot) | heartbeat + stale warning; decide how ticks are driven (cron vs session) at go-live |

## 7. Site- and project-specific parts (to abstract when generalising)

| part | msdelta / Aurora today | generic form |
|---|---|---|
| batch system | PBS Pro: `qsub/qstat/qdel`, `-W depend`, "would exceed" text | backend interface (PBS, Slurm, fake) |
| queue table | capacity (2 running), debug / debug-scaling (1 R + 1 Q, 1 h), generic Q limit | config: queues with limits, max walltime, node caps |
| sweep semantics | `aurora-finetune-sweep.pbs`: arms, `=== done: N/M arms ok`, `RESUME_JOB` | "multi-task node" interface: count tasks, resume missing ones |
| provenance | `pbs/lib/code_snapshot.sh` -> `SNAPSHOT.txt` | "code version" hook |
| approval gate | card ID must be in `notes/DECISIONS.md` | pluggable approval source |
| estimates | parse `pbs/logs/*.OU` | history plugin per job kind |

## 8. Open items

- Live trial (card needed): does it submit, observe and finish correctly on real PBS?
- Whether debug-queue queued jobs count toward the generic limit (currently assumed yes).
- How ticks are driven once live (cron on the login node vs a session loop) and who is alerted.
- Generalisation: split backend / queue policy / sweep plugin / approval source into separate modules and a package boundary.

## 9. Running code from a commit (K90)

Every job runs from a per-job code snapshot (`pbs/lib/code_snapshot.sh`,
`$SCRATCH_ROOT/code-snapshots/<job>/`). Normally that is an rsync of the checkout's working
tree; `SNAPSHOT.txt` records the full commit sha, `mode: working-tree (rsync)`, a
`dirty: yes|no` flag and the dirty files, and a dirty snapshot also holds `uncommitted.diff`
(`git diff HEAD --binary` of the code paths), so any job can be rebuilt as commit (+ patch).

To run ANY commit -- e.g. an unmerged branch -- without checking it out or merging:

    pbs/qsub_ref c25-library-search -q debug -l select=1 -l walltime=00:30:00 \
        -v MODELS=sweeps/arms/x.txt,SPLIT=validation pbs/eval_grouped_retrieval.pbs

`pbs/qsub_ref <ref> <qsub args...> <script>` resolves the ref to a full sha at submit time,
submits the PBS script **from that commit** (copied to
`$SCRATCH_ROOT/code-refs/submit/<sha12>-<name>.pbs`), and adds
`MSDELTA_CODE_REF=<sha>,MSDELTA_CODE_REF_NAME=<ref>,REPO_DIR=<this checkout>` to the one `-v`
(any `-v` you pass are merged into it). The job's snapshot is then `git archive <sha>` of
the code paths plus `configs/`, and `SNAPSHOT.txt` says `mode: git-archive`, the ref, the
sha and `dirty: n/a`. A sha the repository lacks fails the job (exit 3).

Rules:
- **Anything committed comes from the commit**: Python, tests, scripts launched by path,
  rank_wrapper, sweep grids (`SWEEP_ROOT`), committed arms/models/args files, and the
  grid-check generators (run inside the snapshot).
- **Ad-hoc files passed by explicit path come from where they are**: a `MODELS`,
  `ARMS_FILE`, `ARGS_FILE` or grid the commit does not have is read from the checkout path
  given (`msdelta_code_path`). A path the commit HAS is always the commit's version, even
  if the checkout has an edited copy.
- **Outputs land in the checkout you submitted from** (`REPO_DIR`, cwd unchanged):
  `results/`, `pbs/logs`, `./runs`. They carry the sha: `SNAPSHOT.txt`, the job_finish
  notification/manifest (`commit`, `branch` = the ref), the eval JSONs' `code` field, and
  a sweep's `CODE_SNAPSHOT.txt` (config snapshot) / `CODE_SNAPSHOT-<job>.txt` (each arm).
- **From the checkout, by design**: `pbs/lib/*.sh` (code_snapshot, job_finish),
  `pbs/load_keys.sh` and `.keys`, `.venv`, data and checkpoints. So the checkout must
  itself have K90's `code_snapshot.sh` while such jobs are queued.
- Files named inside a `training.args` by repo-relative path (`configs/deepspeed-zero2.json`)
  are resolved by the trainer against the cwd, i.e. the checkout; the sweep runner warns
  when the checkout's copy differs from the commit's.
- qsub_ref refuses scripts that take no snapshot (they would silently ignore the ref) and
  warns for a script from before K90 (no `MSDELTA_CODE_REF` in it): its Python comes from
  the commit but its repo-relative reads (`tests/...`) come from the checkout.

**Resuming a sweep** (`RESUME_JOB=<job>`, by hand or by the scheduler) continues THAT job:
its config snapshot `runs/.configs-<job>` (no grid check; SWEEP_ROOT as it is now is not
read) and its code -- the commit in its `SNAPSHOT.txt`, rebuilt with git archive, or, if
that snapshot was taken from a dirty tree (or the commit is gone), that snapshot directory
itself, reused as is (`MSDELTA_CODE_REUSE`, logged with `!!!`). `RESUME_CONFIGS=current` /
`RESUME_CODE=current` restore the old behaviour (e.g. jobs from before snapshots existed).
Before K90 a resume re-read SWEEP_ROOT from the checkout, re-ran the grid check and ran
the checkout's current working tree.
