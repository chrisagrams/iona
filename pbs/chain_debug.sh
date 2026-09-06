#!/bin/bash
# Login-node supervisor: keep resubmitting 1-hour debug jobs until training finishes.
#
# Compute nodes on Polaris cannot run qsub, so this stays on the login node, polls the
# queue, and submits the next job whenever the previous one ends. Each job resumes from
# the newest checkpoint in a single stable output directory.
#
#   bash pbs/prep_data.sh                     # ONE TIME: download + warm the dataset cache
#   nohup bash pbs/chain_debug.sh > pbs/logs/chain.log 2>&1 &
#   tail -f pbs/logs/chain.log
#
# Stop it with:  touch pbs/logs/chain.stop     (or kill the pid)

set -uo pipefail   # deliberately not -e: the loop must survive a transient qstat failure

REPO_DIR=${REPO_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
cd "$REPO_DIR"

JOB_SCRIPT=${JOB_SCRIPT:-pbs/pairstream-50m.pbs}
JOB_NAME_MATCH=${JOB_NAME_MATCH:-msdelta-pairst}      # qstat truncates the name
CHAIN_NAME=${CHAIN_NAME:-pairstream-50m-chain}
CHAIN_OUTPUT_DIR=${CHAIN_OUTPUT_DIR:-/eagle/UIC-HPC/$USER/msdelta/runs/$CHAIN_NAME}
CHAIN_SAVE_STEPS=${CHAIN_SAVE_STEPS:-500}
MAX_JOBS=${MAX_JOBS:-200}
POLL_SECONDS=${POLL_SECONDS:-120}
STOP_FILE=${STOP_FILE:-$REPO_DIR/pbs/logs/chain.stop}
STALL_LIMIT=${STALL_LIMIT:-3}                          # give up after N jobs with no progress

mkdir -p "$CHAIN_OUTPUT_DIR" "$REPO_DIR/pbs/logs"
rm -f "$STOP_FILE"

log() { echo "[$(date -Is)] $*"; }

latest_checkpoint() {
    local best="" best_n=-1 n
    for c in "$CHAIN_OUTPUT_DIR"/checkpoint-*; do
        [[ -d $c ]] || continue
        n=${c##*checkpoint-}
        [[ $n =~ ^[0-9]+$ ]] || continue
        if ((n > best_n)); then best_n=$n; best=$c; fi
    done
    printf '%s' "$best"
}

latest_step() {
    local c; c=$(latest_checkpoint)
    if [[ -n $c ]]; then printf '%s' "${c##*checkpoint-}"; else printf '0'; fi
}

queue_busy() { qstat -u "$USER" -w 2>/dev/null | grep -q "$JOB_NAME_MATCH"; }

SENTINEL=${SENTINEL:-${HF_HOME:-/eagle/UIC-HPC/$USER/msdelta/huggingface}/.msdelta-prep-done}
if [[ ! -f $SENTINEL ]]; then
    echo "ERROR: dataset prep has not been run." >&2
    echo "  A 1-hour debug job cannot download ~80 GB and preprocess it before being killed," >&2
    echo "  so the chain would never advance. Run this first (once):" >&2
    echo "      bash pbs/prep_data.sh" >&2
    exit 2
fi

log "chain start"
log "  job script : $JOB_SCRIPT"
log "  output dir : $CHAIN_OUTPUT_DIR"
log "  wandb id   : $CHAIN_NAME"
log "  stop with  : touch $STOP_FILE"

submitted=0
stalled=0
while :; do
    if [[ -f $STOP_FILE ]]; then log "stop file found -- exiting"; break; fi

    if [[ -d $CHAIN_OUTPUT_DIR/final ]]; then
        log "TRAINING COMPLETE: $CHAIN_OUTPUT_DIR/final exists after $submitted job(s)"
        break
    fi

    if queue_busy; then
        sleep "$POLL_SECONDS"
        continue
    fi

    if ((submitted >= MAX_JOBS)); then
        log "hit MAX_JOBS=$MAX_JOBS -- exiting"; break
    fi

    step_before=$(latest_step)
    ckpt=$(latest_checkpoint)

    vars="RUN_MODE=full,CHAIN_OUTPUT_DIR=$CHAIN_OUTPUT_DIR,CHAIN_SAVE_STEPS=$CHAIN_SAVE_STEPS"
    [[ -n ${ARGS_FILE:-} ]] && vars="$vars,ARGS_FILE=$ARGS_FILE"
    [[ -n ${PHASE:-} ]] && vars="$vars,PHASE=$PHASE"
    vars="$vars,WANDB_RUN_ID=$CHAIN_NAME,WANDB_RESUME=allow"
    if [[ -n $ckpt ]]; then
        vars="$vars,RESUME_FROM_CHECKPOINT=$ckpt"
        log "submitting job $((submitted + 1)), resuming from step $step_before"
    else
        log "submitting job $((submitted + 1)), starting from scratch"
    fi

    if ! jobid=$(qsub -v "$vars" "$JOB_SCRIPT" 2>&1); then
        log "qsub FAILED: $jobid -- retrying in $POLL_SECONDS s"
        sleep "$POLL_SECONDS"
        continue
    fi
    submitted=$((submitted + 1))
    log "  submitted $jobid"

    # wait for it to appear, then to leave the queue
    for _ in $(seq 1 30); do queue_busy && break; sleep 5; done
    while queue_busy; do
        [[ -f $STOP_FILE ]] && break
        sleep "$POLL_SECONDS"
    done

    step_after=$(latest_step)
    log "  job ended: step $step_before -> $step_after"
    if ((step_after <= step_before)); then
        stalled=$((stalled + 1))
        log "  WARNING: no progress ($stalled/$STALL_LIMIT). Check the run's train-*.err"
        if ((stalled >= STALL_LIMIT)); then
            log "ABORTING: $STALL_LIMIT consecutive jobs made no progress"
            break
        fi
    else
        stalled=0
    fi
done
log "chain finished after $submitted job(s); last step $(latest_step)"
