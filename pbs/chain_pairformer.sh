#!/bin/bash
# Login-node supervisor for the AF3-style Pairformer experiment on the DEBUG queue.
#
# A single debug job is capped at 1 hour, which is far short of the 56250-step run. This
# wrapper points pbs/chain_debug.sh at pbs/pairformer-50m.pbs and keeps resubmitting a
# fresh 1-hour debug job whenever the previous one ends, each resuming from the newest
# checkpoint in one stable output directory. Everything (wandb id, output dir, job-name
# match) is derived from the phase so chained runs never collide with each other.
#
# ONE TIME (downloads + warms the ~80 GB dataset cache; a 1-hour job cannot do this):
#     bash pbs/prep_data.sh
#
# RUN THE DEFAULT MODEL (B3 + write-back, as config.json ships it):
#     nohup bash pbs/chain_pairformer.sh > pbs/logs/chain-pairformer.log 2>&1 &
#     tail -f pbs/logs/chain-pairformer.log
#
# RUN ONE ABLATION PHASE (B1|B2|B3|B4|B5):
#     nohup bash pbs/chain_pairformer.sh B3 > pbs/logs/chain-pairformer-b3.log 2>&1 &
#
# RUN THE WHOLE LADDER, ONE PHASE AT A TIME (sequential -- see note at the bottom):
#     nohup bash -c 'for P in B1 B2 B3 B5; do bash pbs/chain_pairformer.sh $P; done' \
#         > pbs/logs/chain-pairformer-ladder.log 2>&1 &
#
# STOP everything:  touch pbs/logs/chain.stop     (or kill the pid)

set -uo pipefail

REPO_DIR=${REPO_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
cd "$REPO_DIR"

# Phase comes from $1 (default: config.json default, which is B3 + write-back).
PHASE_ARG=${1:-DEFAULT}
case "$PHASE_ARG" in
    B1|B2|B3|B4|B5)
        export PHASE=$PHASE_ARG
        SUFFIX=${PHASE_ARG,,} ;;
    DEFAULT)
        unset PHASE 2>/dev/null || true   # let config.json (B3+write-back) drive
        SUFFIX=default ;;
    *)
        echo "ERROR: phase must be one of B1 B2 B3 B4 B5 (or omit for the config default), got '$PHASE_ARG'" >&2
        exit 2 ;;
esac

# Point the generic supervisor at the pairformer job. Names/dirs are per-phase so two
# phases never share a checkpoint dir or a wandb run.
export JOB_SCRIPT=pbs/pairformer-50m.pbs
export JOB_NAME_MATCH=msdelta-pairfo            # #PBS -N is msdelta-pairformer-50m; qstat truncates
export CHAIN_NAME=pairformer-50m-${SUFFIX}-chain
export CHAIN_OUTPUT_DIR=${CHAIN_OUTPUT_DIR:-/eagle/UIC-HPC/$USER/msdelta/runs/$CHAIN_NAME}
export ARGS_FILE=${ARGS_FILE:-configs/msdelta-pairformer-50m/training.args}

echo "=== pairformer debug chain ==="
echo "  phase       : ${PHASE:-<config default (B3+write-back)>}"
echo "  job script  : $JOB_SCRIPT"
echo "  output dir  : $CHAIN_OUTPUT_DIR"
echo "  wandb id    : $CHAIN_NAME"
echo

exec bash "$REPO_DIR/pbs/chain_debug.sh"

# ---------------------------------------------------------------------------------
# WHY THE LADDER RUNS ONE PHASE AT A TIME
#   All pairformer phases share the PBS job name msdelta-pairformer-50m, and the
#   supervisor's queue_busy check matches on that name. If you launched two phase
#   supervisors at once they would each see the other's job as "busy" and neither
#   would submit correctly -- and the debug queue would not run them concurrently
#   anyway. So the ladder loop above runs each phase's chain to completion (its
#   CHAIN_OUTPUT_DIR/final appears) before starting the next. Expect this to take
#   many wall-clock hours per phase; the chain survives across them.
#
# TRIANGLE ATTENTION (B4)
#   B4 is memory-bound. The pairformer PBS drops the micro-batch to 8 and caps
#   max_peaks to 96 for B4 automatically, so no extra flags are needed here -- but it
#   is the slow arm; consider running it on its own rather than inside the ladder loop.
# ---------------------------------------------------------------------------------
