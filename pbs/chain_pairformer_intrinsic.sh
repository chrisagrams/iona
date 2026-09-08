#!/bin/bash
# Login-node supervisor for the SPECTRUM-INTRINSIC Pairformer on the DEBUG queue.
#
# Same kill-resistant chain as pbs/chain_pairformer.sh, but pointed at the intrinsic model
# (configs/msdelta-pairformer-intrinsic-50m/training.args) with its own output dir + wandb id so
# it never collides with a base-pairformer run. Each 1-hour debug job checkpoints and the next
# resubmission resumes from the newest checkpoint, so repeated 1-hour allocations survive being
# killed. CHAIN_SAVE_STEPS defaults BELOW eval_steps (500) so a checkpoint always exists before
# the first eval -- an eval-time failure then resumes instead of restarting from step 0.
#
# ONE TIME (a 1-hour job cannot download ~80 GB; run on a login node first):
#     bash pbs/prep_data.sh
#
# RUN IT -- inside tmux so the supervisor survives disconnects (see KILL RESISTANCE below):
#     tmux new -s intrinsic
#     bash pbs/chain_pairformer_intrinsic.sh            # default model (B5: triangle + writeback)
#     # detach with: Ctrl-b then d
#
# FULL MODEL (adds triangle attention; memory-bound arm, micro-batch/peaks auto-reduced):
#     bash pbs/chain_pairformer_intrinsic.sh B4
#
# STOP:  touch pbs/logs/chain.stop      (or kill the tmux session)

set -uo pipefail

REPO_DIR=${REPO_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
cd "$REPO_DIR"

PHASE_ARG=${1:-DEFAULT}
case "$PHASE_ARG" in
    B1|B2|B3|B4|B5) export PHASE=$PHASE_ARG; SUFFIX=${PHASE_ARG,,} ;;
    DEFAULT)        unset PHASE 2>/dev/null || true; SUFFIX=default ;;
    *) echo "ERROR: phase must be B1..B5 (or omit for the config default), got '$PHASE_ARG'" >&2
       exit 2 ;;
esac

export JOB_SCRIPT=pbs/pairformer-50m.pbs           # same launcher; ARGS_FILE swaps the model
export ARGS_FILE=configs/msdelta-pairformer-intrinsic-50m/training.args
export JOB_NAME_MATCH=msdelta-pairfo               # #PBS -N msdelta-pairformer-50m; qstat truncates
export CHAIN_NAME=pairformer-intrinsic-50m-${SUFFIX}-chain
export CHAIN_OUTPUT_DIR=${CHAIN_OUTPUT_DIR:-/eagle/UIC-HPC/$USER/msdelta/runs/$CHAIN_NAME}
export CHAIN_SAVE_STEPS=${CHAIN_SAVE_STEPS:-250}   # < eval_steps(500): checkpoint before eval

echo "=== pairformer-intrinsic debug chain ==="
echo "  phase       : ${PHASE:-<config default (B5: triangle + writeback)>}"
echo "  args file   : $ARGS_FILE"
echo "  output dir  : $CHAIN_OUTPUT_DIR"
echo "  wandb id    : $CHAIN_NAME"
echo "  save every  : $CHAIN_SAVE_STEPS steps (eval every 500)"
echo

exec bash "$REPO_DIR/pbs/chain_debug.sh"

# ---------------------------------------------------------------------------------
# KILL RESISTANCE -- three layers:
#   1. Per-allocation: each debug job saves a checkpoint every CHAIN_SAVE_STEPS; when the 1-hour
#      wall-clock kills it, the next submission resumes from the newest checkpoint. No lost work
#      beyond the last <=250 steps.
#   2. Supervisor survives disconnect: run this inside tmux (above). `nohup bash ... &` also works
#      but tmux lets you reattach (`tmux attach -t intrinsic`) and survives more cases.
#   3. Supervisor is idempotent: if the login node reboots and kills tmux too, just re-run this
#      exact command -- it finds the latest checkpoint in CHAIN_OUTPUT_DIR and resumes. The
#      checkpoints on /eagle are the durable state; the supervisor is disposable.
# The chain stops on its own when CHAIN_OUTPUT_DIR/final appears (training complete) or after
# STALL_LIMIT consecutive no-progress jobs. Watch it with: tail -f pbs/logs/chain.log
# ---------------------------------------------------------------------------------
