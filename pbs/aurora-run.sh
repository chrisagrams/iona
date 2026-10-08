#!/usr/bin/env bash
# Launch one Iona training run on the comma-separated hosts in IONA_HOSTS.
# Usage: IONA_HOSTS=host1,host2 bash pbs/aurora-run.sh [iona.train arguments]
# Requires the environment exported by pbs/aurora-env.sh.
#
# Arguments of the form --config.<field>=<value> are collected into a single
# --config_overrides, which replaces any --config_overrides in ARGS_FILE. All other
# arguments are passed to iona.train after the defaults below, so they win.
#
# Under a W&B sweep agent (WANDB_SWEEP_ID is set), the run is named after the
# agent's WANDB_RUN_ID and RUN_NAME and LOG_DIR are ignored. A trial requeued after
# a walltime stop keeps its run ID, so it resumes from its latest checkpoint.
set -euo pipefail

IFS=, read -r -a HOSTS <<<"${IONA_HOSTS:?set IONA_HOSTS to a comma-separated host list}"
NUM_HOSTS=${#HOSTS[@]}
TOTAL_XPUS=$((NUM_HOSTS * XPUS_PER_HOST))

config_overrides=
EXTRA_ARGS=()
for arg in "$@"; do
    case "$arg" in
        --config.*=*)
            override=${arg#--config.}
            config_overrides+="${config_overrides:+,}$override"
            ;;
        *)
            EXTRA_ARGS+=("$arg")
            ;;
    esac
done
if [[ -n $config_overrides ]]; then
    EXTRA_ARGS+=(--config_overrides "$config_overrides")
fi

if [[ -n ${WANDB_SWEEP_ID:-} ]]; then
    RUN_NAME=$CONFIG_RUN_NAME-${WANDB_RUN_ID:?the W&B agent did not set WANDB_RUN_ID}
    LOG_DIR=$LOG_ROOT/$RUN_NAME/logs
    RESUME_FROM_CHECKPOINT=auto
else
    RUN_NAME=${RUN_NAME:-${CONFIG_RUN_NAME}-${JOB_ID}}
    LOG_DIR=${LOG_DIR:-$LOG_ROOT/$RUN_NAME/logs}
    export WANDB_RUN_ID=${WANDB_RUN_ID:-$RUN_NAME}
fi
if [[ $LOG_DIR != /* ]]; then
    echo "LOG_DIR must be an absolute shared-filesystem path" >&2
    exit 2
fi
RUN_DIR=$CHECKPOINT_DIR/$RUN_NAME
mkdir -p "$RUN_DIR" "$LOG_DIR"
if [[ ${RESUME_FROM_CHECKPOINT:-} == latest || ${RESUME_FROM_CHECKPOINT:-} == auto ]]; then
    RESUME_FROM_CHECKPOINT=$("$VENV_DIR/bin/python" - "$RUN_DIR" "$RESUME_FROM_CHECKPOINT" <<'PY'
import re
import sys
from pathlib import Path

run_dir = Path(sys.argv[1])
candidates = []
for path in run_dir.glob("checkpoint-*"):
    match = re.fullmatch(r"checkpoint-(\d+)", path.name)
    if match and path.is_dir() and (path / "trainer_state.json").is_file():
        candidates.append((int(match.group(1)), path))
if candidates:
    print(max(candidates)[1])
elif sys.argv[2] == "latest":
    raise SystemExit(f"no complete checkpoint-N directories found in {run_dir}")
PY
    )
    if [[ -n $RESUME_FROM_CHECKPOINT ]]; then
        echo "resuming latest checkpoint: $RESUME_FROM_CHECKPOINT"
    fi
fi
DDP_ARGS_FILE=$RUN_DIR/training-ddp.args
{
    awk '
        $1 != "--deepspeed" &&
        $1 != "--logging_nan_inf_filter"
    ' "$ARGS_FILE"
    printf '%s\n' '--logging_nan_inf_filter false'
} >"$DDP_ARGS_FILE"

NODE_LIST=$RUN_DIR/pbs-nodes
printf '%s\n' "${HOSTS[@]}" >"$NODE_LIST"

DENOMINATOR=$((MICRO_BATCH_SIZE * TOTAL_XPUS))
if ((GLOBAL_BATCH_SIZE % DENOMINATOR != 0)); then
    echo "global batch size $GLOBAL_BATCH_SIZE is not divisible by" \
        "$MICRO_BATCH_SIZE micro-batch x $TOTAL_XPUS XPUs" >&2
    exit 2
fi
GRADIENT_ACCUMULATION_STEPS=$((GLOBAL_BATCH_SIZE / DENOMINATOR))
if ((GRADIENT_ACCUMULATION_STEPS < 1)); then
    echo "global batch size $GLOBAL_BATCH_SIZE is smaller than one distributed micro-batch" >&2
    exit 2
fi

echo "$(date -Is) run=$RUN_NAME hosts=$NUM_HOSTS total_xpus=$TOTAL_XPUS accumulation=$GRADIENT_ACCUMULATION_STEPS logs=$LOG_DIR"
echo "nodes:"
sed 's/^/  /' "$NODE_LIST"

export MASTER_ADDR=${HOSTS[0]}
export MASTER_PORT=${MASTER_PORT:-29500}
export IONA_WORLD_SIZE=$TOTAL_XPUS
export LOG_DIR
export WANDB_DIR=${WANDB_DIR:-$LOG_DIR}

TRAIN_ARGS=(
    --args_file "$DDP_ARGS_FILE"
    --dataset_cache_dir "$DATASET_CACHE_DIR"
    --preprocessed_dataset_dir "$PREPROCESSED_DATASET_DIR"
    --preprocessed_probe_dir "$PREPROCESSED_PROBE_DIR"
    --run_name "$RUN_NAME"
    --output_dir "$RUN_DIR"
    --per_device_train_batch_size "$MICRO_BATCH_SIZE"
    --gradient_accumulation_steps "$GRADIENT_ACCUMULATION_STEPS"
    --ddp_backend xccl
    --logging_nan_inf_filter false
    --probe_execution "$PROBE_EXECUTION"
)
if [[ $PROBE_EXECUTION == sidecar ]]; then
    TRAIN_ARGS+=(
        --sidecar_launcher "$REPO_DIR/pbs/aurora-probe.sh"
        --sidecar_denoise_device "xpu:$DENOISE_XPU_TILES"
        --sidecar_retrieval_device "xpu:$RETRIEVAL_XPU_TILES"
    )
fi
if [[ -n ${RESUME_FROM_CHECKPOINT:-} ]]; then
    TRAIN_ARGS+=(--resume_from_checkpoint "$RESUME_FROM_CHECKPOINT")
fi

exec mpiexec \
    --verbose \
    --envall \
    --pmi=pmix \
    --no-vni \
    -n "$TOTAL_XPUS" \
    --ppn "$XPUS_PER_HOST" \
    --cpu-bind "$CPU_BIND" \
    --hostfile="$NODE_LIST" \
    /usr/bin/env bash "$REPO_DIR/pbs/aurora-rank.sh" \
    "$VENV_DIR/bin/python" \
    -m iona.train \
    "${TRAIN_ARGS[@]}" \
    ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"} \
    >"$LOG_DIR/train-$JOB_ID.out" \
    2>"$LOG_DIR/train-$JOB_ID.err"
