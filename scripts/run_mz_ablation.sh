#!/usr/bin/env bash
# Run the A1/B1/C1/D1 m/z ablation sequentially with two local GPUs.
#
# Batch size, accumulation, warmup, and total exposure are defined in the
# shared config. The launcher overrides only the two architecture switches.

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_DIR=$(cd -- "$SCRIPT_DIR/.." && pwd)
CONFIG=${CONFIG:-$REPO_DIR/configs/mz_ablation.yaml}
TORCHRUN=${TORCHRUN:-torchrun}
RUN_GROUP=${RUN_GROUP:-$(date -u +%Y%m%dT%H%M%SZ)}

# Respect an explicit device selection while defaulting to the first two GPUs.
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-4}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-4}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

architectures=(A B C D)
absolute_mz=(false true false true)
delta_bias=(false false true true)

cd "$REPO_DIR"

for i in "${!architectures[@]}"; do
    architecture=${architectures[$i]}
    run_name="mz-ablation-${architecture}1-${RUN_GROUP}"

    echo "[$(date -Is)] starting $run_name on CUDA devices $CUDA_VISIBLE_DEVICES"
    "$TORCHRUN" \
        --standalone \
        --nnodes=1 \
        --nproc_per_node=2 \
        -m msdelta.train \
        --config "$CONFIG" \
        --run-name "$run_name" \
        --architecture_id "$architecture" \
        --use_absolute_mz "${absolute_mz[$i]}" \
        --use_delta_mz_bias "${delta_bias[$i]}"
    echo "[$(date -Is)] completed $run_name"
done
