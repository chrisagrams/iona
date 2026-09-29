#!/bin/bash
# Two concurrent 200m arms: on the SAME card (tiles 0,1) vs DIFFERENT cards (tiles 0,2).
# If the shared-HBM story is right, same-card faults and different-card does not.
cd /home/khuss/code/msdelta
source "${REPO_DIR:-${PBS_O_WORKDIR:-$PWD}}/pbs/lib/load_frameworks.sh"
export ONEAPI_DEVICE_SELECTOR=level_zero:gpu ZE_FLAT_DEVICE_HIERARCHY=FLAT
export PYTHONPATH=/home/khuss/code/msdelta
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 HF_HOME=/lus/flare/projects/UIC-HPC/khuss/msdelta/huggingface
export WANDB_MODE=disabled WANDB_DISABLED=true
unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy
export OMP_NUM_THREADS=4
C200=/flare/UIC-HPC/khuss/msdelta/pretrained/msdelta-200m-production-01-checkpoint-192799
OUT=/lus/flare/projects/UIC-HPC/khuss/msdelta/runs/sibling; mkdir -p "$OUT"

one() {  # tile, tag
  ZE_AFFINITY_MASK=$1 \
  /home/khuss/code/msdelta/.venv/bin/python -m msdelta.finetune_contrastive \
    --args_file configs/finetune-contrastive-50m/training.args \
    --pretrained_path "$C200" --output_dir "$OUT/$2" \
    --max_steps 150 --save_strategy no --report_to none --logging_steps 50 \
    --ddp_backend "" > "$OUT/$2.log" 2>&1
}
pair() {  # tagA tileA tagB tileB label
  echo "--- $5 (tiles $2 and $4) ---"
  one "$2" "$1" & local p1=$!
  one "$4" "$3" & local p2=$!
  wait $p1; local s1=$?
  wait $p2; local s2=$?
  for t in "$1" "$3"; do
    if grep -q 'Segmentation fault from GPU' "$OUT/$t.log"; then v=GPUFAULT
    elif grep -q 'train_runtime' "$OUT/$t.log"; then v=PASS; else v="OTHER"; fi
    echo "    $t: $v  $(grep -oE 'reserved [0-9.]+ GB' "$OUT/$t.log" | tail -1)"
  done
}
pair sameA 0 sameB 1 "SAME CARD"
pair diffA 0 diffB 2 "DIFFERENT CARDS"
