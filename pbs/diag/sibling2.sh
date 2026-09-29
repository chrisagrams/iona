#!/bin/bash
cd /home/khuss/code/msdelta
source "${REPO_DIR:-${PBS_O_WORKDIR:-$PWD}}/pbs/lib/load_frameworks.sh"
export ONEAPI_DEVICE_SELECTOR=level_zero:gpu ZE_FLAT_DEVICE_HIERARCHY=FLAT
export PYTHONPATH=/home/khuss/code/msdelta
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 HF_HOME=/lus/flare/projects/UIC-HPC/khuss/msdelta/huggingface
export WANDB_MODE=disabled WANDB_DISABLED=true
unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy
export OMP_NUM_THREADS=4 MSDELTA_XPUS_PER_HOST=1 MSDELTA_WORLD_SIZE=1
export CCL_PROCESS_LAUNCHER=pmix CCL_ATL_TRANSPORT=mpi CCL_KVS_MODE=mpi
unset CCL_ZE_IPC CCL_ZE_IPC_EXCHANGE
export MASTER_ADDR=$(hostname)
C200=/flare/UIC-HPC/khuss/msdelta/pretrained/msdelta-200m-production-01-checkpoint-192799
OUT=/lus/flare/projects/UIC-HPC/khuss/msdelta/runs/sibling2; mkdir -p "$OUT"

one() {  # tile tag port
  ZE_AFFINITY_MASK=$1 MASTER_PORT=$3 \
  mpiexec --envall --pmi=pmix -n 1 --ppn 1 --cpu-bind depth --depth 8 \
    /home/khuss/code/msdelta/pbs/rank_wrapper.sh \
    /home/khuss/code/msdelta/.venv/bin/python -m msdelta.finetune_contrastive \
      --args_file configs/finetune-contrastive-50m/training.args \
      --pretrained_path "$C200" --output_dir "$OUT/$2" \
      --max_steps 100 --save_strategy no --report_to none --logging_steps 50 \
      > "$OUT/$2.log" 2>&1
}
pair() {  # tagA tileA tagB tileB label
  echo "--- $5 : tiles $2 and $4 ---"
  one "$2" "$1" 29601 & local p1=$!
  one "$4" "$3" 29602 & local p2=$!
  wait $p1; wait $p2
  for t in "$1" "$3"; do
    if grep -q 'Segmentation fault from GPU' "$OUT/$t.log"; then v=GPUFAULT
    elif grep -q 'train_runtime' "$OUT/$t.log"; then v=PASS
    else v="OTHER: $(grep -oE '[A-Za-z]+Error:[^|]{0,50}' "$OUT/$t.log" | tail -1)"; fi
    echo "    $t  $v   $(grep -oE 'reserved [0-9.]+ GB' "$OUT/$t.log" | tail -1)"
  done
}
pair sameA 0 sameB 1 "SAME CARD"
pair diffA 0 diffB 2 "DIFFERENT CARDS"
