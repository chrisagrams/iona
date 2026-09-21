#!/bin/bash
# Does the fault depend on HOW MANY arms share the node? Bisect 2 -> 4 -> 8 -> 12.
# Solo and same-card-pair both reserve 38.75 GB and pass; the grid reserves 67.11 GB
# at step 50 with 12 arms and dies. Something about the count, not the card.
cd /home/khuss/code/msdelta
source /usr/share/lmod/lmod/init/bash 2>/dev/null; module load frameworks/2025.3.1 2>/dev/null
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
OUT=/lus/flare/projects/UIC-HPC/khuss/msdelta/runs/armcount; mkdir -p "$OUT"

one() {  # tile tag port
  ZE_AFFINITY_MASK=$1 MASTER_PORT=$3 \
  mpiexec --envall --pmi=pmix -n 1 --ppn 1 --cpu-bind depth --depth 4 \
    /home/khuss/code/msdelta/pbs/rank_wrapper.sh \
    /home/khuss/code/msdelta/.venv/bin/python -m msdelta.finetune_contrastive \
      --args_file configs/finetune-contrastive-50m/training.args \
      --pretrained_path "$C200" --output_dir "$OUT/$2" \
      --max_steps 80 --save_strategy no --report_to none --logging_steps 40 \
      > "$OUT/$2.log" 2>&1
}
wave() {  # n
  local n=$1 pids=()
  echo "--- $n concurrent arms ---"
  for (( t=0; t<n; t++ )); do one "$t" "n${n}_t${t}" $(( 29700 + t )) & pids+=($!); done
  for p in "${pids[@]}"; do wait "$p"; done
  local ok=0 fault=0 maxres=0
  for (( t=0; t<n; t++ )); do
    f="$OUT/n${n}_t${t}.log"
    grep -q 'Segmentation fault from GPU' "$f" && fault=$((fault+1))
    grep -q 'train_runtime' "$f" && ok=$((ok+1))
    r=$(grep -oE 'reserved [0-9.]+' "$f" | tail -1 | awk '{print $2}')
    [ -n "$r" ] && awk -v a="$r" -v b="$maxres" 'BEGIN{exit !(a>b)}' && maxres=$r
  done
  echo "    n=$n  pass=$ok  gpufault=$fault  max reserved/tile=${maxres} GB"
}
for n in 4 8 12; do wave "$n"; done
