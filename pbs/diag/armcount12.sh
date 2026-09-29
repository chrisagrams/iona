#!/bin/bash
# 12 concurrent arms -- the grid's shape -- while sampling HOST memory every 10s.
# Host RAM is the only resource genuinely shared by all twelve processes, and it is the
# remaining candidate after card placement was ruled out.
cd /home/khuss/code/msdelta
source "${REPO_DIR:-${PBS_O_WORKDIR:-$PWD}}/pbs/lib/load_frameworks.sh"
export ONEAPI_DEVICE_SELECTOR=level_zero:gpu ZE_FLAT_DEVICE_HIERARCHY=FLAT
export PYTHONPATH=/home/khuss/code/msdelta
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 HF_HOME=/lus/flare/projects/UIC-HPC/khuss/msdelta/huggingface
export WANDB_MODE=disabled WANDB_DISABLED=true
unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy
export OMP_NUM_THREADS=4 MSDELTA_XPUS_PER_HOST=1 MSDELTA_WORLD_SIZE=1
export CCL_PROCESS_LAUNCHER=pmix CCL_ATL_TRANSPORT=mpi CCL_KVS_MODE=${CCL_KVS_MODE:-pmi}  # K138-I: "mpi" breaks on the 2026-09 stack
unset CCL_ZE_IPC CCL_ZE_IPC_EXCHANGE
export MASTER_ADDR=$(hostname)
C200=/flare/UIC-HPC/khuss/msdelta/pretrained/msdelta-200m-production-01-checkpoint-192799
OUT=/lus/flare/projects/UIC-HPC/khuss/msdelta/runs/armcount12; mkdir -p "$OUT"

( while true; do
    echo "$(date +%H:%M:%S) $(free -g | awk '/^Mem:/{printf \"used %sG avail %sG of %sG\", $3, $7, $2}')"
    sleep 10
  done ) > "$OUT/hostmem.log" 2>&1 &
SAMPLER=$!

pids=()
for (( t=0; t<12; t++ )); do
  ZE_AFFINITY_MASK=$t MASTER_PORT=$(( 29800 + t )) \
  mpiexec --envall --pmi=pmix -n 1 --ppn 1 --cpu-bind depth --depth 4 \
    /home/khuss/code/msdelta/pbs/rank_wrapper.sh \
    /home/khuss/code/msdelta/.venv/bin/python -m msdelta.finetune_contrastive \
      --args_file configs/finetune-contrastive-50m/training.args \
      --pretrained_path "$C200" --output_dir "$OUT/t$t" \
      --max_steps 80 --save_strategy no --report_to none --logging_steps 40 \
      > "$OUT/t$t.log" 2>&1 &
  pids+=($!)
done
for p in "${pids[@]}"; do wait "$p"; done
kill $SAMPLER 2>/dev/null

ok=0; fault=0
for (( t=0; t<12; t++ )); do
  grep -q 'Segmentation fault from GPU' "$OUT/t$t.log" && fault=$((fault+1))
  grep -q 'train_runtime' "$OUT/t$t.log" && ok=$((ok+1))
done
echo "RESULT 12 arms: pass=$ok gpufault=$fault"
echo "reserved per tile:"; for (( t=0; t<12; t++ )); do
  printf "  t%-2s %s\n" "$t" "$(grep -oE 'peak [0-9.]+ GB reserved [0-9.]+ GB' "$OUT/t$t.log" | tail -1)"
done
echo "host memory low-water mark:"; sort -t' ' -k4 -n "$OUT/hostmem.log" 2>/dev/null | head -2
