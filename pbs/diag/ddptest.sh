#!/bin/bash
# Run the REAL training entry point at 200m, varying only the distributed backend.
cd /home/khuss/code/msdelta
source "${REPO_DIR:-${PBS_O_WORKDIR:-$PWD}}/pbs/lib/load_frameworks.sh"
export ONEAPI_DEVICE_SELECTOR=level_zero:gpu ZE_FLAT_DEVICE_HIERARCHY=FLAT ZE_AFFINITY_MASK=0
export PYTHONPATH=/home/khuss/code/msdelta HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
export HF_HOME=/lus/flare/projects/UIC-HPC/khuss/msdelta/huggingface
export OMP_NUM_THREADS=4 MSDELTA_XPUS_PER_HOST=1 MSDELTA_WORLD_SIZE=1
export CCL_PROCESS_LAUNCHER=pmix CCL_ATL_TRANSPORT=mpi CCL_KVS_MODE=${CCL_KVS_MODE:-pmi}  # K138-I: "mpi" breaks on the 2026-09 stack
unset CCL_ZE_IPC CCL_ZE_IPC_EXCHANGE
export MASTER_ADDR=$(hostname) MASTER_PORT=29511
C200=/flare/UIC-HPC/khuss/msdelta/pretrained/msdelta-200m-production-01-checkpoint-192799
OUT=/lus/flare/projects/UIC-HPC/khuss/msdelta/runs/ddptest-$$

run() {
  local tag=$1; shift
  printf "  %-22s " "$tag"
  mpiexec --envall --pmi=pmix -n 1 --ppn 1 --cpu-bind depth --depth 8 \
    /home/khuss/code/msdelta/pbs/rank_wrapper.sh \
    /home/khuss/code/msdelta/.venv/bin/python -m msdelta.finetune_contrastive \
      --args_file configs/finetune-contrastive-50m/training.args \
      --pretrained_path "$C200" --output_dir "$OUT/$tag" \
      --max_steps 40 --save_strategy no --report_to none --logging_steps 10 \
      "$@" > "$OUT-$tag.log" 2>&1
  local s=$?
  if [[ $s -eq 0 ]]; then echo "PASS"
  elif grep -q 'Segmentation fault from GPU' "$OUT-$tag.log"; then
    echo "GPUFAULT  $(grep -oE 'at 0x[0-9a-f]+' "$OUT-$tag.log" | tail -1)"
  else echo "FAIL($s)  $(grep -oE '[A-Za-z]+Error[^|]{0,70}' "$OUT-$tag.log" | tail -1)"; fi
}
mkdir -p "$OUT"
echo "=== 200m contrastive, real entry point, varying ONLY the backend ==="
run "as-is_xccl"
run "no_ddp_backend"       --ddp_backend ""
run "no_workers"           --dataloader_num_workers 0
run "fp32_no_bf16"         --bf16 false
echo "  logs: $OUT-*.log"
