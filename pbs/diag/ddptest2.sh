#!/bin/bash
cd /home/khuss/code/msdelta
source "${REPO_DIR:-${PBS_O_WORKDIR:-$PWD}}/pbs/lib/load_frameworks.sh"
export ONEAPI_DEVICE_SELECTOR=level_zero:gpu ZE_FLAT_DEVICE_HIERARCHY=FLAT ZE_AFFINITY_MASK=0
export PYTHONPATH=/home/khuss/code/msdelta
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 HF_HOME=/lus/flare/projects/UIC-HPC/khuss/msdelta/huggingface
# The previous attempt hung retrying urllib3 for minutes despite --report_to none.
export WANDB_MODE=disabled WANDB_DISABLED=true
unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy
export OMP_NUM_THREADS=4 MSDELTA_XPUS_PER_HOST=1 MSDELTA_WORLD_SIZE=1
export CCL_PROCESS_LAUNCHER=pmix CCL_ATL_TRANSPORT=mpi CCL_KVS_MODE=mpi
unset CCL_ZE_IPC CCL_ZE_IPC_EXCHANGE
export MASTER_ADDR=$(hostname) MASTER_PORT=29512
C200=/flare/UIC-HPC/khuss/msdelta/pretrained/msdelta-200m-production-01-checkpoint-192799
OUT=/lus/flare/projects/UIC-HPC/khuss/msdelta/runs/ddp2
mkdir -p "$OUT"
run() {
  local tag=$1; shift
  mpiexec --envall --pmi=pmix -n 1 --ppn 1 --cpu-bind depth --depth 8 \
    /home/khuss/code/msdelta/pbs/rank_wrapper.sh \
    /home/khuss/code/msdelta/.venv/bin/python -m msdelta.finetune_contrastive \
      --args_file configs/finetune-contrastive-50m/training.args \
      --pretrained_path "$C200" --output_dir "$OUT/$tag" \
      --max_steps 40 --save_strategy no --report_to none --logging_steps 10 \
      "$@" > "$OUT/$tag.log" 2>&1
  local s=$?
  if [[ $s -eq 0 ]]; then echo "$tag PASS"
  elif grep -q 'Segmentation fault from GPU' "$OUT/$tag.log"; then echo "$tag GPUFAULT"
  else echo "$tag FAIL($s)"; fi
}
run "A_as_is"
run "B_no_ddp_backend" --ddp_backend ""
