#!/bin/bash
# K168-C + K162/K167: run ON a held debug node (ssh in; see memory interactive-debug-jobs). Two things side by side:
#   tiles 0-1   K168-C: 400m consensus seed 0 as-is vs with --gradcache_trim_padding false (old .venv, 20 steps,
#               PYTHONFAULTHANDLER on, tile memory traced with xpu-smi) -- does variable chunk width cause the fault?
#   tiles 2-11  K167-I: P2 Pairformer on .venv-2026 (frameworks/2026.1.0): eager vs torch.compile, 10 tiles x micro
#               24 = 480 per step, stop 300, step time over steps 150-300 (pbs/diag/p2_cost.pbs's measure).
#   bash pbs/diag/k168_node.sh <tag>
set -uo pipefail
cd "$(dirname "$0")/../.." || exit 1
REPO=$PWD; TAG=${1:-k168}
S=/lus/flare/projects/UIC-HPC/khuss/msdelta; D=$S/tmp/sweep-k168; OUT=$S/runs/k168/$TAG; mkdir -p "$OUT"
hostname > "$OUT/nodefile"

# --- memory trace (card 0 = tiles 0 and 1)
( command -v xpu-smi >/dev/null || module load xpu-smi 2>/dev/null
  xpu-smi dump -d 0 -m 0,5,18 -i 1 > "$OUT/xpusmi_card0.csv" 2>&1 ) &
SMI=$!

# --- K168-C: the 400m arm, as-is and no-trim, one launcher run (slots 0 and 1)
( PBS_NODEFILE=$OUT/nodefile PBS_JOBID=$TAG-400m REPO_DIR=$REPO SWEEP_ROOT=$D SKIP_GRID_CHECK=1 \
  SWEEP_MODULE=msdelta.finetune_contrastive ARMS_FILE=$D/arms.txt MAX_STEPS=20 PYTHONFAULTHANDLER=1 \
  bash pbs/aurora-finetune-sweep.pbs > "$OUT/400m.log" 2>&1; echo "400m rc=$?" >> "$OUT/400m.log" ) &
C400=$!

# --- K167-I: compile vs eager on tiles 2-11
elapsed() { tr '\r' '\n' < "$1" | grep -o " $2/540423 \[[0-9:]*" | head -1 | grep -o "[0-9:]*$" \
            | awk -F: '{s=0; for (i=1; i<=NF; i++) s = s*60 + $i; print s}'; }
for spec in ${K167_ARMS:-p2-cz32-k5:0:0 p2-cz32-k5:0:150 p2-cz32-k5:1:150 p2-cz32-k1:1:150 p2-cz32-k1:0:150}; do
    IFS=: read -r arm comp pad <<<"$spec"; t=$arm-c$comp-p$pad
    sed -e 's/^--eval_strategy .*/--eval_strategy no/' -e 's/^--save_steps .*/--save_steps 300/' \
        -e 's/^--logging_steps .*/--logging_steps 10/' -e 's/^--report_to .*/--report_to none/' \
        "configs/p2/$arm/training.args" > "$OUT/$t.args"
    (( pad > 0 )) && echo "--pad_to_multiple_of $pad" >> "$OUT/$t.args"
    echo "=== $(date -Is) $t" | tee -a "$OUT/summary.txt"
    timeout 900 env FRAMEWORKS_MODULE=frameworks/2026.1.0 VENV_DIR=$REPO/.venv-2026 PBS_NODEFILE=$OUT/nodefile \
        PBS_JOBID=$TAG-$t ARGS_FILE="$OUT/$t.args" CHECKPOINT_DIR="$OUT" RUN_NAME="$t" \
        HF_HOME=$S/huggingface PREPROCESSED_DATASET_DIR=$S/data/p2-cap150-half/preprocessed \
        ZE_AFFINITY_MASK=2,3,4,5,6,7,8,9,10,11 XPUS_PER_HOST=10 MICRO_BATCH_SIZE=24 GLOBAL_BATCH_SIZE=480 \
        TORCH_COMPILE=$comp PROBE_EXECUTION=off CCL_KVS_MODE=pmi WANDB_MODE=disabled MSDELTA_STOP_AT_STEP=300 \
        bash pbs/aurora-pretrain.pbs > "$OUT/$t.log" 2>&1
    rc=$?
    err=$(find "$OUT/$t" -name "train-*.err" 2>/dev/null | head -1)
    t1=$( [[ -n $err ]] && elapsed "$err" 1 ); a=$( [[ -n $err ]] && elapsed "$err" 150 ); b=$( [[ -n $err ]] && elapsed "$err" 300 )
    sps=NA; [[ -n $a && -n $b ]] && sps=$(awk -v a="$a" -v b="$b" 'BEGIN{printf "%.3f", (b-a)/150}')
    echo "RESULT $t rc=$rc s_per_step=$sps t1=${t1:-NA} t150=${a:-NA} t300=${b:-NA}" | tee -a "$OUT/summary.txt"
done
wait $C400; kill $SMI 2>/dev/null
grep -E "FAIL|done:|rc=" "$OUT/400m.log" | tee -a "$OUT/summary.txt"
