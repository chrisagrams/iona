#!/bin/bash
# K66-C pipeline (approved card): smoke -> 4 capacity jobs (one per scale) -> per-scale scoring on
# validation / OOD / mouse / human. Stops (does not improvise) if the smoke or a training job fails.
source "${REPO_DIR:-${PBS_O_WORKDIR:-$PWD}}/pbs/lib/homes.sh" || exit 1   # K198a: data homes (configs/homes.env)
set -uo pipefail
cd /home/khuss/code/msdelta
SMOKE=${1:?smoke job id}
shift
PRESET="$*"          # optional: 400m=ID 200m=ID ... to resume monitoring only
S=/lus/flare/projects/UIC-HPC/khuss/msdelta
LOG=${K66_LOG:-/lus/flare/projects/UIC-HPC/khuss/msdelta/logs/k66_pipeline.log}
log() { echo "$(date -u +%FT%H:%MZ) $*" | tee -a "$LOG"; }
sub() {  # retry until the per-user queue limit lets it in
    local out
    while true; do
        out=$(qsub "$@" 2>&1)
        [[ $out == *aurora-pbs* ]] && break
        if [[ $out != *"would exceed"* ]]; then
            echo "$(date -u +%FT%H:%MZ) qsub REJECTED (not a queue limit): $out -- stopping" >> "$LOG"; exit 2
        fi
        sleep 300
    done
    echo "${out%%.*}"
}
alive() { qstat "$1" >/dev/null 2>&1; }
oulog() { ls pbs/logs/"$1".*.OU 2>/dev/null | head -1; }

if [[ -z $PRESET ]]; then
while alive "$SMOKE"; do sleep 120; done
if ! grep -q "=== done: 2/2 arms ok" "$(oulog "$SMOKE")" 2>/dev/null; then
    log "SMOKE $SMOKE FAILED or incomplete -- stopping; nothing submitted to capacity"; exit 1
fi
log "smoke $SMOKE ok"
fi

declare -A JOB WALL=([400m]=14:00:00 [200m]=10:00:00 [100m]=10:00:00 [025m]=10:00:00)
declare -A SCORED=()
NSCORED=0
ORDER=(400m 200m 100m 025m)       # longest first: two slots finish ~14 h instead of ~16 h
if [[ -n $PRESET ]]; then
    for kv in $PRESET; do JOB[${kv%%=*}]=${kv#*=}; done
    ORDER=("${!JOB[@]}")   # monitor only the preset scales
    log "monitoring preset jobs: $PRESET"
else
for s in "${ORDER[@]}"; do
    JOB[$s]=$(sub -q capacity -l select=1 -l walltime=${WALL[$s]} -N hps$s \
        -v SWEEP_ROOT=configs/sweep-hp-scale,SWEEP_MODULE=msdelta.finetune_contrastive,ARMS_FILE=sweeps/arms/hp_scale_$s.txt \
        pbs/aurora-finetune-sweep.pbs)
    log "submitted $s training: ${JOB[$s]} (walltime ${WALL[$s]})"
done
fi

score_scale() {
    local s=$1 j=$2 m=sweeps/arms/score_hp_scale_$s.txt
    { echo "# K66-C $s finals (job $j)"
      for d in $S/runs/sweep-s${s}_ck540k_*-"$j"; do
          echo "$(basename "$d" | sed 's/^sweep-//; s/-[0-9]*$//') $d/final"; done; } > "$m"
    local DS=$(dirname "$0")/k66_datasets.txt
    while read -r n q v; do
        [[ -z $n || $n == \#* ]] && continue
        local id; id=$(sub -q $q -l select=1 -l walltime=01:00:00 -N hs${s}$n \
            -v MODELS=$m,$v,OUT_DIR=$MSDELTA_EVAL/contrastive/hp-scale-$s-$n pbs/eval_grouped_retrieval.pbs)
        log "submitted $s scoring $n: $id"
    done < "$DS"
}

while (( NSCORED < ${#ORDER[@]} )); do
    for s in "${ORDER[@]}"; do
        [[ -n ${SCORED[$s]:-} ]] && continue
        alive "${JOB[$s]}" && continue
        if grep -q "=== done: 12/12 arms ok" "$(oulog "${JOB[$s]}")" 2>/dev/null; then
            log "$s training ${JOB[$s]} ok"; score_scale "$s" "${JOB[$s]}"; SCORED[$s]=ok; NSCORED=$((NSCORED+1))
        else
            log "$s training ${JOB[$s]} NOT fully ok -- not scored; needs a decision (RESUME_JOB?)"
            SCORED[$s]=failed; NSCORED=$((NSCORED+1))
        fi
    done
    sleep 300
done
log "all training finished; scoring submitted where training succeeded"
