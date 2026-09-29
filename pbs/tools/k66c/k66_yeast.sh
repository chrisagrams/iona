#!/bin/bash
# K66-C yeast scoring (K76-C canonical full yeast, K81-C capacity 3 h): after each scale's training succeeds.
set -uo pipefail
cd /home/khuss/code/msdelta
S=/lus/flare/projects/UIC-HPC/khuss/msdelta
Y=$S/baselines/nine_yeast/prepared
LOG=${K66_LOG:-/lus/flare/projects/UIC-HPC/khuss/msdelta/logs/k66_pipeline.log}
log() { echo "$(date -u +%FT%H:%MZ) [yeast] $*" >> "$LOG"; }
declare -A JOB=([400m]=${K66_JOB_400M:-8876824} [025m]=${K66_JOB_025M:-8876825})   # resubmitted 2026-09-29 after the Aurora update; 100m, 200m already scored
declare -A DONE=()
n=0
while (( n < ${#JOB[@]} )); do
  for s in "${!JOB[@]}"; do
    [[ -n ${DONE[$s]:-} ]] && continue
    qstat "${JOB[$s]}" >/dev/null 2>&1 && continue
    f=$(ls pbs/logs/${JOB[$s]}.*.OU 2>/dev/null | head -1)
    if grep -q "=== done: 12/12 arms ok" "$f" 2>/dev/null; then
      m=sweeps/arms/score_hp_scale_$s.txt
      until [[ -f $m ]]; do sleep 60; done          # written by the main pipeline
      until out=$(qsub -q capacity -l select=1 -l walltime=03:00:00 -N hs${s}yeast \
            -v MODELS=$m,DATA=$Y,OUT_DIR=results/raw/finetune/contrastive/hp-scale-$s-yeast pbs/eval_grouped_retrieval.pbs 2>&1) \
            && [[ $out == *aurora-pbs* ]]; do
        [[ $out == *"would exceed"* ]] || { log "$s qsub REJECTED: $out"; break; }
        sleep 300
      done
      log "submitted $s scoring yeast: ${out%%.*}"
    else
      log "$s training ${JOB[$s]} not fully ok -- yeast not scored"
    fi
    DONE[$s]=1; n=$((n+1))
  done
  sleep 300
done
