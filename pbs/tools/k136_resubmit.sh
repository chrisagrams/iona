#!/bin/bash
# K140-S (2026-09-29): resubmit K136 (50m lr4e-4_p170k2, K66-C card) once the per-user queued limit allows,
# then point its scoring (feeder plan k136_score) at the new job. Run detached:
#   setsid nohup pbs/tools/k136_resubmit.sh >> $S/logs/feeder_k136.log 2>&1 < /dev/null &
cd "$(dirname "$0")/../.." || exit 1
S=/lus/flare/projects/UIC-HPC/khuss/msdelta
until out=$(qsub -q capacity -l select=1 -l walltime=10:00:00 -A UIC-HPC -l filesystems=home:flare -N hps050m \
        -v REPO_DIR=$PWD,SWEEP_ROOT=configs/sweep-hp-scale,SWEEP_MODULE=msdelta.finetune_contrastive,ARMS_FILE=sweeps/arms/hp_scale_050m.txt \
        pbs/aurora-finetune-sweep.pbs 2>&1); [[ $out =~ ^[0-9] ]]; do
    sleep 180
done
J=${out%%.*}
echo "$(date -Is) K136 resubmitted: $J"
{
    echo "# K136-C 50m lr4e-4_p170k2 finals (job $J)"
    for s in 0 1 2; do
        echo "s050m_ck540k_lr4e-4_p170k2_seed$s $S/runs/sweep-s050m_ck540k_lr4e-4_p170k2_seed$s-$J/final"
    done
} > sweeps/arms/score_hp_scale_050m.txt
mkdir -p "$S/feeder/k136_score"
echo "$J" > "$S/feeder/k136_score/k136_train"
exec pbs/tools/feeder.sh pbs/tools/feeder_plans/k136_score.txt
