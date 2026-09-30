#!/bin/bash
# K142-C (approved 2026-09-30): if K66-C 400m (8878459) ends without "12/12 arms ok" (e.g. killed at its 14 h
# walltime), resubmit it with RESUME_JOB=8878459 (same configs + code, continues from each arm's last
# checkpoint, writes into the same run dirs) and, once that succeeds, score it via the feeder plan
# k66_400m_resume_score. If 8878459 succeeds on its own, the K66 pipeline watcher scores it and this exits.
#   setsid nohup pbs/tools/k66_400m_resume.sh >> $S/logs/k66_400m_resume.log 2>&1 < /dev/null &
cd "$(dirname "$0")/../.." || exit 1
S=/lus/flare/projects/UIC-HPC/khuss/msdelta
ORIG=8878459
log() { echo "$(date -Is) $*"; }
until qstat -xf $ORIG 2>/dev/null | grep -q "job_state = F"; do sleep 120; done
if grep -q "=== done: 12/12 arms ok" pbs/logs/$ORIG.*.OU 2>/dev/null; then
    log "$ORIG finished 12/12 ok -- no resume needed (K66 pipeline scores it)"; exit 0
fi
log "$ORIG ended without 12/12 ok ($(grep -h '=== done' pbs/logs/$ORIG.*.OU 2>/dev/null)) -- resubmitting with RESUME_JOB"
until out=$(qsub -q capacity -l select=1 -l walltime=08:00:00 -A UIC-HPC -l filesystems=home:flare -N hps400mR \
        -v REPO_DIR=$PWD,RESUME_JOB=$ORIG,SWEEP_ROOT=configs/sweep-hp-scale,SWEEP_MODULE=msdelta.finetune_contrastive,ARMS_FILE=sweeps/arms/hp_scale_400m.txt \
        pbs/aurora-finetune-sweep.pbs 2>&1); [[ $out =~ ^[0-9] ]]; do
    log "qsub not accepted yet: $out"; sleep 180
done
J=${out%%.*}; log "resume submitted: $J"
mkdir -p "$S/feeder/k66_400m_resume_score"
echo "$J" > "$S/feeder/k66_400m_resume_score/resumed"
exec pbs/tools/feeder.sh pbs/tools/feeder_plans/k66_400m_resume_score.txt
