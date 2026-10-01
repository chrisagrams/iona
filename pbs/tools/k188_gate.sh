#!/bin/bash
# K188-C: start the consensus all-checkpoint feeder only after (1) its debug smoke exited 0 and (2) every training job of
# the K185/K186 (k185_p2u) and K182 (d) (k182d_tbase) plans has been submitted and has left the queue (user: finish the
# Pairformer experiments first). Detached:  setsid nohup pbs/tools/k188_gate.sh <smoke job id> >> $S/logs/k188_gate.log 2>&1 &
cd "$(dirname "$0")/../.." || exit 1
S=/lus/flare/projects/UIC-HPC/khuss/msdelta
SMOKE=$1
log() { echo "$(date -Is) $*"; }
finished() { qstat -xf "$1" 2>/dev/null | grep -q "job_state = F"; }
until finished "$SMOKE"; do sleep 300; done
st=$(qstat -xf "$SMOKE" | awk '/Exit_status =/{print $3}')
[[ $st == 0 ]] || { log "smoke $SMOKE exited $st: NOT starting K188"; echo "$(date -Is) k188_gate: smoke $SMOKE exited $st" >> "$S/notifications/feeder.txt"; exit 1; }
log "smoke $SMOKE ok"
while true; do
    busy=0
    for plan in k185_p2u:5 k182d_tbase:3; do
        d=$S/feeder/${plan%%:*}; n=$(ls "$d" 2>/dev/null | grep -c '^t_')
        (( n < ${plan##*:} )) && { busy=1; continue; }
        for f in "$d"/t_*; do j=$(cat "$f"); [[ $j == BLOCKED ]] || finished "$j" || busy=1; done
    done
    (( busy )) || break
    sleep 300
done
log "K185/K186 and K182 (d) trainings done: starting the K188 feeder"
exec pbs/tools/feeder.sh pbs/tools/feeder_plans/k188_cons_allck.txt
