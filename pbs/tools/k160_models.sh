#!/bin/bash
# K160-C: write the scoring models file for the consensus twins (training job ID from the feeder state) plus the
# no-consensus K155-C arms that finished before the pause (job 8880712, not yet scored), and print its path.
# Called from feeder_plans/k160_cons.txt at submission time.
cd "$(dirname "$0")/../.." || exit 1
S=/lus/flare/projects/UIC-HPC/khuss/msdelta
J=$(cat "$S/feeder/k160_cons/cons_train")
[[ $J =~ ^[0-9]+$ ]] || { echo "k160_models: no job ID in the feeder state" >&2; exit 1; }
F=sweeps/arms/score_cons.txt
{
    echo "# K160-C consensus twins (job $J) + finished K155-C no-consensus arms (job 8880712)"
    while read -r arm; do echo "$arm $S/runs/sweep-$arm-$J/final"; done < sweeps/arms/cons_all.txt
    for ck in 220k 330k 430k; do for seed in 0 1 2; do
        arm=s100m_ck${ck}_lr4e-4_p170k2_seed$seed; echo "$arm $S/runs/sweep-$arm-8880712/final"
    done; done
} > "$F"
echo "$F"
