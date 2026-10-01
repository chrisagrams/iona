#!/bin/bash
# K188-C: write the scoring models file for the all-checkpoint consensus twins (training job ID from the feeder
# state) and print its path.
# Called from feeder_plans/k188_cons_allck.txt at submission time.
cd "$(dirname "$0")/../.." || exit 1
S=/lus/flare/projects/UIC-HPC/khuss/msdelta
J=$(cat "$S/feeder/k188_cons_allck/cons_allck_train")
[[ $J =~ ^[0-9]+$ ]] || { echo "k188_models: no job ID in the feeder state" >&2; exit 1; }
F=sweeps/arms/score_cons_allck.txt
{
    echo "# K188-C consensus twins of the 26 non-final checkpoints x 4 scales (job $J); no-consensus partners: K155 (allck-*)"
    while read -r arm; do echo "$arm $S/runs/sweep-$arm-$J/final"; done < sweeps/arms/cons_allck.txt
} > "$F"
echo "$F"
