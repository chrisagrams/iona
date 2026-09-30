#!/bin/bash
# K155-C: write the scoring models file for the all-checkpoint job (its ID from the feeder state) and print
# its path. Called from feeder_plans/k155_allck.txt at submission time (the ID is unknown before).
cd "$(dirname "$0")/../.." || exit 1
S=/lus/flare/projects/UIC-HPC/khuss/msdelta
J=$(cat "$S/feeder/k155_allck/allck_train")
[[ $J =~ ^[0-9]+$ ]] || { echo "k155_models: no job ID in the feeder state" >&2; exit 1; }
F=sweeps/arms/score_allck.txt
{
    echo "# K155-C all-checkpoint finals (job $J)"
    while read -r arm; do echo "$arm $S/runs/sweep-$arm-$J/final"; done < sweeps/arms/allck_all.txt
} > "$F"
echo "$F"
