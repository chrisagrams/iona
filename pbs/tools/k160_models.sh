#!/bin/bash
# K160-C: write the scoring models file for the final-checkpoint consensus twins (training job ID from the feeder
# state) and print its path.
# Called from feeder_plans/k160_cons.txt at submission time.
cd "$(dirname "$0")/../.." || exit 1
S=/lus/flare/projects/UIC-HPC/khuss/msdelta
J=$(cat "$S/feeder/k160_cons/cons_train")
[[ $J =~ ^[0-9]+$ ]] || { echo "k160_models: no job ID in the feeder state" >&2; exit 1; }
F=sweeps/arms/score_cons.txt
{
    echo "# K160-C consensus twins of the final checkpoints (job $J); their no-consensus partners are scored in hp-scale-*"
    while read -r arm; do echo "$arm $S/runs/sweep-$arm-$J/final"; done < sweeps/arms/cons_finals.txt
} > "$F"
echo "$F"
