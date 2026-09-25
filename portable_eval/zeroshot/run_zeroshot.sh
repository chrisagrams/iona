#!/usr/bin/env bash
# Zero-shot (frozen-encoder) retrieval for every checkpoint in checkpoints.tsv, on CPU.
# Per encoder: MAP@R of every layer on the ms-contrastive-100k test set, raw and after
# all-but-the-top (ABTT, D in {8,32,64,128}) fitted on a separate TRAIN sample.
#
#   CKPT_ROOT=/data/pretrained DATA=/data/ms-contrastive-100k-test-mp512 \
#   FIT=/data/ms-contrastive-100k-train10k-mp512 OUT=results/zeroshot \
#   JOBS=4 bash portable_eval/zeroshot/run_zeroshot.sh
#
# Run from the msdelta repo root (it imports msdelta). One JSON per encoder is written to
# OUT and skipped when present, so the script can be stopped and restarted.
# JOBS encoders run at once; each gets THREADS torch threads (default: cores / JOBS).
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
: "${CKPT_ROOT:?set CKPT_ROOT (dir holding the msdelta-*-production-01-checkpoint-* folders)}"
: "${DATA:?set DATA (prepared ms-contrastive-100k-test-mp512)}"
: "${FIT:?set FIT (prepared ms-contrastive-100k-train10k-mp512)}"
OUT=${OUT:-results/zeroshot}
PY=${PY:-python}
JOBS=${JOBS:-4}
ABTT=${ABTT:-8:32:64:128}
THREADS=${THREADS:-$(( $(nproc) / JOBS ))}; (( THREADS < 1 )) && THREADS=1
mkdir -p "$OUT/lists"
export PYTHONPATH=$PWD${PYTHONPATH:+:$PYTHONPATH}

run_one() {   # name dir
  local name=$1 dir=$2
  [[ -f $OUT/$name.json ]] && { echo "skip $name (done)"; return 0; }
  [[ -f $CKPT_ROOT/$dir/model.safetensors ]] || { echo "MISSING $CKPT_ROOT/$dir"; return 1; }
  echo "$name $CKPT_ROOT/$dir" > "$OUT/lists/$name.txt"
  local t0=$(date +%s)
  OMP_NUM_THREADS=$THREADS MKL_NUM_THREADS=$THREADS $PY -m msdelta.eval_zeroshot_layers \
      --models "$OUT/lists/$name.txt" --data "$DATA" --out-dir "$OUT" --abtt "$ABTT" --fit-data "$FIT" \
      > "$OUT/$name.log" 2>&1 \
    && echo "done $name ($(( $(date +%s) - t0 ))s)" || echo "FAILED $name (see $OUT/$name.log)"
}
export -f run_one; export OUT CKPT_ROOT DATA FIT PY ABTT THREADS

echo "=== $(date -Is) zero-shot: $(grep -cv '^#' "$HERE/checkpoints.tsv") encoders, $JOBS at a time, $THREADS threads each"
# smallest first, so the first results arrive quickly
grep -v '^#' "$HERE/checkpoints.tsv" | sort -k1,1 | xargs -P "$JOBS" -L 1 bash -c 'run_one "$0" "$1"'
echo "=== $(date -Is) plotting"
$PY "$HERE/plot_zeroshot_scaling.py" "$OUT"
