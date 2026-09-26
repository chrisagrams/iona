#!/bin/bash
# One node's share of pbs/rerank_psm_full.pbs: each tile embeds and featurizes its runs.
# Existing outputs are skipped.
set -uo pipefail
RANK=${PMIX_RANK:-${PMI_RANK:-${PALS_RANKID:-0}}}
NNODES=${NNODES:?}; TILES=${TILES:-12}
mapfile -t RUNS < <(grep -v '^#' "$RUNS_FILE" | sed '/^$/d')
cd "$REPO_DIR"
for (( t = 0; t < TILES; t++ )); do
  (
    slot=$(( RANK * TILES + t )); stride=$(( NNODES * TILES ))
    for (( i = slot; i < ${#RUNS[@]}; i += stride )); do
      run=${RUNS[$i]}; name=$(basename "$run")
      rows="$BASE/rows/$name"; hand="$BASE/handfeat/$name"; log="$BASE/logs/$name.log"
      if [[ ! -f $rows ]]; then
        ZE_AFFINITY_MASK=$t "$PY" -m msdelta.rerank_psm_embed --run "$run" --encoder "$ENCODER" \
          --student "$STUDENT" --cache "$CACHE" --out "$rows.tmp" >> "$log" 2>&1 \
          && mv "$rows.tmp" "$rows" || { echo "[node $RANK tile $t] EMBED FAILED $run"; continue; }
      fi
      if [[ ! -f $hand ]]; then
        OMP_NUM_THREADS=1 "$PY" -m msdelta.rerank_psm_handfeat --run "$run" --out "$hand.tmp" >> "$log" 2>&1 \
          && mv "$hand.tmp" "$hand" || echo "[node $RANK tile $t] HANDFEAT FAILED $run"
      fi
      echo "[node $RANK tile $t] done $run"
    done
  ) &
done
wait
