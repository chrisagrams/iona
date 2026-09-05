#!/bin/bash
# Build a small local dataset for smoke runs, from shards already in the HF cache.
# Avoids re-downloading the ~80 GB Hub dataset just to run 200 steps.
set -euo pipefail
REPO_DIR=${REPO_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
HF_HOME=${HF_HOME:-/eagle/UIC-HPC/$USER/msdelta/huggingface}
N_SHARDS=${N_SHARDS:-3}

SNAP=$(find "$HF_HOME/hub" -type d -path "*massive_kb_v1_shuffled/snapshots/*/train" | head -1)
[[ -d $SNAP ]] || { echo "no downloaded train shards under $HF_HOME" >&2; exit 2; }

DEST=$REPO_DIR/data/smoke
rm -rf "$DEST"; mkdir -p "$DEST"
i=0
for f in $(ls "$SNAP"/*.parquet | sort | head -"$N_SHARDS"); do
    ln -sf "$(readlink -f "$f")" "$DEST/$(printf 'train_%03d.parquet' "$i")"
    i=$((i + 1))
done
echo "linked $i shards into $DEST  ($(du -shL "$DEST" | cut -f1))"
