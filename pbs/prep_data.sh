#!/bin/bash
# ONE-TIME login-node prep: make sure the training dataset is fully downloaded.
#
# Downloading ~80 GB inside a 1-hour debug job would eat the whole walltime, so do it
# here instead. Preprocessing is deliberately NOT done here: measured in-job at 32 s for
# 1,000,234 rows with 24 workers, so it is cheaper to let each job redo it than to put
# that load on a shared login node.
#
#   bash pbs/prep_data.sh          # safe to re-run; resumes a partial download
set -euo pipefail

REPO_DIR=${REPO_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
cd "$REPO_DIR"

export HF_HOME=${HF_HOME:-/eagle/UIC-HPC/$USER/msdelta/huggingface}
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
ALCF_PROXY=${ALCF_PROXY:-http://proxy.alcf.anl.gov:3128}
export HTTP_PROXY=${HTTP_PROXY:-$ALCF_PROXY} HTTPS_PROXY=${HTTPS_PROXY:-$ALCF_PROXY}
export http_proxy=${http_proxy:-$ALCF_PROXY} https_proxy=${https_proxy:-$ALCF_PROXY}

REPO_ID=${REPO_ID:-chrisagrams/massive_kb_v1_shuffled}
SENTINEL=$HF_HOME/.msdelta-prep-done

mkdir -p "$HF_HOME"
echo "HF_HOME=$HF_HOME  repo=$REPO_ID"

"$REPO_DIR/.venv/bin/python" - "$REPO_ID" <<'PY'
import sys

from msdelta.data.loading import hf_split_paths

repo_id = sys.argv[1]
print("downloading shards (resumes if partial)...", flush=True)
train_paths, val_paths = hf_split_paths(repo_id)
bad = [p for p in train_paths + val_paths if not p.exists() or p.stat().st_size == 0]
if bad:
    raise SystemExit(f"ERROR: {len(bad)} shard(s) missing or empty after download")
print(f"{len(train_paths)} train shards, {len(val_paths)} val shards, all present", flush=True)
PY

touch "$SENTINEL"
echo "prep complete -- sentinel: $SENTINEL"
echo "now start the chain:  nohup bash pbs/chain_debug.sh > pbs/logs/chain.log 2>&1 &"
