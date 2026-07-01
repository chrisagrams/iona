#!/bin/bash
# One-time Hugging Face pre-fetch for the v14_cap_XL_hf run. Login-node safe:
# pure download, no GPU work. Run this ONCE before qsub-ing pbs/v14_xl_hf_ddp.pbs.
#
# Why pre-download: the training job runs on a compute node with 4 DDP ranks.
# Fetching ~28.5M spectra mid-job would burn walltime and race across ranks, and
# the compute node only reaches the hub through the ALCF proxy. Pulling the data
# here (onto eagle, not home — home has a tight quota) lets the job run with
# HF_HUB_OFFLINE=1 and just read the cache.
#
# HF_HOME points the whole HF cache (hub snapshots + datasets arrow cache) at
# eagle, matching the /eagle/UIC-HPC/cgrams/... convention of the other scripts.
# The training PBS exports the SAME HF_HOME so snapshot_download / load_dataset
# resolve straight to what we fetch here.

set -euo pipefail

REPO=/home/cgrams/msdelta
VENV=$REPO/.venv

# Shared HF cache on eagle (survives across jobs; keep it off home).
export HF_HOME=/eagle/UIC-HPC/cgrams/hf_cache
mkdir -p "$HF_HOME"

# Credentials stay at the standard `hf auth login` location (~/.cache/huggingface),
# NOT on shared eagle. Relocating HF_HOME above also moves HF's token lookup, so
# surface the login token via HF_TOKEN when the private probe repo needs it.
# (The training job runs HF_HUB_OFFLINE=1 and needs no token — the cache suffices.)
if [[ -z "${HF_TOKEN:-}" && -f "$HOME/.cache/huggingface/token" ]]; then
    export HF_TOKEN="$(cat "$HOME/.cache/huggingface/token")"
fi

# Repos to fetch — must match configs/v14_cap_XL_hf.yaml.
TRAIN_REPO=chrisagrams/massive_kb_v1_shuffled                 # data.hf_repo
PROBE_REPO=chrisagrams/ms2-peptide-replicate-retrieval        # log.replicate_retrieval_repo

cd "$REPO"
echo "$(date -Is) HF_HOME=$HF_HOME"
echo "$(date -Is) fetching $TRAIN_REPO (train/*.parquet, val/*.parquet) + $PROBE_REPO ..."

# For private repos, `huggingface-cli login` once first (token lands in HF_HOME),
# or export HF_TOKEN before running this.
"$VENV/bin/python" - "$TRAIN_REPO" "$PROBE_REPO" <<'PY'
import sys
from huggingface_hub import snapshot_download
from datasets import load_dataset

train_repo, probe_repo = sys.argv[1:3]

# Training dataset: fetch ONLY the two splits the loader reads, using the exact
# allow_patterns from msdelta.data.hf_split_paths so the training job's
# snapshot_download hits this same cache entry.
local = snapshot_download(
    train_repo,
    repo_type="dataset",
    allow_patterns=["train/*.parquet", "val/*.parquet"],
)
print(f"[train] cached -> {local}", flush=True)

# Replicate-retrieval benchmark: build its Arrow cache so the inline probe can
# run under HF_HUB_OFFLINE=1 (default split is 'test' in replicate_retrieval.py).
# Non-fatal: it's a private, optional probe repo — a miss here shouldn't undo the
# train-data fetch above. Authenticate (hf auth login) and re-run to cache it,
# or drop log.replicate_retrieval_repo from the config to skip the probe.
try:
    ds = load_dataset(probe_repo, split="test")
    print(f"[probe] cached benchmark: {len(ds)} spectra", flush=True)
except Exception as e:
    print(f"[probe] WARNING: could not fetch {probe_repo}: {type(e).__name__}: {e}",
          file=sys.stderr, flush=True)
    print("[probe] train data is cached; fix auth and re-run, or remove "
          "replicate_retrieval_repo from the config to skip.", file=sys.stderr, flush=True)
PY

echo "$(date -Is) done. cache tree:"
ls -1 "$HF_HOME/hub" 2>/dev/null | sed 's/^/  /' || true
