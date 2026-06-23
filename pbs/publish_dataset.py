"""Publish the MS2 peptide-replicate-retrieval benchmark to a (private) HF repo.

Creates the dataset repo if needed and uploads the parquet shards + dataset card
from the local benchmark dir. Authentication comes from the HF_TOKEN env var or a
prior `huggingface-cli login` — this script never hard-codes a token.

    pip install huggingface_hub          # or: uv pip install huggingface_hub
    export HF_TOKEN=hf_xxx               # or: huggingface-cli login
    python pbs/publish_dataset.py --repo-id <username>/ms2-peptide-replicate-retrieval

Use --public to make it public (default is private).
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-id", required=True,
                    help="<username-or-org>/ms2-peptide-replicate-retrieval")
    ap.add_argument("--dir", type=Path,
                    default=Path("data/ms2-peptide-replicate-retrieval"))
    ap.add_argument("--public", action="store_true",
                    help="make the repo public (default: private)")
    ap.add_argument("--commit-message", default="Add MS2 peptide-replicate-retrieval benchmark")
    args = ap.parse_args()

    from huggingface_hub import HfApi  # imported here so --help works without the dep

    files = sorted(args.dir.glob("*.parquet"))
    if not files:
        raise SystemExit(f"no parquet files under {args.dir}")
    token = os.environ.get("HF_TOKEN")  # falls back to cached login if None

    api = HfApi(token=token)
    api.create_repo(args.repo_id, repo_type="dataset",
                    private=not args.public, exist_ok=True)
    print(f"repo ready: {args.repo_id} (private={not args.public})")
    print(f"uploading {len(files)} parquet shard(s) + README from {args.dir} ...")
    api.upload_folder(
        folder_path=str(args.dir),
        repo_id=args.repo_id,
        repo_type="dataset",
        commit_message=args.commit_message,
        allow_patterns=["*.parquet", "README.md"],
    )
    print(f"done → https://huggingface.co/datasets/{args.repo_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
