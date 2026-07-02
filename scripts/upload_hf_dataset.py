"""Upload the shuffled consensus split to a Hugging Face dataset repo.

Optional archive/versioning step (training on Polaris reads the shuffled shards
locally from eagle). The shuffle writes shards flat (consensus_shuf_train_*.parquet
/ consensus_shuf_val_*.parquet) so local training can use split_paths; the HF
pipeline (msdelta.data.hf_split_paths) instead expects train/*.parquet and
val/*.parquet SUBDIRS. We bridge that by hard-linking the flat shards into a
train/ + val/ staging tree (free — same eagle filesystem, no data copy) and
uploading that with the large-folder uploader.

Needs a WRITE token: `hf auth login` (or export HF_TOKEN=...) with write scope.

  python scripts/upload_hf_dataset.py \\
      --local-dir /eagle/.../consensus_100M_shuffled \\
      --repo-id   chrisagrams/consensus_100M_shuffled
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--local-dir", required=True, type=Path,
                    help="dir holding the flat <prefix>_train_*/<prefix>_val_* shards")
    ap.add_argument("--repo-id", default="chrisagrams/consensus_100M_shuffled")
    ap.add_argument("--prefix", default="consensus_shuf")
    ap.add_argument("--private", action="store_true", help="create the repo private")
    ap.add_argument("--stage-dir", type=Path, default=None,
                    help="where to build the train/ + val/ hardlink tree "
                         "(default: <local-dir>/_hf_staging)")
    args = ap.parse_args(argv)

    from huggingface_hub import HfApi

    # Pick up a login token even if HF_HOME was relocated (as in pbs/download_hf_dataset.sh).
    if not os.environ.get("HF_TOKEN"):
        tok = Path.home() / ".cache/huggingface/token"
        if tok.is_file():
            os.environ["HF_TOKEN"] = tok.read_text().strip()

    stage = args.stage_dir or (args.local_dir / "_hf_staging")
    for split in ("train", "val"):
        (stage / split).mkdir(parents=True, exist_ok=True)
        shards = sorted(args.local_dir.glob(f"{args.prefix}_{split}_*.parquet"))
        if not shards:
            raise SystemExit(f"no {split} shards ({args.prefix}_{split}_*.parquet) in {args.local_dir}")
        for s in shards:
            link = stage / split / s.name
            if not link.exists():
                os.link(s, link)  # hardlink: same fs, zero copy
        print(f"[stage] {split}: {len(shards)} shards -> {stage/split}", flush=True)

    api = HfApi()
    api.create_repo(args.repo_id, repo_type="dataset", private=args.private, exist_ok=True)
    print(f"[upload] {stage} -> {args.repo_id} (dataset)", flush=True)
    api.upload_large_folder(
        repo_id=args.repo_id,
        repo_type="dataset",
        folder_path=str(stage),
    )
    print(f"[upload] done. use with data.hf_repo={args.repo_id} "
          f"(hf_train_split=train, hf_val_split=val)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
