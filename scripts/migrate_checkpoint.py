"""Convert one MSDelta checkpoint directory into an Iona checkpoint.

Usage:
    uv run python scripts/migrate_checkpoint.py OLD_CHECKPOINT_DIR NEW_CHECKPOINT_DIR

Only the model weights, config, and processor are migrated. Trainer/DeepSpeed resume state
(optimizer, scheduler, global_step*/) is not carried over.
"""

import argparse
import json
from pathlib import Path

from iona import IonaForDenoising, IonaForPreTraining, IonaForRetrieval, IonaProcessor

MODEL_CLASSES = {
    "msdelta": IonaForPreTraining,
    "msdelta-denoising": IonaForDenoising,
    "msdelta-retrieval": IonaForRetrieval,
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert one MSDelta checkpoint to Iona.")
    parser.add_argument("old_dir", type=Path)
    parser.add_argument("new_dir", type=Path)
    args = parser.parse_args()

    if args.new_dir.exists():
        parser.error(f"{args.new_dir} already exists")
    model_type = json.loads((args.old_dir / "config.json").read_text())["model_type"]
    if model_type not in MODEL_CLASSES:
        parser.error(f"unexpected model_type {model_type!r}; is this an MSDelta checkpoint?")

    model = MODEL_CLASSES[model_type].from_pretrained(
        args.old_dir, key_mapping={r"^msdelta\.": "iona."}
    )
    if hasattr(model.config, "encoder"):
        # Drop stale MSDelta class references copied into the nested encoder config.
        model.config.encoder.architectures = None
        model.config.encoder.__dict__.pop("auto_map", None)
    # save_original_format=False keeps the renamed keys instead of reverting to msdelta.*
    model.save_pretrained(args.new_dir, save_original_format=False)
    if (args.old_dir / "preprocessor_config.json").exists():
        IonaProcessor.from_pretrained(args.old_dir).save_pretrained(args.new_dir)
    print(f"migrated {model_type} checkpoint -> {args.new_dir}")


if __name__ == "__main__":
    main()
