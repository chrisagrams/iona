"""Convert one MSDelta or legacy peptide-alignment checkpoint directory into an Iona checkpoint.

Usage:
    uv run python scripts/migrate_checkpoint.py OLD_CHECKPOINT_DIR NEW_CHECKPOINT_DIR

A directory with only `model.safetensors` and `sequence_encoder.*` weights is a peptide
encoder saved before it was a PreTrainedModel; its config is rebuilt from the weight shapes.

Only the model weights, config, and processor are migrated. Trainer/DeepSpeed resume state
(optimizer, scheduler, global_step*/) is not carried over.
"""

import argparse
import json
import re
from pathlib import Path

from safetensors.torch import load_file

from iona import (
    IonaForDenoising,
    IonaForPreTraining,
    IonaForRetrieval,
    IonaPeptideConfig,
    IonaPeptideForAlignment,
    IonaProcessor,
)

MODEL_CLASSES = {
    "msdelta": IonaForPreTraining,
    "msdelta-denoising": IonaForDenoising,
    "msdelta-retrieval": IonaForRetrieval,
}


def migrate_peptide(old_dir: Path, new_dir: Path, num_attention_heads: int) -> None:
    """Wrap a bare peptide-alignment state dict in a config and save it with save_pretrained."""
    state = load_file(str(old_dir / "model.safetensors"))
    prefix = "sequence_encoder."
    hidden_size = state[f"{prefix}residue.weight"].shape[1]
    config = IonaPeptideConfig(
        embedding_size=state[f"{prefix}projection.3.weight"].shape[0],
        hidden_size=hidden_size,
        num_hidden_layers=len({m.group(1) for k in state
                               if (m := re.match(rf"{prefix}encoder\.layers\.(\d+)\.", k))}),
        num_attention_heads=num_attention_heads,
        max_position_embeddings=state[f"{prefix}position.weight"].shape[0],
        n_charges=state[f"{prefix}charge.weight"].shape[0],
        mod_n_freqs=state[f"{prefix}mod_features.freqs"].shape[0],
        pooling=("mean+max" if state[f"{prefix}projection.0.weight"].shape[1] == 2 * hidden_size
                 else "mean"),
    )
    model = IonaPeptideForAlignment(config)
    model.load_state_dict(state, strict=True)
    model.save_pretrained(new_dir)
    print(f"migrated peptide encoder checkpoint -> {new_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert one legacy checkpoint to Iona.")
    parser.add_argument("old_dir", type=Path)
    parser.add_argument("new_dir", type=Path)
    parser.add_argument("--num-attention-heads", type=int, default=8,
                        help="peptide encoder heads; not recoverable from the weights")
    args = parser.parse_args()

    if args.new_dir.exists():
        parser.error(f"{args.new_dir} already exists")
    if not (args.old_dir / "config.json").exists():
        if not (args.old_dir / "model.safetensors").exists():
            parser.error(f"{args.old_dir} has neither config.json nor model.safetensors")
        migrate_peptide(args.old_dir, args.new_dir, args.num_attention_heads)
        return
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
