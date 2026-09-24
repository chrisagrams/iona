"""Zero-shot retrieval from FROZEN pretrained encoders, at every depth, on the 100k test.

    python -m msdelta.eval_zeroshot_layers --models FILE --out-dir DIR [--shard I --num-shards N]

Redoes the "frozen embedding is no better than random at any layer" finding (OBSERVATIONS,
"The pretrained encoder learns a real representation, but a NON-LINEAR one") with the
metric and eval that replaced its two weaknesses: it was scored on the separation ratio,
which C0 showed does not predict retrieval, over the 99-group replicate eval. Here:
MAP@R / Hit@1 on ms-contrastive-100k's test split (the rows eval_grouped_retrieval
prepared), mean+max pooling of every block's output plus the final normalised state
(`final`, the readout every other zero-shot number uses). No training, no head.

--models: one `name path` per line, path = a pretrained MSDeltaForPreTraining checkpoint.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--pooling", default="mean+max")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    cli = ap.parse_args(argv)

    from datasets import load_from_disk

    from msdelta.contrastive import encoder_layer_states, retrieval_metrics_topk
    from msdelta.finetune_contrastive import ContrastiveCollator
    from msdelta.grouped_retrieval import group_ids
    from msdelta.modeling_msdelta import MSDeltaForPreTraining
    from msdelta.reranking import pool_sequence

    device = torch.device("xpu" if torch.xpu.is_available() else "cpu")
    models = [l.split() for l in Path(cli.models).read_text().splitlines()
              if l.strip() and not l.startswith("#")][cli.shard::cli.num_shards]
    out_dir = Path(cli.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    rows = load_from_disk(cli.data)
    groups = group_ids(rows)
    experimental = np.array([s == "experimental" for s in rows["source"]])
    collator = ContrastiveCollator(max_peptide_length=64, pad_spectra_to=512)
    features = list(rows)

    for name, path in models:
        target = out_dir / f"{name}.json"
        if target.exists():
            print(f"  {name}: already scored", flush=True); continue
        t0 = time.time()
        model = MSDeltaForPreTraining.from_pretrained(path).to(device).eval()
        encoder = getattr(model, "msdelta", model)
        per_layer: dict[str, list] = {}
        with torch.no_grad():
            for start in range(0, len(features), cli.batch_size):
                batch = collator(features[start:start + cli.batch_size])
                mz, li, mask = (batch[k].to(device) for k in
                                ("mz", "log_intensity", "attention_mask"))
                states, final = encoder_layer_states(encoder, mz, li, mask)
                for key, state in [*((f"block{i:02d}", s) for i, s in enumerate(states)),
                                   ("final", final)]:
                    pooled = F.normalize(pool_sequence(state, mask, cli.pooling).float(), dim=-1)
                    per_layer.setdefault(key, []).append(pooled.half().cpu())
        del model
        if device.type == "xpu":
            torch.xpu.empty_cache()
        result = {"name": name, "path": path, "pooling": cli.pooling, "layers": {}}
        mask = torch.from_numpy(experimental)
        for key, chunks in per_layer.items():
            emb = torch.cat(chunks).float()
            result["layers"][key] = {
                f"experimental/{k}": v for k, v in retrieval_metrics_topk(
                    emb[mask], groups[experimental], device=device).items()}
        target.write_text(json.dumps(result, indent=1))
        best = max(result["layers"], key=lambda k: result["layers"][k]["experimental/MAP@R"])
        print(f"  {name} ({time.time() - t0:.0f}s): final "
              f"{result['layers']['final']['experimental/MAP@R']:.4f}  best {best} "
              f"{result['layers'][best]['experimental/MAP@R']:.4f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
