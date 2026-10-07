"""Zero-shot retrieval from FROZEN pretrained encoders, at every depth, on the 100k test.

    python scripts/eval_zeroshot_layers.py --models FILE --out-dir DIR [--shard I --num-shards N]

Mean+max pooling of every block's output plus the final state; no training, no head.
--models: one `name path` per line. --abtt D,... --fit-data TRAIN_DIR also scores each layer
after all-but-the-top post-processing (Mu & Viswanath, 2018), fitted on train and on test.
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_from_disk

from iona.contrastive import encoder_layer_states
from iona.data import group_ids
from iona.finetune.contrastive import ContrastiveCollator
from iona.inference import PredictionTrainer
from iona.modeling_iona import IonaForPreTraining, pool_sequence
from iona.retrieval import retrieval_metrics


def fit_abtt(fit: torch.Tensor, max_components: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Mean and the `max_components` leading principal directions of `fit` (rows = samples)."""
    if max_components >= min(fit.shape):
        raise ValueError(f"all-but-top components ({max_components}) must be smaller than "
                         f"min(n_spectra, embedding_size) ({min(fit.shape)})")
    device = fit.device
    fit = fit.double().cpu()
    mean = fit.mean(dim=0, keepdim=True)
    centered = fit - mean
    _, vectors = torch.linalg.eigh(centered.T @ centered)   # ascending eigenvalues
    top = vectors[:, -max_components:].flip(-1).T
    return mean.float().to(device), top.float().contiguous().to(device)


def apply_abtt(x: torch.Tensor, mean: torch.Tensor, directions: torch.Tensor,
               components: int) -> torch.Tensor:
    """All-but-the-top with a given fit: x - mean, minus its projection on the top directions."""
    centered = x - mean
    top = directions[:components]
    return centered - (centered @ top.T) @ top if components > 0 else centered


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
    ap.add_argument("--abtt", default="", help="list of D, e.g. 1,2,4,8,16,32 or 1:2:4")
    ap.add_argument("--fit-data", default="", help="prepared TRAIN sample for the abtt fit")
    cli = ap.parse_args(argv)
    abtt = [int(d) for d in re.split(r"[,:\s]+", cli.abtt) if d]
    if abtt and not cli.fit_data:
        ap.error("--abtt needs --fit-data (the train fit is the headline)")

    device = torch.device("xpu" if torch.xpu.is_available() else "cpu")
    models = [l.split() for l in Path(cli.models).read_text().splitlines()
              if l.strip() and not l.startswith("#")][cli.shard::cli.num_shards]
    out_dir = Path(cli.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    rows = load_from_disk(cli.data)
    groups = group_ids(rows)
    experimental = np.array([s == "experimental" for s in rows["source"]])
    collator = ContrastiveCollator(max_peptide_length=64, pad_spectra_to=512)
    fit_rows = None
    if abtt:
        fit_rows = load_from_disk(cli.fit_data)
        fit_rows = fit_rows.select(np.flatnonzero(np.array(fit_rows["source"]) == "experimental"))
        print(f"abtt D={abtt}: fit on {len(fit_rows):,} experimental train spectra",
              flush=True)

    for name, path in models:
        target = out_dir / f"{name}.json"
        if target.exists():
            print(f"  {name}: already scored", flush=True); continue
        t0 = time.time()
        model = IonaForPreTraining.from_pretrained(path).to(device).eval()
        encoder = getattr(model, "iona", model)
        def embed(rows, raw):
            """Pooled vectors per layer: unit-norm halves, or raw float32 for abtt."""
            def predict(enc, x):
                mask = x["attention_mask"]
                states, final = encoder_layer_states(enc, x["mz"], x["log_intensity"], mask)
                out = {}
                for key, state in [*((f"block{i:02d}", s) for i, s in enumerate(states)),
                                   ("final", final)]:
                    pooled = pool_sequence(state, mask, cli.pooling).float()
                    out[key] = pooled if raw else F.normalize(pooled, dim=-1).half()
                return out

            per_layer = PredictionTrainer(encoder, predict, data_collator=collator,
                                          batch_size=cli.batch_size,
                                          device=device).predict_sorted(rows)
            return {k: torch.from_numpy(v).float() for k, v in per_layer.items()}

        per_layer = embed(rows, raw=bool(abtt))
        fit_layer = embed(fit_rows, raw=True) if abtt else {}
        t_embed = time.time() - t0
        del model
        if device.type == "xpu":
            torch.xpu.empty_cache()
        result = {"name": name, "path": path, "pooling": cli.pooling, "layers": {}}
        mask = torch.from_numpy(experimental)
        g = groups[experimental]

        def score(emb):
            return {f"experimental/{k}": v for k, v in
                    retrieval_metrics(emb, g, device).items()}

        for key, emb in per_layer.items():
            result["layers"][key] = score(emb[mask])
        if abtt:
            result["abtt"] = {"components": abtt, "fit_spectra": {"train": len(fit_layer["final"]),
                              "test": int(mask.sum())}, "train": {}, "test": {}}
            for key, emb in per_layer.items():
                x = emb[mask].to(device)
                for fit_name, fit in (("train", fit_layer[key].to(device)), ("test", x)):
                    mean, dirs = fit_abtt(fit, max(abtt))
                    out = result["abtt"][fit_name]
                    out.setdefault("center", {})[key] = score(apply_abtt(x, mean, dirs, 0))
                    for d in abtt:
                        out.setdefault(str(d), {})[key] = score(apply_abtt(x, mean, dirs, d))
                print(f"    {key}: raw {result['layers'][key]['experimental/MAP@R']:.4f}  "
                      + "  ".join(f"D{d} {result['abtt']['train'][str(d)][key]['experimental/MAP@R']:.4f}"
                                  for d in abtt) + f"  ({time.time() - t0:.0f}s)", flush=True)
        result["seconds"] = {"embed": round(t_embed), "total": round(time.time() - t0)}
        target.write_text(json.dumps(result, indent=1))
        best = max(result["layers"], key=lambda k: result["layers"][k]["experimental/MAP@R"])
        print(f"  {name} ({time.time() - t0:.0f}s): final "
              f"{result['layers']['final']['experimental/MAP@R']:.4f}  best {best} "
              f"{result['layers'][best]['experimental/MAP@R']:.4f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
