"""Zero-shot retrieval from FROZEN pretrained encoders, at every depth, on the 100k test.

    python -m msdelta.eval.eval_zeroshot_layers --models FILE --out-dir DIR [--shard I --num-shards N]

Redoes the "frozen embedding is no better than random at any layer" finding (OBSERVATIONS,
"The pretrained encoder learns a real representation, but a NON-LINEAR one") with the
metric and eval that replaced its two weaknesses: it was scored on the separation ratio,
which C0 showed does not predict retrieval, over the 99-group replicate eval. Here:
MAP@R / Hit@1 on ms-contrastive-100k's test split (the rows eval_grouped_retrieval
prepared), mean+max pooling of every block's output plus the final normalised state
(`final`, the readout every other zero-shot number uses). No training, no head.

--models: one `name path` per line, path = a pretrained MSDeltaForPreTraining checkpoint.

--abtt 1,2,4,8,16,32 --fit-data TRAIN_DIR additionally scores every layer after
"all-but-the-top" post-processing (Mu & Viswanath, ICLR 2018, arXiv:1702.01417): subtract
the mean, then remove the D leading principal directions, which in anisotropic embeddings
carry common, non-discriminative variance that dominates cosine similarity. Applied to the
RAW pooled vectors (as in the paper), then cosine as usual. Two fits, reported separately:
  train  mean + directions from the experimental spectra of a prepared TRAIN sample (the
         headline: nothing is estimated on the spectra being scored)
  test   fitted on the scored test spectra themselves, as the paper does (unsupervised
         and label-free, but transductive; a check that the train fit loses nothing)
`center` (D = 0 with the mean removed) is always included: centering alone changes cosine.
Results go under "abtt" -> fit -> "center" | "D" -> layer, beside the unchanged "layers".
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


def all_but_top(embeddings: np.ndarray, components: int) -> np.ndarray:
    """Center embeddings and remove their leading principal directions."""
    if components <= 0:
        return embeddings
    limit = min(embeddings.shape)
    if components >= limit:
        raise ValueError(
            f"all-but-top components ({components}) must be smaller than "
            f"min(n_spectra, embedding_size) ({limit})"
        )
    centered = embeddings - embeddings.mean(axis=0, keepdims=True)
    _, _, directions = np.linalg.svd(centered, full_matrices=False)
    top = directions[:components]
    return centered - (centered @ top.T) @ top


def fit_abtt(fit: torch.Tensor, max_components: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Mean and the `max_components` leading principal directions of `fit` (rows = samples).
    One decomposition serves every D <= max_components: the directions are nested."""
    if max_components >= min(fit.shape):
        raise ValueError(f"all-but-top components ({max_components}) must be smaller than "
                         f"min(n_spectra, embedding_size) ({min(fit.shape)})")
    device = fit.device
    fit = fit.double().cpu()                      # float64 on CPU: a stable, exact decomposition
    mean = fit.mean(dim=0, keepdim=True)
    centered = fit - mean
    _, vectors = torch.linalg.eigh(centered.T @ centered)   # ascending eigenvalues
    top = vectors[:, -max_components:].flip(-1).T          # = SVD's leading right vectors
    return mean.float().to(device), top.float().contiguous().to(device)


def apply_abtt(x: torch.Tensor, mean: torch.Tensor, directions: torch.Tensor,
               components: int) -> torch.Tensor:
    """all_but_top with a given fit: x - mean, minus its projection on the top directions."""
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

    from datasets import load_from_disk

    from msdelta.finetuning.contrastive.contrastive import encoder_layer_states, retrieval_metrics_topk
    from msdelta.finetuning.contrastive.finetune_contrastive import ContrastiveCollator
    from msdelta.data.grouped_retrieval import group_ids
    from msdelta.models.loading import load_strict
    from msdelta.models.modeling_msdelta import MSDeltaForPreTraining
    from msdelta.rescoring.reranking import pool_sequence

    device = torch.device("xpu" if torch.xpu.is_available() else "cpu")
    models = [l.split() for l in Path(cli.models).read_text().splitlines()
              if l.strip() and not l.startswith("#")][cli.shard::cli.num_shards]
    out_dir = Path(cli.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    rows = load_from_disk(cli.data)
    groups = group_ids(rows)
    experimental = np.array([s == "experimental" for s in rows["source"]])
    collator = ContrastiveCollator(max_peptide_length=64, pad_spectra_to=512)
    features = list(rows)
    fit_features = []
    if abtt:
        fit_rows = load_from_disk(cli.fit_data)
        fit_features = [f for f in fit_rows if f["source"] == "experimental"]
        print(f"abtt D={abtt}: fit on {len(fit_features):,} experimental train spectra",
              flush=True)

    for name, path in models:
        target = out_dir / f"{name}.json"
        if target.exists():
            print(f"  {name}: already scored", flush=True); continue
        t0 = time.time()
        model = load_strict(MSDeltaForPreTraining, path).to(device).eval()
        encoder = getattr(model, "msdelta", model)
        def embed(feats, raw):
            """Pooled vectors per layer: unit-norm halves, or raw float32 for abtt."""
            per_layer: dict[str, list] = {}
            with torch.no_grad():
                for start in range(0, len(feats), cli.batch_size):
                    batch = collator(feats[start:start + cli.batch_size])
                    mz, li, mask = (batch[k].to(device) for k in
                                    ("mz", "log_intensity", "attention_mask"))
                    states, final = encoder_layer_states(encoder, mz, li, mask)
                    for key, state in [*((f"block{i:02d}", s) for i, s in enumerate(states)),
                                       ("final", final)]:
                        pooled = pool_sequence(state, mask, cli.pooling).float()
                        pooled = pooled.cpu() if raw else F.normalize(pooled, dim=-1).half().cpu()
                        per_layer.setdefault(key, []).append(pooled)
            return {k: torch.cat(v).float() for k, v in per_layer.items()}

        per_layer = embed(features, raw=bool(abtt))
        fit_layer = embed(fit_features, raw=True) if abtt else {}
        t_embed = time.time() - t0
        del model
        if device.type == "xpu":
            torch.xpu.empty_cache()
        result = {"name": name, "path": path, "pooling": cli.pooling, "layers": {}}
        mask = torch.from_numpy(experimental)
        g = groups[experimental]

        def score(emb):
            return {f"experimental/{k}": v for k, v in
                    retrieval_metrics_topk(emb, g, device=device).items()}

        for key, emb in per_layer.items():
            result["layers"][key] = score(emb[mask])   # retrieval L2-normalises: raw is fine
        if abtt:
            result["abtt"] = {"components": abtt, "fit_spectra": {"train": len(fit_features),
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
