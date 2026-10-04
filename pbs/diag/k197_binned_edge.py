"""K197-C: what edge does binned cosine have over our spectrum encoders? Per-query diagnostic, no training.

    python pbs/diag/k197_binned_edge.py --data DIR --set NAME --out DIR \
        --model 400m=/path/to/final --model 25m=/path/to/final [--max-rows N]

Context: binned cosine 0.1 Da beats the consensus-recipe encoders in open search on oodval / mouse / yeast
(nine-species) but not on ms-contrastive-100k test or human. Binned 1 Da scores about the same as 0.1 Da on mouse
and yeast, so the edge is not fine m/z precision. This scores every experimental query with each method (binned
0.1 Da and each encoder) and saves per-query numbers, so the queries one method wins and the other loses can be
characterised offline (group size, charge, peak count, what the wrong top hit is):

  <out>/<set>_meta.npz          per experimental row: group, peptide, charge, precursor, n_peaks
  <out>/<set>_<method>.npz      per row: n_rel (R), ap (AP@R), hit1, first (rank of first correct, k+1 if none in
                                top k), top (top-5 neighbour rows), top_cos, pos_max / pos_mean (cosine to the
                                row's true replicates), neg_max (best wrong cosine)
  <out>/<set>_summary.json      mean MAP@R / Hit@1 per method (must match the scored JSONs), and late fusion
                                alpha * encoder cosine + (1 - alpha) * binned cosine per alpha: if a mix beats
                                both, the two carry complementary information.

Rows and groups are exactly the evaluation's (msdelta.eval.eval_grouped_retrieval, experimental variant: queries
and gallery are the experimental rows; peptide+charge groups; a query's own row excluded).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

K = 100
ALPHAS = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)


def encode(path, rows, device, batch_size):
    from msdelta.finetuning.contrastive.contrastive import MSDeltaForContrastive, embed_dataset
    from msdelta.finetuning.contrastive.finetune_contrastive import ContrastiveCollator
    from msdelta.models.loading import load_strict
    from msdelta.models.modeling_msdelta import MSDeltaForPreTraining

    encoder = load_strict(MSDeltaForPreTraining, path)
    head_file = Path(path) / "projection_head.pt"
    head = torch.load(head_file, map_location="cpu") if head_file.exists() else None
    model = MSDeltaForContrastive(encoder, None, pooling="mean+max", kl_weight=0,
                                  projection_hidden=head["projection_hidden"] if head else 0,
                                  projection_dim=head["projection_dim"] if head else 0,
                                  projection_dropout=head["projection_dropout"] if head else 0.1)
    if head:
        model.projection.load_state_dict(head["state_dict"])
    model.readout = "head"
    emb, _ = embed_dataset(model.to(device), rows, ContrastiveCollator(max_peptide_length=64, pad_spectra_to=512),
                           device, max_rows=len(rows), batch_size=batch_size)
    del model, encoder
    return F.normalize(emb.float(), dim=-1).to(device)


def per_query(sims, g, chunk=1024):
    """sims(q) -> (len(q), n) similarity rows. Per-query AP@R etc., the retrieval_metrics_topk definitions."""
    n = len(g)
    n_rel = torch.bincount(g)[g] - 1
    k = min(K, n - 1)
    assert int(n_rel.max()) <= k
    ranks = torch.arange(1, k + 1, dtype=torch.float32, device=g.device).unsqueeze(0)
    out = {x: [] for x in ("ap", "hit1", "first", "top", "top_cos", "pos_max", "pos_mean", "neg_max")}
    for start in range(0, n, chunk):
        q = torch.arange(start, min(start + chunk, n), device=g.device)
        sim = sims(q)
        sim[torch.arange(len(q), device=g.device), q] = float("-inf")
        same = g.unsqueeze(0) == g[q].unsqueeze(1)
        same[torch.arange(len(q), device=g.device), q] = False
        val, idx = sim.topk(k, dim=1)
        hit = (g[idx] == g[q].unsqueeze(1)).float()
        r = n_rel[q].float()
        prec = hit.cumsum(1) / ranks
        ap = (prec * hit * (ranks <= r.unsqueeze(1))).sum(1) / r.clamp_min(1)
        first = torch.where(hit.any(1), hit.argmax(1) + 1, torch.full_like(r, k + 1, dtype=torch.long))
        pos = sim.masked_fill(~same, float("nan"))
        out["ap"].append(ap); out["hit1"].append(hit[:, 0]); out["first"].append(first)
        out["top"].append(idx[:, :5]); out["top_cos"].append(val[:, :5])
        out["pos_max"].append(torch.nan_to_num(pos, nan=-2.0).max(1).values)
        out["pos_mean"].append(torch.nanmean(pos, dim=1))
        notself = torch.ones_like(same); notself[torch.arange(len(q), device=g.device), q] = False
        out["neg_max"].append(sim.masked_fill(same | ~notself, float("-inf")).max(1).values)
    res = {x: torch.cat(v).cpu().numpy() for x, v in out.items()}
    res["n_rel"] = n_rel.cpu().numpy()
    return res


def summary(res):
    keep = res["n_rel"] > 0
    return {"MAP@R": float(res["ap"][keep].mean()), "Hit@1": float(res["hit1"][keep].mean()), "queries": int(keep.sum())}


def main(argv=None) -> int:
    from datasets import load_from_disk

    from msdelta.data.grouped_retrieval import group_ids
    from msdelta.eval.eval_grouped_retrieval import binned_embeddings

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True)
    ap.add_argument("--set", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", action="append", default=[], help="name=path (repeatable)")
    ap.add_argument("--max-rows", type=int, default=0, help="first N rows only (smoke test)")
    ap.add_argument("--batch-size", type=int, default=16)
    cli = ap.parse_args(argv)
    device = torch.device("xpu" if torch.xpu.is_available() else "cpu")
    out = Path(cli.out); out.mkdir(parents=True, exist_ok=True)

    rows = load_from_disk(cli.data)
    if cli.max_rows:
        rows = rows.select(range(min(cli.max_rows, len(rows))))
    exp = np.array([s == "experimental" for s in rows["source"]])
    rows = rows.select(np.flatnonzero(exp))
    g = torch.as_tensor(group_ids(rows), dtype=torch.long, device=device)
    np.savez(out / f"{cli.set}_meta.npz", group=g.cpu().numpy(), peptide=np.array(rows["peptide"]),
             charge=np.asarray(rows["charge"]), precursor=np.asarray(rows["precursor"], dtype=np.float64),
             n_peaks=np.array([len(m) for m in rows["mz"]]))
    print(f"[k197] {cli.set}: {len(rows):,} experimental rows, {int(g.max()) + 1:,} groups on {device}", flush=True)

    t0 = time.time()
    emb = {"binned0.1": F.normalize(binned_embeddings(rows, 0.1), dim=-1).to(device)}
    for spec in cli.model:
        name, path = spec.split("=", 1)
        emb[name] = encode(path, rows, device, cli.batch_size)
        print(f"[k197] embedded {name} ({time.time() - t0:.0f}s)", flush=True)

    summ = {"set": cli.set, "data": cli.data, "rows": len(rows), "methods": {}, "fusion": {}}
    for name, e in emb.items():
        res = per_query(lambda q, e=e: e[q] @ e.T, g)
        np.savez(out / f"{cli.set}_{name}.npz", **res)
        summ["methods"][name] = summary(res)
        print(f"[k197] {name}: {summ['methods'][name]}", flush=True)
    b = emb["binned0.1"]
    for name in [n for n in emb if n != "binned0.1"]:
        e = emb[name]
        summ["fusion"][name] = {}
        for a in ALPHAS:
            res = per_query(lambda q, a=a: a * (e[q] @ e.T) + (1 - a) * (b[q] @ b.T), g)
            summ["fusion"][name][f"{a:.1f}"] = summary(res)
            if a == 0.5:
                np.savez(out / f"{cli.set}_fuse0.5_{name}.npz", **res)
        print(f"[k197] fusion {name}: " + "  ".join(f"{a}:{v['MAP@R']:.4f}" for a, v in summ['fusion'][name].items()),
              flush=True)
    (out / f"{cli.set}_summary.json").write_text(json.dumps(summ, indent=1))
    print(f"[k197] done ({time.time() - t0:.0f}s)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
