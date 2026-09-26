"""Retrieval with and without a precursor filter, on the same embeddings.

    python pbs/diag/mass_window_eval.py embed   --models FILE --data NAME=DIR ... --emb-dir DIR --shard I --num-shards N
    python pbs/diag/mass_window_eval.py analyse --models FILE --data NAME=DIR ... --emb-dir DIR --out FILE

THE QUESTION. Contrastive training with same-mass batches (C19) teaches the encoder to separate
peptides of nearly the same mass, which is what a search engine asks of it after its precursor
filter. Every number so far is OPEN retrieval (whole gallery, no filter). Does C19's gain hold,
shrink or grow once candidates are restricted to the query's precursor window? And where do the
remaining open-retrieval errors come from: near-mass neighbours (a filter cannot remove them)
or far-mass ones (a filter removes them for free)?

Experimental spectra only (queries and gallery), as in the headline experimental/MAP@R.
Windows, all on THEORETICAL values computed from the peptide (the quantity C19 sorts on), so a
query's replicates are always inside its own window and R is unchanged:
    open          no filter (reproduces experimental/MAP@R)
    mass_1Da      |neutral mass difference| <= 1 Da, any charge
    mz_20ppm      same charge and |precursor m/z difference| <= 20 ppm (a standard library search)
Diagnostics on open retrieval: MAP@R by crowding (number of other-peptide gallery spectra within
1 Da of the query), and for Hit@1 failures the |mass difference| to the wrong top-1.
I/L-isobaric peptides are different groups here, as in every other evaluation.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

PROTON = 1.007276


def read_models(path):
    out = []
    for line in Path(path).read_text().splitlines():
        if line.strip() and not line.startswith("#"):
            p = line.split()
            out.append((p[0], p[1], p[2] if len(p) > 2 else "mean+max"))
    return out


def read_data(specs):
    return dict(s.split("=", 1) for s in specs)


def experimental_rows(path):
    from datasets import load_from_disk
    d = load_from_disk(path)
    keep = [i for i, s in enumerate(d["source"]) if s == "experimental"]
    return d.select(keep) if len(keep) < len(d) else d


def embed(cli) -> int:
    from msdelta.contrastive import MSDeltaForContrastive, embed_dataset
    from msdelta.finetune_contrastive import ContrastiveCollator
    from msdelta.modeling_msdelta import MSDeltaForPreTraining

    device = torch.device("xpu" if torch.xpu.is_available() else "cpu")
    collator = ContrastiveCollator(max_peptide_length=64, pad_spectra_to=512)
    jobs = [(m, d) for m in read_models(cli.models) for d in read_data(cli.data).items()]
    mine = jobs[cli.shard::cli.num_shards]
    print(f"[embed] shard {cli.shard}/{cli.num_shards}: {len(mine)} of {len(jobs)} on {device}", flush=True)
    cache = {}
    for (name, path, pooling), (dname, dpath) in mine:
        target = Path(cli.emb_dir) / dname / f"{name}.npy"
        if target.exists():
            print(f"  {dname}/{name}: cached", flush=True)
            continue
        if dname not in cache:
            cache[dname] = experimental_rows(dpath)
        rows = cache[dname]
        t0 = time.time()
        encoder = MSDeltaForPreTraining.from_pretrained(path)
        model = MSDeltaForContrastive(encoder, None, pooling=pooling, kl_weight=0).to(device)
        emb, _ = embed_dataset(model, rows, collator, device, max_rows=len(rows), batch_size=16)
        del model, encoder
        if device.type == "xpu":
            torch.xpu.empty_cache()
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_suffix(".tmp.npy")
        np.save(tmp, emb.float().numpy())
        tmp.rename(target)
        print(f"  {dname}/{name}: {tuple(emb.shape)} ({time.time() - t0:.0f}s)", flush=True)
    return 0


def windowed_metrics(e, g, allowed_fn, chunk=1024, k=100):
    """MAP@R / Hit@1 where each query only sees gallery items allowed_fn(q) marks True.
    Same arithmetic as contrastive.retrieval_metrics_topk; filtered-out and self entries are
    -inf and never count as hits."""
    n = len(e)
    n_rel = torch.bincount(g)[g] - 1
    ranks = torch.arange(1, k + 1, dtype=torch.float32, device=e.device).unsqueeze(0)
    mapr = hit1 = 0.0
    per_query_ap = torch.zeros(n, device=e.device)
    per_query_top = torch.zeros(n, dtype=torch.long, device=e.device)
    scorable = 0
    for s in range(0, n, chunk):
        q = torch.arange(s, min(s + chunk, n), device=e.device)
        sim = e[q] @ e.T
        allowed = allowed_fn(q)
        sim[~allowed] = float("-inf")
        sim[torch.arange(len(q), device=e.device), q] = float("-inf")
        val, idx = sim.topk(k, dim=1)
        hit = ((g[idx] == g[q].unsqueeze(1)) & torch.isfinite(val)).float()
        r = n_rel[q].float()
        keep = r > 0
        prec = hit.cumsum(1) / ranks
        within = ranks <= r.unsqueeze(1)
        ap = (prec * hit * within).sum(1) / r.clamp(min=1)
        per_query_ap[q] = torch.where(keep, ap, torch.nan)
        per_query_top[q] = idx[:, 0]
        mapr += float(ap[keep].sum())
        hit1 += float(hit[keep, 0].sum())
        scorable += int(keep.sum())
    return {"MAP@R": mapr / scorable, "Hit@1": hit1 / scorable, "queries": scorable}, per_query_ap, per_query_top


def analyse(cli) -> int:
    from msdelta.grouped_retrieval import group_ids
    from msdelta.reranking import peptide_neutral_mass

    device = torch.device("xpu" if torch.xpu.is_available() else "cpu")
    report = {"windows": ["open", "mass_1Da", "mz_20ppm"], "results": {}}
    for dname, dpath in read_data(cli.data).items():
        rows = experimental_rows(dpath)
        g = torch.as_tensor(group_ids(rows), device=device)
        charge = torch.as_tensor([int(c) for c in rows["charge"]], device=device)
        mass = torch.as_tensor([peptide_neutral_mass(p) for p in rows["peptide"]],
                               dtype=torch.float64, device=device)
        mz = (mass + charge * PROTON) / charge
        windows = {
            "open": lambda q: torch.ones(len(q), len(g), dtype=torch.bool, device=device),
            "mass_1Da": lambda q: (mass[q, None] - mass[None]).abs() <= 1.0,
            "mz_20ppm": lambda q: (charge[q, None] == charge[None])
                                  & ((mz[q, None] - mz[None]).abs() <= 20e-6 * mz[q, None]),
        }
        # crowding: other-group gallery spectra within 1 Da (model independent)
        crowd = torch.zeros(len(g), dtype=torch.long, device=device)
        cand = torch.zeros(len(g), dtype=torch.long, device=device)
        for s in range(0, len(g), 2048):
            q = torch.arange(s, min(s + 2048, len(g)), device=device)
            other = g[q, None] != g[None]
            crowd[q] = (windows["mass_1Da"](q) & other).sum(1)
            cand[q] = (windows["mz_20ppm"](q) & other).sum(1)
        bins = [(0, 0), (1, 5), (6, 20), (21, 10**9)]
        info = {"spectra": len(g), "groups": int(g.max()) + 1,
                "median_other_within_1Da": float(crowd.float().median()),
                "median_other_within_20ppm_same_charge": float(cand.float().median()),
                "fraction_queries_by_crowding": {f"{a}-{b}": float(((crowd >= a) & (crowd <= b)).float().mean())
                                                 for a, b in bins}}
        report["results"][dname] = {"data": dpath, "info": info, "models": {}}
        print(f"== {dname}: {info}", flush=True)
        for name, _, _ in read_models(cli.models):
            f = Path(cli.emb_dir) / dname / f"{name}.npy"
            if not f.exists():
                print(f"  {name}: no embeddings", flush=True)
                continue
            e = F.normalize(torch.from_numpy(np.load(f)).to(device), dim=-1)
            res = {}
            for w, fn in windows.items():
                m, ap, top = windowed_metrics(e, g, fn)
                res[w] = m
                if w == "open":
                    valid = ~torch.isnan(ap)
                    res["open_by_crowding"] = {
                        f"{a}-{b}": float(ap[valid & (crowd >= a) & (crowd <= b)].mean())
                        for a, b in bins}
                    miss = valid & (g[top] != g)
                    dm = (mass[top] - mass).abs()[miss]
                    res["open_top1_errors"] = {
                        "count": int(miss.sum()),
                        "fraction_within_1Da": float((dm <= 1.0).float().mean()) if miss.any() else 0.0,
                        "fraction_within_20ppm_same_charge": float(
                            ((dm / mass[miss] <= 20e-6) & (charge[top][miss] == charge[miss])).float().mean())
                            if miss.any() else 0.0}
            report["results"][dname]["models"][name] = res
            print(f"  {name:40s} " + "  ".join(f"{w} {res[w]['MAP@R']:.4f}" for w in windows)
                  + f"  | err<=1Da {res['open_top1_errors']['fraction_within_1Da']:.2f}", flush=True)
    Path(cli.out).parent.mkdir(parents=True, exist_ok=True)
    Path(cli.out).write_text(json.dumps(report, indent=1))
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["embed", "analyse"])
    ap.add_argument("--models", required=True)
    ap.add_argument("--data", nargs="+", required=True, help="NAME=DIR, prepared splits")
    ap.add_argument("--emb-dir", required=True)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--out", default="results/finetune/contrastive/mass-window/summary.json")
    cli = ap.parse_args()
    return embed(cli) if cli.cmd == "embed" else analyse(cli)


if __name__ == "__main__":
    raise SystemExit(main())
