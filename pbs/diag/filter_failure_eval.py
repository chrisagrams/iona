"""When the precursor filter is wrong: retrieval on the queries a 20 ppm filter fails.

    python pbs/diag/filter_failure_eval.py embed    --models FILE --data NAME=DIR ... --emb-dir DIR --shard I --num-shards N
    python pbs/diag/filter_failure_eval.py analyse  --models FILE --data NAME=DIR ... --emb-dir DIR --out FILE
    python pbs/diag/filter_failure_eval.py selftest   (toy CPU check of the metric code)

THE QUESTION. On real data the RECORDED precursor m/z is sometimes off by an isotope
(k x 1.00336 Da / z; notes/OBSERVATIONS.md "20 ppm ceilings are precursor isotope
mis-assignment"). A 20 ppm filter on the measured precursor then removes correct matches.
Conditioned on that happening, does open (unfiltered) embedding retrieval recover them, and
how much does filtering cost overall?

Conventions as pbs/diag/mass_window_eval.py (embed step reused verbatim): experimental spectra
only, groups = peptide+charge (grouped_retrieval.group_ids), MAP@R and Hit@1 at k=100, a
positive removed by a filter counts as a miss (R unchanged, entries -inf).
Filters, on the MEASURED `precursor` m/z of query and gallery spectrum, same charge:
    open        no filter
    20ppm       |dmz| <= 20 ppm
    iso20ppm    |dmz - k*1.00336/z| <= 20 ppm for some k in ISO_K = {-2..2} (k applies to
                the difference of two measured values, each of which may be offset)
Query subsets (per query q, positives P(q) = other spectra of its group):
    F      at least one positive outside the 20 ppm measured window
    F_all  every positive outside (the 20 ppm filter makes q unanswerable)
    Fbar   the rest (all positives inside)
Also: net loss = fraction of queries with open Hit@1 right but filtered Hit@1 wrong (and the
reverse, net gain); rescue rate = open Hit@1 on F_all.
Binned cosine (no model) is scored as models named binned_<width>.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mass_window_eval as mwe  # noqa: E402

ISOTOPE = 1.00336
ISO_K = (-2, -1, 0, 1, 2)
PPM = 20.0
FILTERS = ("open", "20ppm", "iso20ppm")
SUBSETS = ("full", "F", "Fbar", "F_all")


def make_filters(prec, charge, ppm=PPM, iso_k=ISO_K):
    """Allowed-gallery functions q -> bool[len(q), n] on measured precursor m/z."""
    n = len(prec)

    def open_(q):
        return torch.ones(len(q), n, dtype=torch.bool, device=prec.device)

    def ppm_(q):
        tol = ppm * 1e-6 * prec[q, None]
        return (charge[q, None] == charge[None]) & ((prec[q, None] - prec[None]).abs() <= tol)

    def iso_(q):
        tol = ppm * 1e-6 * prec[q, None]
        d = prec[q, None] - prec[None]
        step = ISOTOPE / charge[q, None].double()
        ok = torch.zeros(len(q), n, dtype=torch.bool, device=prec.device)
        for k in iso_k:
            ok |= (d - k * step).abs() <= tol
        return ok & (charge[q, None] == charge[None])

    return {"open": open_, "20ppm": ppm_, "iso20ppm": iso_}


def filtered_metrics(e, g, allowed_fn, chunk=1024, k=100):
    """Per-query AP@R (nan when R=0) and Hit@1 (0/1), same arithmetic as
    mass_window_eval.windowed_metrics; excluded and self entries are -inf, never hits."""
    n = len(e)
    k = min(k, n - 1)
    n_rel = torch.bincount(g)[g] - 1
    ranks = torch.arange(1, k + 1, dtype=torch.float32, device=e.device).unsqueeze(0)
    ap_q = torch.full((n,), float("nan"), device=e.device)
    hit_q = torch.zeros(n, device=e.device)
    for s in range(0, n, chunk):
        q = torch.arange(s, min(s + chunk, n), device=e.device)
        sim = e[q] @ e.T
        sim[~allowed_fn(q)] = float("-inf")
        sim[torch.arange(len(q), device=e.device), q] = float("-inf")
        val, idx = sim.topk(k, dim=1)
        hit = ((g[idx] == g[q].unsqueeze(1)) & torch.isfinite(val)).float()
        r = n_rel[q].float()
        prec = hit.cumsum(1) / ranks
        within = ranks <= r.unsqueeze(1)
        ap = (prec * hit * within).sum(1) / r.clamp(min=1)
        ap_q[q] = torch.where(r > 0, ap, torch.nan)
        hit_q[q] = hit[:, 0]
    return ap_q, hit_q


def positive_outside(g, allowed_fn, chunk=2048):
    """Per query: number of positives, and number of them the filter excludes."""
    n = len(g)
    n_pos = torch.bincount(g)[g] - 1
    n_out = torch.zeros(n, dtype=torch.long, device=g.device)
    for s in range(0, n, chunk):
        q = torch.arange(s, min(s + chunk, n), device=g.device)
        pos = g[q, None] == g[None]
        pos[torch.arange(len(q), device=g.device), q] = False
        n_out[q] = (pos & ~allowed_fn(q)).sum(1)
    return n_pos, n_out


def subset_masks(g, filters):
    n_pos, n_out = positive_outside(g, filters["20ppm"])
    valid = n_pos > 0
    masks = {"full": valid, "F": valid & (n_out > 0), "F_all": valid & (n_out == n_pos)}
    masks["Fbar"] = valid & ~masks["F"]
    return masks, n_pos, n_out


def summarise(ap, hit, masks, open_hit=None):
    out = {}
    for sname, m in masks.items():
        c = int(m.sum())
        out[sname] = {"n": c,
                      "MAP@R": float(ap[m].mean()) if c else None,
                      "Hit@1": float(hit[m].mean()) if c else None}
    if open_hit is not None:
        full = masks["full"]
        lost = (open_hit > 0) & (hit == 0) & full
        gained = (open_hit == 0) & (hit > 0) & full
        out["net_loss"] = float(lost.sum()) / int(full.sum())
        out["net_gain"] = float(gained.sum()) / int(full.sum())
        out["net_loss_in_F"] = float((lost & masks["F"]).sum()) / max(int(masks["F"].sum()), 1)
    return out


def isotope_breakdown(rows, prec, charge):
    """Measured vs theoretical precursor m/z (peptide_neutral_mass), as in OBSERVATIONS."""
    from msdelta.reranking import peptide_neutral_mass
    m = torch.tensor([peptide_neutral_mass(p) for p in rows["peptide"]], dtype=torch.float64)
    z = charge.cpu().double()
    p = prec.cpu()
    mz = (m + z * mwe.PROTON) / z
    ppm = (p - mz) / mz * 1e6
    dneutral = (p - mz) * z
    k = torch.round(dneutral / ISOTOPE)
    resid_ppm = (dneutral - k * ISOTOPE).abs() / m * 1e6
    out = (ppm.abs() > PPM)
    return {"spectra": len(p),
            "median_abs_ppm": float(ppm.abs().median()),
            "fraction_exactly_theoretical(<0.01ppm)": float((ppm.abs() < 0.01).double().mean()),
            "fraction_outside_20ppm": float(out.double().mean()),
            "outside_by_isotope_k": {int(kk): float(((k == kk) & out).double().mean())
                                     for kk in torch.unique(k[out]).tolist()},
            "fraction_not_explained_by_isotope(resid>20ppm)": float((resid_ppm > PPM).double().mean())}


def load_embeddings(name, dname, rows, emb_dir, device):
    if name.startswith("binned_"):
        from msdelta.eval_grouped_retrieval import binned_embeddings
        e = binned_embeddings(rows, float(name.split("_", 1)[1]))
    else:
        f = Path(emb_dir) / dname / f"{name}.npy"
        if not f.exists():
            return None
        e = torch.from_numpy(np.load(f))
    return F.normalize(e.float().to(device), dim=-1)


def analyse(cli) -> int:
    from msdelta.grouped_retrieval import group_ids

    device = torch.device("xpu" if torch.xpu.is_available() else "cpu")
    report = {"ppm": PPM, "iso_k": ISO_K, "isotope_da": ISOTOPE, "filters": FILTERS,
              "subsets": SUBSETS, "results": {}}
    models = [m[0] for m in mwe.read_models(cli.models)]
    lines = []
    for dname, dpath in mwe.read_data(cli.data).items():
        rows = mwe.experimental_rows(dpath)
        g = torch.as_tensor(group_ids(rows), device=device)
        charge = torch.as_tensor([int(c) for c in rows["charge"]], device=device)
        prec = torch.as_tensor(np.asarray(rows["precursor"], dtype=np.float64), device=device)
        filters = make_filters(prec, charge)
        masks, n_pos, n_out = subset_masks(g, filters)
        _, n_out_iso = positive_outside(g, filters["iso20ppm"])
        full = masks["full"]
        info = {"isotope": isotope_breakdown(rows, prec, charge),
                "queries": int(full.sum()),
                "F": int(masks["F"].sum()), "F_all": int(masks["F_all"].sum()),
                "Fbar": int(masks["Fbar"].sum()),
                "positive_pairs": int(n_pos[full].sum()),
                "positive_pairs_outside_20ppm": int(n_out[full].sum()),
                "positive_pairs_outside_iso20ppm": int(n_out_iso[full].sum()),
                "queries_with_positive_outside_iso20ppm": int((full & (n_out_iso > 0)).sum())}
        print(f"== {dname}: {json.dumps(info)}", flush=True)
        dm = {}
        names = models + ["binned_0.1"] + (["binned_1.0"] if dname in cli.low_res else [])
        for name in names:
            e = load_embeddings(name, dname, rows, cli.emb_dir, device)
            if e is None:
                print(f"  {name}: no embeddings", flush=True)
                continue
            res, open_hit = {}, None
            for w in FILTERS:
                ap, hit = filtered_metrics(e, g, filters[w])
                if w == "open":
                    open_hit = hit
                    res[w] = summarise(ap, hit, masks)
                else:
                    res[w] = summarise(ap, hit, masks, open_hit)
            res["rescue_rate_open_hit1_on_F_all"] = res["open"]["F_all"]["Hit@1"]
            dm[name] = res
            del e
            if device.type == "xpu":
                torch.xpu.empty_cache()
            print(f"  {name:16s} " + "  ".join(
                f"{w}: " + "/".join(f"{res[w][s]['MAP@R']:.3f}" if res[w][s]["n"] else "-"
                                    for s in ("full", "F", "Fbar"))
                for w in FILTERS)
                + f"  netloss {res['20ppm']['net_loss']:.4f} rescue {res['rescue_rate_open_hit1_on_F_all']}",
                flush=True)
        report["results"][dname] = {"data": dpath, "info": info, "models": dm}
        lines += table(dname, info, dm)
    Path(cli.out).parent.mkdir(parents=True, exist_ok=True)
    Path(cli.out).write_text(json.dumps(report, indent=1))
    Path(cli.out).with_suffix(".txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0


def _f(x, fmt=".3f"):
    return "  -  " if x is None else format(x, fmt)


def table(dname, info, dm):
    q = info["queries"]
    iso = info["isotope"]
    out = [f"== {dname}: {q} queries; spectra outside 20 ppm of theory {iso['fraction_outside_20ppm']:.1%} "
           f"(by k {', '.join(f'{k:+d}: {v:.1%}' for k, v in iso['outside_by_isotope_k'].items())}); "
           f"|F| {info['F']} ({info['F'] / q:.1%}), |F_all| {info['F_all']} ({info['F_all'] / q:.1%}); "
           f"positive pairs outside 20ppm {info['positive_pairs_outside_20ppm'] / info['positive_pairs']:.1%}, "
           f"outside iso20ppm {info['positive_pairs_outside_iso20ppm'] / info['positive_pairs']:.2%}",
           "MAP@R            |       full        |         F         |       Fbar        |  F_all | Hit@1 F        | netloss      rescue",
           "model            | open  20ppm  iso  | open  20ppm  iso  | open  20ppm  iso  |  open  | open 20ppm iso | 20ppm  iso   F_all"]
    for name, r in dm.items():
        cells = [" ".join(_f(r[w][s]["MAP@R"]) for w in FILTERS) for s in ("full", "F", "Fbar")]
        out.append(f"{name:16s} | " + " | ".join(cells)
                   + f" | {_f(r['open']['F_all']['MAP@R'])}"
                   + f" | {' '.join(_f(r[w]['F']['Hit@1']) for w in FILTERS)}"
                   + f" | {_f(r['20ppm']['net_loss'], '.4f')} {_f(r['iso20ppm']['net_loss'], '.4f')}"
                   + f" {_f(r['rescue_rate_open_hit1_on_F_all'])}")
    return out + [""]


def selftest() -> int:
    """Toy check: 3 groups; one replicate with a +1 isotope precursor error."""
    torch.manual_seed(0)
    z = 2
    base = {0: 500.0, 1: 500.004, 2: 700.0}  # groups 0 and 1 within 20 ppm of each other
    g = torch.tensor([0, 0, 0, 1, 1, 2, 2])
    prec = torch.tensor([base[int(x)] for x in g], dtype=torch.float64)
    prec[2] += ISOTOPE / z  # spectrum 2 recorded one isotope high
    charge = torch.full((len(g),), z)
    centers = torch.randn(3, 8)
    e = F.normalize(centers[g] + 0.05 * torch.randn(len(g), 8), dim=-1)
    filters = make_filters(prec, charge)
    masks, n_pos, n_out = subset_masks(g, filters)
    assert n_out.tolist() == [1, 1, 2, 0, 0, 0, 0], n_out
    assert masks["F"].tolist() == [True, True, True, False, False, False, False]
    assert masks["F_all"].tolist() == [False, False, True, False, False, False, False]
    for w in FILTERS:
        ap, hit = filtered_metrics(e, g, filters[w])
        agg, ap2, _ = mwe.windowed_metrics(e, g, filters[w])
        assert abs(agg["MAP@R"] - float(ap.nanmean())) < 1e-6
        assert torch.allclose(ap, ap2, equal_nan=True)
        print(w, [round(x, 3) for x in ap.tolist()], hit.tolist())
        if w == "open":
            assert hit.tolist() == [1.0] * 7 and torch.allclose(ap, torch.ones(7))
        if w == "20ppm":
            # q0: positives {1 in, 2 out}: rank1 = 1 (hit), rank2 = group1 or -inf -> AP 1/2
            assert abs(ap[0] - 0.5) < 1e-6 and hit[0] == 1
            # q2: both positives out, nothing in its window -> AP 0, Hit@1 0 (top is -inf)
            assert ap[2] == 0 and hit[2] == 0
        if w == "iso20ppm":
            assert torch.allclose(ap, torch.ones(7))
    # all-(-inf) row: topk returns arbitrary indices; they must not count as hits
    none = lambda q: torch.zeros(len(q), len(g), dtype=torch.bool)  # noqa: E731
    ap, hit = filtered_metrics(e, g, none)
    assert hit.sum() == 0 and ap.sum() == 0
    s = summarise(*filtered_metrics(e, g, filters["20ppm"]), masks,
                  filtered_metrics(e, g, filters["open"])[1])
    assert s["F_all"]["n"] == 1 and abs(s["net_loss"] - 1 / 7) < 1e-9, s
    print("selftest ok", json.dumps(s))
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["embed", "analyse", "selftest"])
    ap.add_argument("--models")
    ap.add_argument("--data", nargs="+")
    ap.add_argument("--emb-dir")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--low-res", nargs="*", default=["hek"],
                    help="datasets that also get binned cosine at 1.0 Da")
    ap.add_argument("--out", default="results/finetune/contrastive/filter-failure/summary.json")
    cli = ap.parse_args()
    if cli.cmd == "selftest":
        return selftest()
    return mwe.embed(cli) if cli.cmd == "embed" else analyse(cli)


if __name__ == "__main__":
    raise SystemExit(main())
