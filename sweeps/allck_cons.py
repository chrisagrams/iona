"""K188-C: consensus fine-tunes on every pretraining checkpoint, per scale (mean +- sd over 3 seeds).

Arms: $MSDELTA_EVAL/contrastive/cons-allck-<set>/s<scale>_ck<NNN>k_lr4e-4_p170k2_cons_seed<i>.json (job 8901080);
the 540k finals come from the K163 twins in cons-<set>/ (same recipe). Library search and experimental retrieval with
no filter / 20 ppm / isotope-tolerant 20 ppm; F = queries failing the filter, Fbar = passing it (F columns only when a
set has failures). Writes results/k188_allck_cons/<set>.md.
"""
import argparse
import json
import re
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import homes  # noqa: E402  (data homes, configs/homes.env)

ROOT = Path(__file__).resolve().parents[1]
R = homes.EVAL / "contrastive"
SCALES = ["025m", "050m", "100m", "200m", "400m"]
COLS = [("library/open/full/Hit@1", "lib Hit@1"), ("library/20ppm/full/Hit@1", "lib 20ppm"),
        ("library/iso20ppm/full/Hit@1", "lib iso20"), ("experimental/open/full/MAP@R", "exp MAP@R"),
        ("experimental/20ppm/full/MAP@R", "exp 20ppm"), ("experimental/iso20ppm/full/MAP@R", "exp iso20")]
FCOLS = [("library/open/F/Hit@1", "lib F open"), ("library/20ppm/F/Hit@1", "lib F 20ppm"),
         ("library/iso20ppm/F/Hit@1", "lib F iso20"), ("library/20ppm/Fbar/Hit@1", "lib Fbar 20ppm"),
         ("experimental/open/F/MAP@R", "exp F open"), ("experimental/20ppm/F/MAP@R", "exp F 20ppm"),
         ("experimental/iso20ppm/F/MAP@R", "exp F iso20"), ("experimental/20ppm/Fbar/MAP@R", "exp Fbar 20ppm")]


def load(path):
    return json.loads(path.read_text())["metrics"] if path.exists() else None


def ms(xs):
    xs = [x for x in xs if x is not None]
    if not xs:
        return "--"
    return f"{statistics.mean(xs):.4f}" + (f" ± {statistics.stdev(xs):.4f}" if len(xs) > 1 else "")


def rows_for(s):
    """(scale, ck in thousands, [metrics per seed]) for every scale/checkpoint that has results."""
    out = []
    for sc in SCALES:
        cks = sorted({int(m[1]) for p in (R / f"cons-allck-{s}").glob(f"s{sc}_ck*_cons_seed*.json")
                      if (m := re.search(r"_ck(\d+)k_", p.name))})
        for ck in cks:
            out.append((sc, ck, [load(R / f"cons-allck-{s}" / f"s{sc}_ck{ck:03d}k_lr4e-4_p170k2_cons_seed{i}.json")
                                 for i in range(3)]))
        final = [load(R / f"cons-{s}" / f"s{sc}_ck540k_lr4e-4_p170k2_cons_seed{i}.json") for i in range(3)]
        if any(final):
            out.append((sc, 540, final))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sets", nargs="*", default=["validation", "test", "oodval", "mouse", "human", "yeast"])
    for s in ap.parse_args().sets:
        rows = rows_for(s)
        if not rows:
            print(f"{s}: no results yet")
            continue
        allm = [m for _, _, ms_ in rows for m in ms_ if m]
        nf = max(max(m.get("library/20ppm/F/n", 0), m.get("experimental/20ppm/F/n", 0)) for m in allm)
        cols = [c for c in COLS + (FCOLS if nf else []) if any(c[0] in m for m in allm)]  # drop columns a set lacks
        lines = [f"# K188-C all-checkpoint scaling with consensus: {s}", "",
                 "Mean ± sd over the seeds present (n). ck = pretraining checkpoint (thousand steps); 540k rows are the "
                 "K163 consensus finals. lib = library search Hit@1, exp = experimental retrieval MAP@R; open = no "
                 f"precursor filter, 20ppm / iso20 = filtered; F = queries failing the 20 ppm filter, Fbar = passing it, each "
                 f"scored with no filter / 20 ppm / iso20. Queries failing the 20 ppm filter: {int(nf)}"
                 + ("" if nf else " (so no F/Fbar split)") + ".", "",
                 "| scale | ck | n | " + " | ".join(c for _, c in cols) + " |",
                 "|---|---:|---:|" + "---:|" * len(cols)]
        for sc, ck, mets in rows:
            got = [m for m in mets if m]
            lines.append(f"| {sc} | {ck} | {len(got)} | " + " | ".join(ms([m.get(k) for m in got]) for k, _ in cols) + " |")
        out = homes.RESULTS / "k188_allck_cons" / f"{s}.md"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text("\n".join(lines) + "\n")
        print("\n".join(lines) + f"\n-> {out}\n")


if __name__ == "__main__":
    main()
