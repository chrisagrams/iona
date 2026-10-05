"""K163-C: consensus twins vs their no-consensus partners, per scale (mean +- sd over 3 seeds).

Twins: $MSDELTA_EVAL/contrastive/cons-<set>/<arm>.json; partners: hp-scale-<scale>-<set>/<arm minus _cons>.json.
Library search with no filter / 20 ppm / isotope-tolerant 20 ppm; F = queries failing the filter, Fbar = passing it.
Prints each set's table and writes it to results/k163_cons/<set>.md.
"""
import argparse
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import homes  # noqa: E402  (data homes, configs/homes.env)

R = homes.EVAL / "contrastive"
SCALES = ["025m", "050m", "100m", "200m", "400m"]
COLS = [("library/open/full/Hit@1", "lib Hit@1"), ("library/20ppm/full/Hit@1", "lib 20ppm"),
        ("library/iso20ppm/full/Hit@1", "lib iso20"), ("library/20ppm/F/Hit@1", "lib 20ppm F"),
        ("library/20ppm/Fbar/Hit@1", "lib 20ppm Fbar"), ("experimental/MAP@R", "MAP@R")]


def load(path):
    return json.loads(path.read_text())["metrics"] if path.exists() else None


def ms(xs):
    xs = [x for x in xs if x is not None]
    if not xs:
        return "--"
    return f"{statistics.mean(xs):.4f}" + (f"+-{statistics.stdev(xs):.4f}" if len(xs) > 1 else "")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sets", nargs="*", default=["validation", "test", "oodval", "mouse", "human", "yeast"])
    for s in ap.parse_args().sets:
        lines = ["", f"## {s}",
                 "| scale | n F/Fbar (20ppm) | " + " | ".join(f"{c} no-cons | cons | delta" for _, c in COLS) + " |"]
        for sc in SCALES:
            arms = [f"s{sc}_ck540k_lr4e-4_p170k2_cons_seed{i}" for i in range(3)]
            cons = [load(R / f"cons-{s}" / f"{a}.json") for a in arms]
            base = [load(R / f"hp-scale-{sc}-{s}" / f"{a.replace('_cons', '')}.json") for a in arms]
            if not any(cons):
                continue
            ref = next(m for m in cons if m)
            row = [sc, f"{ref.get('library/20ppm/F/n', 0):.0f}/{ref.get('library/20ppm/Fbar/n', 0):.0f}"]
            for k, _ in COLS:
                b = [m.get(k) if m else None for m in base]
                c = [m.get(k) if m else None for m in cons]
                d = [y - x for x, y in zip(b, c) if x is not None and y is not None]
                row += [ms(b), ms(c), (f"{statistics.mean(d):+.4f} ({len(d)})" if d else "--")]
            lines.append("| " + " | ".join(row) + " |")
        print("\n".join(lines))
        out = homes.RESULTS / "k163_cons" / f"{s}.md"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
