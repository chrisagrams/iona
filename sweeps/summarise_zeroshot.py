"""Rebuild $MSDELTA_DERIVED/tables/zeroshot-layers-abtt/summary.csv from every
$MSDELTA_EVAL/contrastive/zeroshot-layers-abtt/zs_*.json
(one row per frozen encoder; ABTT numbers from the TRAIN fit). Replaces the 8-encoder summary written
before the 400M/200M reruns finished.

    .venv/bin/python sweeps/summarise_zeroshot.py
"""
import glob, json, os, re, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import homes  # noqa: E402  (data homes, configs/homes.env)

D = str(homes.EVAL / "contrastive" / "zeroshot-layers-abtt")
OUT = str(homes.DERIVED / "tables" / "zeroshot-layers-abtt")
rows = []
for f in glob.glob(f"{D}/zs_*.json"):
    d = json.load(open(f))
    sc, ck = re.match(r"zs_0*(\d+m)_ck(\d+k)", os.path.basename(f)).groups()
    lay = {k: v["experimental/MAP@R"] for k, v in d["layers"].items()}
    tr = d["abtt"]["train"]
    cells = [(D_, L, v["experimental/MAP@R"]) for D_, Ls in tr.items() if D_ != "center" for L, v in Ls.items()]
    best = max(cells, key=lambda t: t[2])
    abtt_final = max(t[2] for t in cells if t[1] == "final")
    rb = max(lay, key=lay.get)
    rows.append((int(sc[:-1]), int(ck[:-1]), f"{sc},{ck},{lay['final']:.4f},{lay[rb]:.4f},{rb},{abtt_final:.4f},"
                 f"{best[2]:.4f},{best[0]},{best[1]}"))
rows.sort()
os.makedirs(OUT, exist_ok=True)
with open(f"{OUT}/summary.csv", "w") as fh:
    fh.write("scale,ckpt,raw_final,raw_best_block,best_block,abtt_final,abtt_best,abtt_best_D,abtt_best_layer\n")
    fh.write("\n".join(r[2] for r in rows) + "\n")
print(len(rows), "encoders ->", f"{OUT}/summary.csv")
