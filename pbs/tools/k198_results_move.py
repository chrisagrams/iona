"""K198a (notes/AGENT_PLAYBOOK_2.md C7): copy everything machine-readable out of results/ into its home on /flare.

    .venv/bin/python pbs/tools/k198_results_move.py plan            # print the mapping summary, touch nothing
    .venv/bin/python pbs/tools/k198_results_move.py copy            # copy + verify (md5), never overwrite
    .venv/bin/python pbs/tools/k198_results_move.py verify          # re-check every copied file against its source

results/ keeps only what people read (Markdown, plots). This tool COPIES the rest; it deletes nothing. The originals
stay until the user approves their removal (K198b; deletion protocol). Human files (PNG / MD) are not copied: they
move inside the repo with `git mv`, listed under "repo" by `plan`.

  results/raw/finetune/<track>/...          -> $MSDELTA_EVAL/<track>/...           (evaluation outputs; primary)
  results/raw/finetune/denoise/*.txt        -> $MSDELTA_DERIVED/denoise/           (grid tables summarise_denoise.py
                                                                                     computes from run dirs; derived)
  results/raw/finetune/{README.md,checkpoint_provenance.txt} -> $MSDELTA_EVAL/
  results/raw/rerank/...                    -> $MSDELTA_EVAL/rerank/...
  results/raw/diag/k195/...                 -> $MSDELTA_DERIVED/k195/...           (curves extracted from logs /
                                                                                     trainer_state, computed floors)
  results/raw/diag/...                      -> $MSDELTA_DIAG/...                   (diagnostic job outputs; primary)
  results/processed/**/*.csv                -> $MSDELTA_DERIVED/{summary,tables}/  (tables behind the reports)
  results/processed/** and results/summary/** *.png / *.md  -> results/<report>/ (git mv; see REPORTS)

A manifest (source, destination, bytes, md5, action) is written to $MSDELTA_STORAGE/k198-logs/.
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "sweeps"))
import homes  # noqa: E402

REPO = homes.REPO
RESULTS = REPO / "results"
FIG_DIRS = {"C_contrastive": "contrastive", "D_denoise": "denoise", "P_pretrain": "pretrain", "SUMMARY": "summary"}
SUBDIRS = {"C1_recipe": "c1_recipe", "superseded": "superseded"}
SKIP = {"results/raw/diag/.gitignore", "results/README.md", "results/processed/figures/README.md"}


def destination(rel: str) -> tuple[str, Path] | None:
    """('storage', path) for a copy to /flare, ('repo', path) for a git mv inside results/, None = not moved."""
    if rel in SKIP:
        return None
    p = rel.split("/")
    if p[1] == "raw":
        if p[2] == "finetune":
            if len(p) == 4:  # README.md, checkpoint_provenance.txt
                return "storage", homes.EVAL / p[3]
            if p[3] == "denoise" and len(p) == 5 and p[4].endswith(".txt"):
                return "storage", homes.DERIVED / "denoise" / p[4]
            return "storage", homes.EVAL.joinpath(*p[3:])
        if p[2] == "rerank":
            return "storage", homes.EVAL.joinpath("rerank", *p[3:])
        if p[2] == "diag":
            if p[3] == "k195":
                return "storage", homes.DERIVED.joinpath(*p[3:])
            return "storage", homes.DIAG.joinpath(*p[3:])
    if p[1] == "processed":
        if rel.endswith(".csv"):
            area = "summary" if p[2] == "figures" else "tables"
            rest = p[4:] if p[2] == "figures" else p[3:]
            return "storage", homes.DERIVED.joinpath(area, *rest)
        if p[2] == "figures":
            return "repo", RESULTS.joinpath(FIG_DIRS[p[3]], *[SUBDIRS.get(x, x) for x in p[4:]])
        if p[2] == "tables" and p[3] == "SUMMARY_TABLES.md":
            return "repo", RESULTS / "summary" / "tables.md"
    if p[1] == "summary":
        name = p[2]
        for prefix, report in (("k188_allck_cons_", "k188_allck_cons"), ("k163_cons_", "k163_cons")):
            if name.startswith(prefix):
                return "repo", RESULTS / report / name[len(prefix):]
        if name == "k197_binned_edge.md":
            return "repo", RESULTS / "k197_binned_edge" / "report.md"
    raise ValueError(f"no rule for {rel}")


def md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def sources() -> list[str]:
    return sorted(str(f.relative_to(REPO)) for d in ("raw", "processed", "summary") for f in (RESULTS / d).rglob("*")
                  if f.is_file() and not f.is_symlink())


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=("plan", "copy", "verify"))
    mode = ap.parse_args(argv).mode
    rows = [(rel, destination(rel)) for rel in sources()]
    storage = [(rel, d) for rel, (kind, d) in ((r, x) for r, x in rows if x) if kind == "storage"]
    repo = [(rel, d) for rel, (kind, d) in ((r, x) for r, x in rows if x) if kind == "repo"]
    skipped = [rel for rel, x in rows if x is None]
    if mode == "plan":
        by_root: dict[str, int] = {}
        for _, d in storage:
            key = str(d.relative_to(homes.STORAGE)).split("/")[0] + "/" + str(d.relative_to(homes.STORAGE)).split("/")[1]
            by_root[key] = by_root.get(key, 0) + 1
        print(f"{len(rows)} files: {len(storage)} copy to /flare, {len(repo)} git mv inside results/, "
              f"{len(skipped)} not moved {skipped}")
        for k, v in sorted(by_root.items()):
            print(f"  {v:5d}  $MSDELTA_STORAGE/{k}")
        targets = {}
        for _, d in repo:
            targets[str(d.parent.relative_to(REPO))] = targets.get(str(d.parent.relative_to(REPO)), 0) + 1
        for k, v in sorted(targets.items()):
            print(f"  {v:5d}  {k}/")
        dup = len({d for _, d in storage + repo}) != len(storage + repo)
        print("destination collisions:", dup)
        return 1 if dup else 0
    log_dir = homes.STORAGE / "k198-logs"
    log_dir.mkdir(exist_ok=True)
    manifest = log_dir / f"manifest_{mode}_{time.strftime('%Y%m%dT%H%M%S')}.tsv"
    bad = 0
    with open(manifest, "w") as out:
        out.write("source\tdestination\tbytes\tmd5_source\tmd5_destination\taction\n")
        for rel, dst in storage:
            src = REPO / rel
            h = md5(src)
            if mode == "copy" and not dst.exists():
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dst)
                action = "copied"
            elif dst.exists():
                action = "present"
            else:
                action = "MISSING"
            hd = md5(dst) if dst.exists() else ""
            if hd != h:
                action = "MISSING" if not hd else "CONFLICT (destination differs; not overwritten)"
                bad += 1
            out.write(f"{rel}\t{dst}\t{src.stat().st_size}\t{h}\t{hd}\t{action}\n")
    counts: dict[str, int] = {}
    for line in manifest.read_text().splitlines()[1:]:
        a = line.split("\t")[-1]
        counts[a] = counts.get(a, 0) + 1
    print(f"{mode}: {counts}; manifest {manifest}")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
