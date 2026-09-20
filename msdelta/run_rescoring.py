"""Train the rescoring classifier with and without the embedding, and compare.

    python -m msdelta.run_rescoring --target_cache PATH --student PATH

The question: the alignment embedding reaches AUROC 0.846 on true-vs-decoy pairs by
itself, which is real signal, but fragment coverage and precursor mass error may already
capture the same discrimination. A redundant feature, however strong alone, contributes
nothing. `--drop_embedding` is the ablation and this runs both arms on identical splits
so the difference is attributable.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from msdelta.rescoring import FEATURE_NAMES, build_candidates, extract_features, train_rescorer


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target_cache", required=True,
                        help="precompute_align cache: spectra plus teacher embeddings")
    parser.add_argument("--student", default="",
                        help="trained alignment student; without it embedding_cosine is 0")
    parser.add_argument("--split", default="validation")
    parser.add_argument("--max_spectra", type=int, default=0)
    parser.add_argument("--near_miss", type=int, default=2)
    parser.add_argument("--mass_matched", type=int, default=2)
    parser.add_argument("--tolerance_ppm", type=float, default=20.0)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="")
    cli = parser.parse_args()

    from datasets import load_from_disk
    cache = Path(cli.target_cache)
    rows = list(load_from_disk(str(cache / cli.split)))
    if cli.max_spectra:
        rows = rows[: cli.max_spectra]
    print(f"[rescore] {len(rows):,} spectra from {cache / cli.split}", flush=True)

    student, manifest = None, {}
    if cli.student:
        from safetensors.torch import load_file
        from msdelta.reranking import PeptideEncoder
        manifest = dict(l.split(": ", 1) for l in
                        (cache / "MANIFEST.txt").read_text().splitlines() if ": " in l)
        state = load_file(Path(cli.student) / "model.safetensors")
        student = PeptideEncoder(embedding_size=int(manifest["embedding_size"]),
                                 hidden_size=256, num_layers=4, num_heads=8,
                                 pooling=manifest["pooling"]).eval()
        student.load_state_dict({k.removeprefix("sequence_encoder."): v
                                 for k, v in state.items()
                                 if k.startswith("sequence_encoder.")})
        print(f"[rescore] student from {cli.student}", flush=True)

    # Mass-matched decoys: real peptides from the corpus near this precursor. These are
    # the candidates a search engine would actually return, so they are the ones worth
    # separating; a random peptide of the wrong mass is free to reject.
    from msdelta.rescoring import peptide_mass, split_peptide
    masses = np.array([peptide_mass(*split_peptide(r["peptide"])) for r in rows])
    peptides = [r["peptide"] for r in rows]
    order = np.argsort(masses)

    rng = np.random.default_rng(cli.seed)
    features, labels, groups, kinds = [], [], [], []
    for index, row in enumerate(rows):
        mz = np.asarray(row["mz"], dtype=np.float64)
        intensity = np.exp(np.asarray(row["log_intensity"], dtype=np.float64)) - 1.0
        charge = int(row.get("charge") or 2)
        precursor = (masses[index] + charge * 1.00727646) / charge

        near = np.searchsorted(masses[order], masses[index])
        window = [peptides[order[j]] for j in
                  range(max(0, near - 4), min(len(order), near + 5))
                  if peptides[order[j]] != row["peptide"]][: cli.mass_matched]
        candidates = build_candidates(row["peptide"], rng, mass_matched=window,
                                      n_near_miss=cli.near_miss)

        cosines = np.zeros(len(candidates.peptides), dtype=np.float32)
        if student is not None:
            from msdelta.reranking import PeptideCollator
            with torch.no_grad():
                emb = student(**PeptideCollator()(candidates.peptides,
                                                  [charge] * len(candidates.peptides)))
            target = torch.nn.functional.normalize(
                torch.tensor(row["target"], dtype=torch.float32), dim=-1)
            cosines = (emb @ target).numpy()

        for peptide, label, kind, cosine in zip(candidates.peptides, candidates.labels,
                                                candidates.kinds, cosines):
            features.append(extract_features(peptide, mz, intensity, precursor, charge,
                                             float(cosine), cli.tolerance_ppm))
            labels.append(label); groups.append(index); kinds.append(kind)
        if index % 500 == 0:
            print(f"[rescore] {index}/{len(rows)}", flush=True)

    features = np.stack(features)
    labels, groups = np.array(labels), np.array(groups)
    print(f"[rescore] {len(labels):,} pairs, {labels.sum():,} true; decoys "
          + ", ".join(f"{k}={kinds.count(k)}" for k in sorted(set(kinds)) if k != "true"),
          flush=True)

    results = {}
    for drop in (False, True):
        out = train_rescorer(features, labels, groups, drop_embedding=drop,
                             epochs=cli.epochs, seed=cli.seed)
        out.pop("model")
        results["without_embedding" if drop else "with_embedding"] = out
        print(f"[rescore] {'without' if drop else 'with':>7} embedding: "
              f"AUROC {out['auroc']:.4f}  hit@1 {out['hit@1']:.4f}  "
              f"({out['features']} features, {out['spectra']} spectra)", flush=True)

    a, b = results["with_embedding"], results["without_embedding"]
    print(f"\n[rescore] embedding contributes: AUROC {a['auroc']-b['auroc']:+.4f}  "
          f"hit@1 {a['hit@1']-b['hit@1']:+.4f}", flush=True)
    if cli.out:
        Path(cli.out).write_text(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
