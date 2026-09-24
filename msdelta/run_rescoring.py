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


def embedding_only_hit1(cosine, labels, groups, kinds) -> dict[str, float]:
    """Rerank by the embedding alone: per spectrum, is the true candidate the argmax cosine?

    `hit@1` over every candidate set; `vs_<kind>` restricts each set to the truth plus
    that decoy kind only, so the result says which decoys the embedding can and cannot
    separate. Ties count as misses (argmax takes the first, and the truth is not
    guaranteed to come first).
    """
    out, hits = {}, []
    for g in np.unique(groups):
        idx = np.flatnonzero(groups == g)
        best = idx[np.argmax(cosine[idx])]
        hits.append(labels[best] == 1 and np.sum(cosine[idx] == cosine[best]) == 1)
    out["hit@1"] = float(np.mean(hits))
    for kind in sorted(set(kinds) - {"true"}):
        wins = []
        for g in np.unique(groups):
            idx = np.flatnonzero(groups == g)
            truth = idx[labels[idx] == 1]
            rivals = idx[kinds[idx] == kind]
            if len(truth) and len(rivals):
                wins.append(cosine[truth[0]] > cosine[rivals].max())
        out[f"vs_{kind}"] = float(np.mean(wins)) if wins else float("nan")
    return out


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
    parser.add_argument("--seeds", type=int, default=1,
                        help="repeat both arms over this many seeds. A single seed "
                             "cannot separate a real effect from training variance: "
                             "hit@1 is measured on a few hundred spectra and adding a "
                             "feature should not make a classifier worse.")
    parser.add_argument("--out", default="")
    parser.add_argument("--diagnose", action="store_true",
                        help="print how embedding_cosine behaves WITHIN each spectrum's "
                             "candidate list, per decoy kind, then exit without training")
    parser.add_argument("--wandb_project", default="",
                        help="log per-seed and paired results to this W&B project; off if empty")
    parser.add_argument("--run_name", default="",
                        help="W&B run name; defaults to the student's run directory")
    parser.add_argument("--wandb_group", default="",
                        help="W&B group, e.g. the teacher, so a teacher's alignment and "
                             "rescoring runs sit together")
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
    truth_errors = []
    true_peptide = []   # per row: the spectrum's TRUE peptide, the unit the split holds out
    for index, row in enumerate(rows):
        mz = np.asarray(row["mz"], dtype=np.float64)
        intensity = np.exp(np.asarray(row["log_intensity"], dtype=np.float64)) - 1.0
        charge = int(row.get("charge") or 2)
        # The MEASURED precursor m/z from the spectrum. This used to be rebuilt from the
        # TRUE peptide's theoretical mass, which made the truth's mass_error_ppm exactly
        # 0 and every differently-massed decoy's non-zero: the label was a feature, and
        # it is why mass-matched decoys scored Hit@1 0.999. No fallback on purpose.
        precursor = float(row.get("precursor") or 0.0)
        if precursor <= 0:
            raise SystemExit("cache has no measured precursor; rebuild it with the "
                             "current build_alignment_datasets")
        truth_errors.append((precursor - (masses[index] + charge * 1.00727646) / charge)
                            / ((masses[index] + charge * 1.00727646) / charge) * 1e6)

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
            true_peptide.append(row["peptide"])
        if index % 500 == 0:
            print(f"[rescore] {index}/{len(rows)}", flush=True)

    features = np.stack(features)
    labels, groups = np.array(labels), np.array(groups)
    print(f"[rescore] {len(labels):,} pairs, {labels.sum():,} true; decoys "
          + ", ".join(f"{k}={kinds.count(k)}" for k in sorted(set(kinds)) if k != "true"),
          flush=True)

    te = np.abs(np.asarray(truth_errors))
    print(f"[rescore] TRUE candidate |mass error| vs measured precursor: median "
          f"{np.median(te):.2f} ppm, p95 {np.percentile(te, 95):.2f} ppm, max {te.max():.1f} "
          f"(0 everywhere would mean the precursor is still derived from the answer)",
          flush=True)
    if student is not None:
        cos = np.asarray([f[FEATURE_NAMES.index("embedding_cosine")] for f in features])
        emb_only = embedding_only_hit1(cos, labels, groups, np.asarray(kinds))
        print("[rescore] EMBEDDING ONLY (argmax student-teacher cosine per spectrum, no "
              "classifier, all spectra): " + "  ".join(f"{k} {v:.4f}" for k, v in emb_only.items()),
              flush=True)
    if cli.diagnose:
        diagnose(np.asarray([f[FEATURE_NAMES.index("embedding_cosine")] for f in features]),
                 np.asarray(labels), np.asarray(groups), np.asarray(kinds))
        return 0

    results: dict[str, list[dict]] = {"with_embedding": [], "without_embedding": []}
    for seed in range(cli.seed, cli.seed + cli.seeds):
        for drop in (False, True):
            out = train_rescorer(features, labels, groups, drop_embedding=drop,
                                 epochs=cli.epochs, seed=seed,
                                 split_keys=np.asarray(true_peptide))
            out.pop("model")
            key = "without_embedding" if drop else "with_embedding"
            results[key].append(out)
            print(f"[rescore] seed {seed} {'without' if drop else 'with':>7} embedding: "
                  f"AUROC {out['auroc']:.4f}  hit@1 {out['hit@1']:.4f}", flush=True)

    print(f"\n[rescore] over {cli.seeds} seed(s):", flush=True)
    summary = {}
    for key, runs in results.items():
        for metric in ("auroc", "hit@1"):
            values = np.array([r[metric] for r in runs])
            summary[f"{key}/{metric}"] = {"mean": float(values.mean()),
                                          "sd": float(values.std(ddof=1)) if len(values) > 1 else 0.0}
            print(f"    {key:<18} {metric:<6} {values.mean():.4f} "
                  f"+- {values.std(ddof=1) if len(values) > 1 else 0.0:.4f}", flush=True)
    for metric in ("auroc", "hit@1"):
        a = np.array([r[metric] for r in results["with_embedding"]])
        b = np.array([r[metric] for r in results["without_embedding"]])
        # Paired: both arms share a seed, hence the same split and initialisation, so
        # the pairing removes most of the variance the comparison is fighting.
        delta = a - b
        spread = delta.std(ddof=1) if len(delta) > 1 else float("nan")
        print(f"[rescore] embedding contributes {metric}: {delta.mean():+.4f} "
              f"+- {spread:.4f} (paired over seeds)", flush=True)
    paired = {}
    for metric in ("auroc", "hit@1"):
        d = (np.array([r[metric] for r in results["with_embedding"]])
             - np.array([r[metric] for r in results["without_embedding"]]))
        paired[metric] = {"mean": float(d.mean()),
                          "sd": float(d.std(ddof=1)) if len(d) > 1 else 0.0}
    runs = results
    results = {"runs": runs, "summary": summary, "paired_delta": paired,
               "student": cli.student, "target_cache": cli.target_cache,
               "teacher": manifest.get("teacher", "")}
    # Written BEFORE any W&B call, so a logging failure can never cost the result.
    if cli.out:
        Path(cli.out).write_text(json.dumps(results, indent=2))
    if cli.wandb_project:
        log_to_wandb(cli, runs, summary, paired, manifest)
    return 0


def diagnose(cos, labels, groups, kinds) -> None:
    """Where does embedding_cosine rank the truth inside each spectrum's candidates?

    Pooled AUROC went up with the feature while Hit@1 fell 11 points, which means it
    separates true from false ACROSS spectra but not WITHIN one. This shows which decoy
    kind beats the truth on cosine, and by how much, spectrum by spectrum.
    """
    import collections
    print("\n[diagnose] mean cosine by candidate kind:", flush=True)
    for k in sorted(set(kinds)):
        c = cos[kinds == k]
        print(f"    {k:<13} n={len(c):6d}  mean {c.mean():+.4f}  sd {c.std():.4f}", flush=True)
    wins = collections.Counter(); ranked = hits = ties = 0; margins = collections.defaultdict(list)
    for g in np.unique(groups):
        rows = groups == g
        if labels[rows].sum() != 1:
            continue
        ranked += 1
        c, lab, kd = cos[rows], labels[rows], kinds[rows]
        t = c[lab == 1][0]
        for k, v in zip(kd[lab == 0], c[lab == 0]):
            margins[k].append(t - v)
        top = int(np.argmax(c))
        if lab[top] == 1:
            hits += 1
        else:
            wins[kd[top]] += 1
        if np.sum(np.isclose(c, c.max())) > 1:
            ties += 1
    print(f"[diagnose] cosine-only Hit@1 within spectrum: {hits/ranked:.4f} over {ranked} spectra"
          f"  (ties at the top: {ties})", flush=True)
    print(f"[diagnose] when cosine picks wrong, the winner is: {dict(wins)}", flush=True)
    for k, m in sorted(margins.items()):
        m = np.asarray(m)
        print(f"    truth - {k:<13} mean {m.mean():+.4f}  truth ahead in {np.mean(m > 0):.3f}"
              f"  exactly tied {np.mean(np.isclose(m, 0)):.3f}", flush=True)


def log_to_wandb(cli, runs, summary, paired, manifest) -> None:
    """One W&B run per rescoring job: per-seed rows, the arm means, and the paired delta."""
    try:
        import wandb
        name = cli.run_name or (Path(cli.student).parent.name if cli.student else "no-student")
        run = wandb.init(project=cli.wandb_project, name=f"rescore-{name}",
                         group=cli.wandb_group or None, job_type="rescoring",
                         config={"student": cli.student, "target_cache": cli.target_cache,
                                 "teacher": manifest.get("teacher", ""),
                                 "seeds": cli.seeds, "epochs": cli.epochs})
        table = wandb.Table(columns=["seed", "arm", "auroc", "hit@1"])
        for arm, rows in runs.items():
            for i, r in enumerate(rows):
                table.add_data(cli.seed + i, arm, r["auroc"], r["hit@1"])
        run.log({"per_seed": table})
        flat = {k.replace("/", "_") + "_mean": v["mean"] for k, v in summary.items()}
        flat.update({f"embedding_contribution_{m}": v["mean"] for m, v in paired.items()})
        flat.update({f"embedding_contribution_{m}_sd": v["sd"] for m, v in paired.items()})
        run.summary.update(flat)
        run.finish()
        print(f"[rescore] logged to W&B project {cli.wandb_project}", flush=True)
    except Exception as error:   # the JSON is already on disk
        print(f"[rescore] W&B logging failed, results are in --out: {error}", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
