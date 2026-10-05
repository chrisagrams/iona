"""Score saved contrastive encoders on retrieval, and against the proxy that chose them.

    python -m msdelta.eval.eval_retrieval --job 8848049 \
        --baseline /flare/UIC-HPC/khuss/msdelta/pretrained/msdelta-50m-production-01-checkpoint-133233

THE QUESTION. Every contrastive result in this project is ranked by
`sep_spectrum/ratio`, out-group over in-group mean distance. Contrastive exists here to
serve retrieval and reranking. Nothing has ever checked that the first predicts the
second, and they can come apart: the ratio averages over ALL pairs, retrieval depends
only on the nearest few, so a model that tightens the bulk of the distribution while
leaving the hardest confusions untouched improves the ratio and not the task.

Two things are reported, and the second matters more than the first:

  ABSOLUTE. Retrieval for each trained encoder, against the untrained pretrained
  encoder as a baseline. If contrastive training does not move Hit@1, the separation
  ratio it moves by 4 points is measuring something that does not matter.

  CORRELATION. Spearman between the separation ratio and each retrieval metric across
  the arms. A high correlation licenses every hyperparameter and scale conclusion drawn
  from the ratio; a low one invalidates the selection, not just the reporting.

Runs on saved encoders, so it costs no retraining -- every contrastive run writes
`final/` as a drop-in --pretrained_path.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
from pathlib import Path

import numpy as np
import torch

RUNS = "/lus/flare/projects/UIC-HPC/khuss/msdelta/runs"


def score_encoder(path: str, datasets, collator, device, max_rows: int,
                  batch_size: int) -> dict:
    """Load one saved encoder; report separation and retrieval from ONE embedding pass.

    Embedding once rather than twice is both the point and a necessity. The point,
    because the two metrics must be computed over identical rows or their correlation
    means nothing. The necessity, because DeltaMZBias materialises a
    (batch, peaks, peaks, 2*n_freqs) tensor -- at batch 16 and 512 peaks that is 8 GiB,
    which is what the first attempt died on.
    """
    from msdelta.finetuning.contrastive.contrastive import (MSDeltaForContrastive, embed_dataset,
                                     retrieval_metrics_exact)
    from msdelta.models.loading import load_strict
    from msdelta.models.modeling_msdelta import MSDeltaForPreTraining
    from msdelta.rescoring.reranking import group_separation_metrics

    encoder = load_strict(MSDeltaForPreTraining, path)
    model = MSDeltaForContrastive(encoder, None, kl_weight=0).to(device)
    try:
        embeddings, groups = embed_dataset(model, datasets["validation"], collator,
                                           device, max_rows=max_rows,
                                           batch_size=batch_size)
        if embeddings is None:
            return {}
        out = dict(group_separation_metrics(embeddings, groups, "sep_spectrum"))
        counts = np.bincount(groups)
        if (counts > 1).sum() >= 2:
            out |= {f"retrieval/{k}": v
                    for k, v in retrieval_metrics_exact(embeddings, groups).items()}
            out["retrieval/queries"] = float(len(embeddings))
            out["retrieval/scorable_groups"] = float((counts > 1).sum())
    finally:
        del model, encoder
        if device.type == "xpu":
            torch.xpu.empty_cache()
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--job", required=True, help="sweep job id whose arms to score")
    ap.add_argument("--baseline", help="an untrained checkpoint, scored for reference")
    ap.add_argument("--runs", default=RUNS)
    ap.add_argument("--arms", help="regex; score only arms whose name matches. "
                    "A 96-arm job does not fit a debug hour, and the half of it "
                    "that answers the question usually does.")
    ap.add_argument("--dataset-repo", default="chrisagrams/ms2-peptide-replicate-retrieval")
    ap.add_argument("--max-rows", type=int, default=2000)
    # DeltaMZBias is O(batch * peaks^2 * n_freqs); 16 x 512 peaks is 8 GiB.
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--out", help="default: $MSDELTA_EVAL/contrastive/retrieval_vs_separation.json "
                    "(source pbs/lib/homes.sh for MSDELTA_EVAL)")
    cli = ap.parse_args(argv)
    if cli.out is None:
        if not os.environ.get("MSDELTA_EVAL"):
            ap.error("--out not given and MSDELTA_EVAL is not set (source pbs/lib/homes.sh)")
        cli.out = os.path.join(os.environ["MSDELTA_EVAL"], "contrastive", "retrieval_vs_separation.json")

    from msdelta.finetuning.contrastive.finetune_contrastive import ContrastiveCollator
    from msdelta.models.processing_msdelta import MSDeltaProcessor
    from msdelta.rescoring.reranking import build_alignment_datasets

    device = torch.device("xpu" if torch.xpu.is_available() else "cpu")
    arms = sorted(glob.glob(f"{cli.runs}/sweep-*-{cli.job}"))
    arms = [a for a in arms if os.path.isdir(os.path.join(a, "final"))]
    if cli.arms:
        pattern = re.compile(cli.arms)
        # Search the arm name only, not the path: RUNS contains the job id and a
        # digit-bearing pattern would otherwise match every arm through the prefix.
        kept = [a for a in arms
                if pattern.search(re.sub(rf"^sweep-|-{cli.job}$", "",
                                         os.path.basename(a)))]
        print(f"[retrieval] --arms {cli.arms!r} kept {len(kept)} of {len(arms)}",
              flush=True)
        arms = kept
    if not arms:
        raise SystemExit(f"no arms with a saved final/ under job {cli.job}")
    print(f"[retrieval] {len(arms)} arms, device {device}", flush=True)

    processor = MSDeltaProcessor.from_pretrained(arms[0] + "/final", max_peaks=512)
    datasets = build_alignment_datasets(cli.dataset_repo, processor,
                                        validation_fraction=0.1, seed=0)
    collator = ContrastiveCollator(max_peptide_length=64, pad_spectra_to=512)

    rows = {}
    if cli.baseline:
        rows["BASELINE-untrained"] = score_encoder(cli.baseline, datasets, collator,
                                                   device, cli.max_rows,
                                                   cli.batch_size)
        print(f"  baseline: {rows['BASELINE-untrained']}", flush=True)
    for a in arms:
        name = re.sub(rf"^sweep-|-{cli.job}$", "", os.path.basename(a))
        rows[name] = score_encoder(a + "/final", datasets, collator, device,
                                   cli.max_rows, cli.batch_size)
        print(f"  {name}: ratio {rows[name].get('sep_spectrum/ratio', float('nan')):.2f}"
              f"  Hit@1 {rows[name].get('retrieval/Hit@1', float('nan')):.4f}", flush=True)

    trained = {k: v for k, v in rows.items() if not k.startswith("BASELINE")}
    ratio = [v.get("sep_spectrum/ratio") for v in trained.values()]
    summary = {}
    for metric in ("retrieval/Hit@1", "retrieval/R@5", "retrieval/MAP@100"):
        vals = [v.get(metric) for v in trained.values()]
        pairs = [(r, m) for r, m in zip(ratio, vals)
                 if r is not None and m is not None]
        if len(pairs) > 2:
            from scipy import stats
            rho, p = stats.spearmanr([r for r, _ in pairs], [m for _, m in pairs])
            summary[metric] = {"spearman_vs_ratio": float(rho), "p": float(p),
                               "n": len(pairs)}
            print(f"  {metric} vs separation ratio: rho={rho:+.3f} p={p:.4f} "
                  f"n={len(pairs)}", flush=True)

    Path(cli.out).parent.mkdir(parents=True, exist_ok=True)
    Path(cli.out).write_text(json.dumps({"job": cli.job, "arms": rows,
                                         "correlation": summary}, indent=1))
    print(f"[retrieval] wrote {cli.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
