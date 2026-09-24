"""Score an alignment student on ms-contrastive-100k's TEST split. No training.

    python -m msdelta.eval_align_test --run RUN_DIR --cache CACHE_DIR \
        --args_file configs/a1-align-100k-.../training.args --data EVAL_DATA --out OUT.json

The in-training check (finetune_align.evaluate_alignment) ranks the first 2,000
VALIDATION spectra against the few hundred peptides that happen to occur in them. This
ranks every experimental test spectrum (replicate-corpus peptides excluded, as prepared by
eval_grouped_retrieval) against EVERY test analyte (peptide + charge, the student's input),
~9,950 candidates -- the realistic search space.

Spectra are embedded by the teacher named in the cache MANIFEST (the exact embedding the
student was trained to hit); candidates by the student in RUN/final. Reported: Hit@1,
Hit@5, MRR over spectra (cross_modal_metrics), plus the teacher's own spectrum->spectrum
MAP@R on the same rows as a reference point.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, help="alignment run dir with final/")
    ap.add_argument("--cache", required=True, help="teacher cache (for MANIFEST)")
    ap.add_argument("--args_file", required=True, help="the student's training.args")
    ap.add_argument("--data", required=True, help="prepared test split (eval_grouped)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch-size", type=int, default=16)
    cli = ap.parse_args(argv)

    from datasets import load_from_disk
    from safetensors.torch import load_file
    from transformers import HfArgumentParser

    from msdelta.contrastive import retrieval_metrics_topk
    from msdelta.finetune_align import AlignDataArguments, AlignModelArguments
    from msdelta.grouped_retrieval import group_ids
    from msdelta.modeling_msdelta import MSDeltaForPreTraining
    from msdelta.reranking import (AlignmentCollator, PeptideEncoder,
                                   SequenceAlignmentModel, cross_modal_metrics)

    device = torch.device("xpu" if torch.xpu.is_available() else "cpu")
    manifest = dict(line.split(": ", 1) for line in
                    (Path(cli.cache) / "MANIFEST.txt").read_text().splitlines()
                    if ": " in line)
    model_args, _, _ = HfArgumentParser((AlignModelArguments, AlignDataArguments)) \
        .parse_args_into_dataclasses(args=["--args_file", cli.args_file],
                                     args_file_flag="--args_file",
                                     return_remaining_strings=True)
    teacher = MSDeltaForPreTraining.from_pretrained(manifest["teacher"])
    student = PeptideEncoder(embedding_size=int(manifest["embedding_size"]),
                             hidden_size=model_args.sequence_hidden_size,
                             num_layers=model_args.sequence_num_layers,
                             num_heads=model_args.sequence_num_heads,
                             max_length=model_args.max_peptide_length,
                             dropout=model_args.sequence_dropout,
                             pooling=model_args.pooling)
    model = SequenceAlignmentModel(teacher, student, pooling=manifest["pooling"])
    state = load_file(str(Path(cli.run) / "final" / "model.safetensors"))
    prefix = "sequence_encoder."
    student_state = {k[len(prefix):]: v for k, v in state.items() if k.startswith(prefix)}
    student.load_state_dict(student_state, strict=True)   # every student weight, no extras
    model = model.to(device).eval()

    rows = load_from_disk(cli.data)
    rows = rows.select(np.flatnonzero(np.array(rows["source"]) == "experimental"))
    collator = AlignmentCollator(max_peptide_length=model_args.max_peptide_length,
                                 pad_spectra_to=int(manifest.get("max_peaks", 512)))
    features = list(rows)

    # candidates: every distinct (peptide, charge) among the test rows, in first-seen order
    keys = [(f["peptide"], int(f["charge"])) for f in features]
    slot: dict = {}
    for k in keys:
        slot.setdefault(k, len(slot))
    candidates = list(slot)
    spectrum_group = np.array([slot[k] for k in keys])

    spectra, sequences = [], []
    with torch.no_grad():
        for start in range(0, len(features), cli.batch_size):
            batch = collator(features[start:start + cli.batch_size])
            spectra.append(model.embed_spectrum(batch["mz"].to(device),
                                                batch["log_intensity"].to(device),
                                                batch["attention_mask"].to(device)).cpu())
        dummy = [{"mz": [100.0], "log_intensity": [1.0], "peptide": p, "charge": c}
                 for p, c in candidates]
        for start in range(0, len(dummy), 256):
            batch = collator(dummy[start:start + 256])
            sequences.append(student(batch["residues"].to(device),
                                     batch["modifications"].to(device),
                                     batch["sequence_mask"].to(device),
                                     batch["charge"].to(device)).float().cpu())
    spectrum_emb, sequence_emb = torch.cat(spectra), torch.cat(sequences)

    metrics = cross_modal_metrics(sequence_emb, spectrum_emb, spectrum_group,
                                  np.arange(len(candidates)))
    teacher_ref = retrieval_metrics_topk(spectrum_emb, group_ids(rows), device=device)
    metrics |= {f"teacher_spectrum/{k}": v for k, v in teacher_ref.items()}
    out = {"run": cli.run, "cache": cli.cache, "teacher": manifest["teacher"],
           "data": cli.data, "n_spectra": len(features), "n_candidates": len(candidates),
           "metrics": metrics}
    Path(cli.out).parent.mkdir(parents=True, exist_ok=True)
    Path(cli.out).write_text(json.dumps(out, indent=1))
    print(json.dumps({k: round(v, 4) for k, v in metrics.items()}, indent=1), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
