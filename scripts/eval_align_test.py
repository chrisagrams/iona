"""Score an alignment student on ms-contrastive-100k's TEST split. No training.

    python scripts/eval_align_test.py --run RUN_DIR --cache CACHE_DIR \
        --args_file configs/finetune/align-100k-50m/training.args --data EVAL_DATA --out OUT.json

Ranks every test spectrum against every test analyte (Hit@1, Hit@5, MRR).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from datasets import load_from_disk
from safetensors.torch import load_file
from transformers import HfArgumentParser

from iona.data import group_ids, map_length_sorted
from iona.finetune_align import AlignDataArguments, AlignModelArguments
from iona.modeling_iona import IonaForPreTraining
from iona.reranking import (
    AlignmentCollator,
    PeptideCollator,
    PeptideEncoder,
    cross_modal_metrics,
    embed_spectrum,
)
from iona.retrieval import retrieval_metrics


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, help="alignment run dir with final/")
    ap.add_argument("--cache", required=True, help="teacher cache (for MANIFEST)")
    ap.add_argument("--args_file", required=True, help="the student's training.args")
    ap.add_argument("--data", required=True, help="prepared test split (eval_grouped)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--save-embeddings", default="",
                    help="also write embeddings + keys (.npz)")
    cli = ap.parse_args(argv)

    device = torch.device("xpu" if torch.xpu.is_available() else "cpu")
    manifest = dict(line.split(": ", 1) for line in
                    (Path(cli.cache) / "MANIFEST.txt").read_text().splitlines()
                    if ": " in line)
    model_args, _, _ = HfArgumentParser((AlignModelArguments, AlignDataArguments)) \
        .parse_args_into_dataclasses(args=["--args_file", cli.args_file,
                                           "--pretrained_path", manifest["teacher"]],
                                     args_file_flag="--args_file",
                                     return_remaining_strings=True)
    teacher = IonaForPreTraining.from_pretrained(manifest["teacher"]).to(device).eval()
    state = load_file(str(Path(cli.run) / "final" / "model.safetensors"))
    student = PeptideEncoder(embedding_size=int(manifest["embedding_size"]),
                             hidden_size=model_args.sequence_hidden_size,
                             num_layers=model_args.sequence_num_layers,
                             num_heads=model_args.sequence_num_heads,
                             max_length=model_args.max_peptide_length,
                             dropout=model_args.sequence_dropout,
                             pooling=model_args.pooling)
    prefix = "sequence_encoder."
    student_state = {k[len(prefix):]: v for k, v in state.items() if k.startswith(prefix)}
    student.load_state_dict(student_state, strict=True)
    student = student.to(device).eval()

    rows = load_from_disk(cli.data)
    rows = rows.select(np.flatnonzero(np.array(rows["source"]) == "experimental"))
    collator = AlignmentCollator(max_peptide_length=model_args.max_peptide_length,
                                 pad_spectra_to=int(manifest["max_peaks"]))
    features = list(rows)

    # candidates: every distinct (peptide, charge) among the test rows, in first-seen order
    keys = [(f["peptide"], int(f["charge"])) for f in features]
    slot: dict = {}
    for k in keys:
        slot.setdefault(k, len(slot))
    candidates = list(slot)
    spectrum_group = np.array([slot[k] for k in keys])

    def embed_batch(chunk):
        batch = collator(chunk)
        return {"spectrum": embed_spectrum(teacher, batch["mz"].to(device),
                                           batch["log_intensity"].to(device),
                                           batch["attention_mask"].to(device),
                                           manifest["pooling"]).float().cpu().numpy()}

    sequences = []
    with torch.no_grad():
        spectra = [torch.from_numpy(np.stack(
            map_length_sorted(rows, embed_batch, cli.batch_size)["spectrum"]))]
        peptide_collator = PeptideCollator(max_length=model_args.max_peptide_length)
        for start in range(0, len(candidates), 256):
            chunk = candidates[start:start + 256]
            batch = peptide_collator([p for p, _ in chunk], [c for _, c in chunk])
            sequences.append(student(batch["residues"].to(device),
                                     batch["modifications"].to(device),
                                     batch["sequence_mask"].to(device),
                                     batch["charge"].to(device)).float().cpu())
    spectrum_emb, sequence_emb = torch.cat(spectra), torch.cat(sequences)

    if cli.save_embeddings:
        np.savez(cli.save_embeddings, spectrum=spectrum_emb.numpy(),
                 sequence=sequence_emb.numpy(), spectrum_group=spectrum_group,
                 cand_peptide=np.array([p for p, _ in candidates]),
                 cand_charge=np.array([c for _, c in candidates]))
    metrics = cross_modal_metrics(sequence_emb, spectrum_emb, spectrum_group,
                                  np.arange(len(candidates)))
    teacher_ref = retrieval_metrics(spectrum_emb, group_ids(rows), device)
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
