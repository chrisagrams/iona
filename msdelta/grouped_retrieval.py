"""chrisagrams/ms-contrastive-100k flattened to one spectrum per row, for contrastive eval and training."""

from __future__ import annotations

import numpy as np
import torch
from datasets import load_dataset

from msdelta.reranking import build_alignment_datasets, peptide_key


def corpus_peptides(repo_id: str) -> set[str]:
    """Every peptide in a corpus, across all of its splits."""
    raw = load_dataset(repo_id)
    return {p for split in raw.values() for p in split["peptide"]}


def flatten_analyte(example, include_consensus: bool) -> dict[str, list]:
    """One analyte row -> parallel lists, one entry per spectrum."""
    spectra = list(example["experimental"])
    sources = ["experimental"] * len(spectra)
    if include_consensus:
        spectra = [example["consensus"], *spectra]
        sources = ["consensus", *sources]
    return {
        "mz": [list(s["mz"]) for s in spectra],
        "intensity": [list(s["intensity"]) for s in spectra],
        "peptide": [example["peptide"]] * len(spectra),
        "charge": [int(example["charge"])] * len(spectra),
        "precursor": [float(example.get("precursor") or 0.0)] * len(spectra),
        "source": sources,
        "analyte_id": [str(example.get("analyte_id") or "")] * len(spectra),
    }


def _process(processor, mz, intensity):
    values = processor(torch.as_tensor(mz, dtype=torch.float32),
                       torch.as_tensor(intensity, dtype=torch.float32), padding=False)
    out_mz, out_li = values["mz"], values["log_intensity"]
    if out_mz and isinstance(out_mz[0], list):
        out_mz, out_li = out_mz[0], out_li[0]
    return out_mz, out_li


def build_grouped_split(split, processor, include_consensus: bool,
                        exclude_peptides: set[str] | None = None, num_proc=None):
    """Flatten, drop excluded peptides and oversized spectra, and process one split."""
    max_peaks = getattr(processor, "max_peaks", None)
    exclude = exclude_peptides or set()
    before_analytes = len(split)
    if exclude:
        split = split.filter(lambda e: e["peptide"] not in exclude, num_proc=num_proc,
                             desc="exclude peptides seen in training")
    excluded = before_analytes - len(split)

    def flatten_batch(batch):
        out = {k: [] for k in ("mz", "intensity", "peptide", "charge", "precursor",
                               "source", "analyte_id")}
        for i in range(len(batch["peptide"])):
            row = flatten_analyte({k: batch[k][i] for k in batch}, include_consensus)
            for k in out:
                out[k].extend(row[k])
        return out

    flat = split.map(flatten_batch, batched=True, remove_columns=split.column_names,
                     num_proc=num_proc, desc="flatten analytes")

    def prepare(example):
        empty = {"mz": [], "log_intensity": []}
        if max_peaks is not None and len(example["mz"]) > max_peaks:
            return empty
        if len(example["mz"]) == 0:
            return empty
        try:
            mz, li = _process(processor, example["mz"], example["intensity"])
        except (ValueError, KeyError, TypeError):
            return empty
        return {"mz": mz, "log_intensity": li}

    rows = flat.map(prepare, remove_columns=["intensity"], num_proc=num_proc,
                    desc="preprocess spectra")
    before = len(rows)
    rows = rows.filter(lambda e: len(e["mz"]) > 0, num_proc=num_proc,
                       desc="drop oversized or invalid spectra")
    print(f"[grouped] {before_analytes:,} analytes, {excluded:,} excluded by peptide; "
          f"{before:,} spectra, {before - len(rows):,} dropped "
          f"({100 * (before - len(rows)) / max(before, 1):.1f}%, over max_peaks="
          f"{max_peaks} or invalid); {len(rows):,} kept", flush=True)
    return rows


def build_grouped_datasets(repo_id, processor, include_consensus: bool = False,
                           exclude_peptides: set[str] | None = None, num_proc=None,
                           splits=("train", "validation", "test")):
    """The corpus's own peptide-disjoint splits, flattened."""
    raw = load_dataset(repo_id)
    return {name: build_grouped_split(raw[name], processor, include_consensus,
                                      exclude_peptides, num_proc)
            for name in splits}


def group_ids(rows) -> np.ndarray:
    """Integer group per row, by peptide_key."""
    keys = [peptide_key(p, int(c)) for p, c in zip(rows["peptide"], rows["charge"])]
    return np.unique(np.array(keys), return_inverse=True)[1]


def load_spectrum_datasets(dataset_format: str, repo_id: str, processor, *,
                           include_consensus: bool = False,
                           exclude_peptides_from: str | None = None, num_proc=None,
                           validation_fraction: float = 0.1, seed: int = 0) -> dict:
    """train + validation, one spectrum per row, from the replicate or grouped corpus."""
    if dataset_format == "replicate":
        return build_alignment_datasets(repo_id, processor, num_proc=num_proc,
                                        validation_fraction=validation_fraction,
                                        seed=seed)
    if dataset_format != "grouped":
        raise ValueError(f"dataset_format must be replicate or grouped, "
                         f"not {dataset_format!r}")
    exclude = corpus_peptides(exclude_peptides_from) if exclude_peptides_from else set()
    return build_grouped_datasets(repo_id, processor, include_consensus=include_consensus,
                                  exclude_peptides=exclude, num_proc=num_proc,
                                  splits=("train", "validation"))
