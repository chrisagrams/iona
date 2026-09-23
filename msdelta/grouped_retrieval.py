"""chrisagrams/ms-contrastive-100k as one spectrum per row, for contrastive eval and training.

The corpus ships ONE ROW PER ANALYTE (peptide+charge): a consensus spectrum plus exactly
three experimental replicates, in train (100,000 analytes), validation (10,000) and test
(10,000) splits that share no peptide. Our contrastive trainer, sampler and retrieval
metrics all expect one spectrum per row carrying `peptide` and `charge`, so each analyte
is flattened into up to four rows here and grouped back by peptide_key downstream.

Two choices are exposed rather than fixed, because each changes what a number means:

  include_consensus. A consensus spectrum is built FROM replicates, so it may sit
  artificially close to them. Eval reports both variants from one embedding pass (see
  eval_grouped_retrieval); training defaults to experimental-only so a positive pair is
  never one spectrum and its own average.

  exclude_peptides. Every contrastive model so far trained on
  ms2-peptide-replicate-retrieval, and 40 of this corpus's 9,100 test peptides are in
  it. Scoring those models here without removing them would reward memorisation.
  Excluded by PEPTIDE (any charge), which is the conservative reading.

Spectra above max_peaks are DROPPED, not truncated, exactly as build_alignment_datasets
does and for the same reason; the count is printed, never swallowed.
"""

from __future__ import annotations

import numpy as np
import torch

GROUPED_REPO = "chrisagrams/ms-contrastive-100k"
REPLICATE_REPO = "chrisagrams/ms2-peptide-replicate-retrieval"


def replicate_corpus_peptides(repo_id: str = REPLICATE_REPO) -> set[str]:
    """Every peptide in the replicate corpus, across all of its splits."""
    from datasets import load_dataset

    raw = load_dataset(repo_id)
    return {p for split in raw.values() for p in split["peptide"]}


def flatten_analyte(example, include_consensus: bool) -> dict[str, list]:
    """One analyte row -> parallel lists, one entry per spectrum. Pure; unit-tested."""
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
    """The corpus's OWN splits, flattened. Never re-split: they are peptide-disjoint."""
    from datasets import load_dataset

    raw = load_dataset(repo_id)
    return {name: build_grouped_split(raw[name], processor, include_consensus,
                                      exclude_peptides, num_proc)
            for name in splits}


def group_ids(rows) -> np.ndarray:
    """Integer group per row, by peptide_key -- the same identity every metric uses."""
    from msdelta.reranking import peptide_key

    keys = [peptide_key(p, int(c)) for p, c in zip(rows["peptide"], rows["charge"])]
    return np.unique(np.array(keys), return_inverse=True)[1]


def load_spectrum_datasets(dataset_format: str, repo_id: str, processor, *,
                           include_consensus: bool = False,
                           exclude_replicate_peptides: bool = True, num_proc=None,
                           validation_fraction: float = 0.1, seed: int = 0) -> dict:
    """train + validation, one spectrum per row with peptide/charge/precursor, for either
    corpus. Shared by contrastive training, the alignment teacher cache and the student,
    so all three see the same rows for the same flags.

      replicate  ms2-peptide-replicate-retrieval, re-split here by peptide (seed,
                 validation_fraction), exactly as before.
      grouped    ms-contrastive-100k's own peptide-disjoint splits, flattened; the
                 split flags are ignored because nothing is re-split.
    """
    if dataset_format == "replicate":
        from msdelta.reranking import build_alignment_datasets
        return build_alignment_datasets(repo_id, processor, num_proc=num_proc,
                                        validation_fraction=validation_fraction,
                                        seed=seed)
    if dataset_format != "grouped":
        raise ValueError(f"dataset_format must be replicate or grouped, "
                         f"not {dataset_format!r}")
    exclude = replicate_corpus_peptides() if exclude_replicate_peptides else set()
    return build_grouped_datasets(repo_id, processor, include_consensus=include_consensus,
                                  exclude_peptides=exclude, num_proc=num_proc,
                                  splits=("train", "validation"))
