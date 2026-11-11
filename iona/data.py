"""Load, preprocess, and collate mass spectrum data."""

from __future__ import annotations

import re
from functools import partial

import numpy as np
import torch
from datasets import Dataset, DatasetDict, load_dataset, load_from_disk

from iona.chemistry import PROTON_MASS, RESIDUE_MASSES, WATER_MASS
from iona.processing_iona import IonaProcessor

N_CHARGES = 8

_MOD_RE = re.compile(r"\[([+-]?[0-9.]+)\]")


def charge_index(peptide_charge: str | None) -> int:
    """Return the charge index from a peptide-charge label."""
    if not peptide_charge:
        return 0
    parts = str(peptide_charge).rsplit("_", 1)
    if len(parts) == 2 and parts[1].isdigit():
        return min(int(parts[1]), N_CHARGES - 1)
    return 0


def precursor_mz(peptide_charge: str | None) -> float:
    """Calculate precursor m/z from a peptide-charge label."""
    if not peptide_charge:
        return 0.0
    parts = str(peptide_charge).rsplit("_", 1)
    if len(parts) != 2 or not parts[1].isdigit():
        return 0.0
    pep, z = parts[0], int(parts[1])
    if z < 1:
        return 0.0
    mods = sum(float(x) for x in _MOD_RE.findall(pep))
    seq = _MOD_RE.sub("", pep)
    mass = WATER_MASS + mods
    for a in seq:
        m = RESIDUE_MASSES.get(a)
        if m is None:
            return 0.0
        mass += m
    return (mass + z * PROTON_MASS) / z


def _preprocess_example(example: dict, processor: IonaProcessor) -> dict:
    """Convert one raw dataset row to preprocessed values."""
    intensity = torch.tensor(example["intensity"], dtype=torch.float32)
    try:
        values = processor(example["mz"], example["intensity"], padding=False, return_labels=True)
    except ValueError:
        values = {"mz": [], "log_intensity": [], "labels": []}
    pc = f"{example['peptide']}_{example['charge']}"
    return {
        "mz": values["mz"],
        "log_intensity": values["log_intensity"],
        "labels": values["labels"],
        "length": len(values["mz"]),
        "charge": charge_index(pc),
        "precursor_mz": precursor_mz(pc),
        "log_tic": float(torch.log1p(intensity.sum())) if intensity.numel() else 0.0,
        "peptide_charge": pc if pc is not None else "",
    }


def collate_preprocessed(features: list[dict]) -> dict[str, torch.Tensor]:
    """Pad preprocessed rows without masking."""
    B = len(features)
    Ks = [len(f["mz"]) for f in features]
    K_max = max(max(Ks) if Ks else 1, 1)
    mz = torch.zeros(B, K_max, dtype=torch.float32)
    log_intensity = torch.zeros(B, K_max, dtype=torch.float32)
    labels = torch.zeros(B, K_max, dtype=torch.float32)
    attention_mask = torch.zeros(B, K_max, dtype=torch.long)
    for b, (f, K) in enumerate(zip(features, Ks)):
        if K == 0:
            continue
        mz[b, :K] = torch.as_tensor(f["mz"], dtype=torch.float32)
        log_intensity[b, :K] = torch.as_tensor(f["log_intensity"], dtype=torch.float32)
        labels[b, :K] = torch.as_tensor(f["labels"], dtype=torch.float32)
        attention_mask[b, :K] = 1
    return {
        "mz": mz,
        "log_intensity": log_intensity,
        "labels": labels,
        "attention_mask": attention_mask,
    }


def build_preprocessed_dataset(
    dataset: Dataset,
    processor: IonaProcessor,
    num_proc: int | None = None,
    load_from_cache_file: bool | None = None,
) -> Dataset:
    """Preprocess a mass-spectrum dataset."""
    dataset = dataset.map(
        partial(_preprocess_example, processor=processor),
        remove_columns=dataset.column_names,
        num_proc=num_proc,
        load_from_cache_file=load_from_cache_file,
        desc="preprocess spectra",
    )
    return dataset.filter(
        lambda ex: len(ex["mz"]) > 0,
        num_proc=num_proc,
        load_from_cache_file=load_from_cache_file,
        desc="drop empty spectra",
    )


def build_pretraining_datasets(
    repo_id: str,
    processor: IonaProcessor,
    train_split: str = "train",
    validation_split: str = "validation",
    num_proc: int | None = None,
    cache_dir: str | None = None,
    load_from_cache_file: bool | None = None,
) -> tuple[Dataset, Dataset]:
    """Load and preprocess training and validation splits from Hugging Face."""
    train, validation = load_dataset(
        repo_id,
        split=[train_split, validation_split],
        cache_dir=cache_dir,
    )
    train = build_preprocessed_dataset(
        train,
        processor,
        num_proc=num_proc,
        load_from_cache_file=load_from_cache_file,
    )
    validation = build_preprocessed_dataset(
        validation,
        processor,
        num_proc=num_proc,
        load_from_cache_file=load_from_cache_file,
    )
    return train, validation


def load_pretraining_datasets_from_disk(
    dataset_path: str,
    train_split: str = "train",
    validation_split: str = "validation",
) -> tuple[Dataset, Dataset]:
    """Load finalized preprocessed splits from disk."""
    datasets = load_from_disk(dataset_path)
    if not isinstance(datasets, DatasetDict):
        raise TypeError(f"Expected a DatasetDict at {dataset_path}")
    return datasets[train_split], datasets[validation_split]


def subsample_train(dataset: Dataset, fraction: float, seed: int = 0) -> Dataset:
    """Select a random subset; smaller fractions are nested in larger ones."""
    if fraction >= 1.0:
        return dataset
    n = max(1, int(len(dataset) * fraction))
    order = np.random.default_rng(seed).permutation(len(dataset))
    # Sorted indices keep reads from the Arrow table close to sequential.
    return dataset.select(np.sort(order[:n]))


def build_denoising_datasets(
    repo_id: str,
    processor: IonaProcessor,
    num_proc: int | None = None,
) -> DatasetDict:
    """Load and preprocess the labeled signal/noise dataset."""
    datasets = load_dataset(repo_id)
    datasets = datasets.filter(
        lambda example: 0 < len(example["mz"]) <= processor.max_peaks,
        num_proc=num_proc,
        desc="drop invalid or oversized denoising spectra",
    )
    return datasets.map(
        lambda example: processor.process_denoising_example(
            example["mz"], example["intensity"], example["noise"]
        ),
        remove_columns=datasets["train"].column_names,
        num_proc=num_proc,
        desc="preprocess denoising spectra",
    )


def build_retrieval_datasets(
    repo_id: str,
    processor: IonaProcessor,
    num_proc: int | None = None,
) -> DatasetDict:
    """Load and preprocess grouped consensus/experimental retrieval spectra."""
    datasets = load_dataset(repo_id)
    datasets = DatasetDict({split: datasets[split] for split in ("train", "validation")})
    datasets = datasets.filter(
        lambda example: all(
            0 < len(spectrum["mz"]) <= processor.max_peaks
            for spectrum in (example["consensus"], *example["experimental"])
        ),
        num_proc=num_proc,
        desc="drop invalid or oversized retrieval groups",
    )

    processed = datasets.map(
        lambda example: processor.process_retrieval_example(
            example["consensus"], example["experimental"]
        ),
        remove_columns=datasets["train"].column_names,
        num_proc=num_proc,
        desc="preprocess retrieval spectra",
    )
    return processed.filter(
        lambda example: len(example["mz"]) == 4 and all(example["mz"]),
        num_proc=num_proc,
        desc="drop invalid retrieval groups",
    )


def build_retrieval_evaluation_datasets(
    validation_dataset: Dataset,
    processor: IonaProcessor,
    *,
    max_analytes: int = 1000,
    replicate_repo_id: str | None = None,
    num_proc: int | None = None,
) -> dict[str, Dataset]:
    """Prepare named retrieval galleries with one spectrum and global label per row."""
    subset = validation_dataset.select(range(min(len(validation_dataset), max_analytes)))
    evaluation_datasets = {
        "retrieval": Dataset.from_list(
            [
                {"mz": mz, "log_intensity": intensity, "retrieval_labels": group_id}
                for group_id, row in enumerate(subset)
                for mz, intensity in zip(row["mz"], row["log_intensity"])
            ]
        ),
    }
    if replicate_repo_id:
        dataset = load_dataset(replicate_repo_id, split="test")
        dataset = dataset.filter(
            lambda row: 0 < len(row["mz"]) <= processor.max_peaks,
            num_proc=num_proc,
            desc="drop invalid or oversized replicate spectra",
        )
        analytes = {}
        labels = [
            analytes.setdefault((peptide, charge), len(analytes))
            for peptide, charge in zip(dataset["peptide"], dataset["charge"])
        ]
        processed = dataset.map(
            lambda row: processor(row["mz"], row["intensity"], padding=False),
            remove_columns=dataset.column_names,
            num_proc=num_proc,
            desc="preprocess replicate retrieval spectra",
        )
        evaluation_datasets["replicate_retrieval"] = processed.add_column(
            "retrieval_labels",
            labels,
        )
    return evaluation_datasets


# Fine-tuning spectra: replicate pairs and grouped analytes, one spectrum per row.


def peptide_key(peptide: str, charge: int) -> str:
    """Identity used to decide whether two spectra are the same peptide and charge."""
    return f"{peptide}_{charge}"


def build_alignment_datasets(repo_id, processor, num_proc=None, validation_fraction=0.1,
                             seed=0):
    """Spectrum/sequence pairs, split by peptide."""
    raw = load_dataset(repo_id)
    split = "train" if "train" in raw else list(raw)[0]

    # Spectra above max_peaks are dropped, not truncated, and the count is printed.
    max_peaks = processor.max_peaks

    def prepare(example):
        if len(example["mz"]) > max_peaks:
            return {"mz": [], "log_intensity": [], "peptide": "", "charge": 0, "precursor": 0.0}
        try:
            values = processor(
                torch.as_tensor(example["mz"], dtype=torch.float32),
                torch.as_tensor(example["intensity"], dtype=torch.float32),
                padding=False,
            )
        except (ValueError, KeyError, TypeError):
            return {"mz": [], "log_intensity": [], "peptide": "", "charge": 0, "precursor": 0.0}
        return {
            "mz": values["mz"][0] if values["mz"] and isinstance(values["mz"][0], list)
                  else values["mz"],
            "log_intensity": (values["log_intensity"][0]
                              if values["log_intensity"]
                              and isinstance(values["log_intensity"][0], list)
                              else values["log_intensity"]),
            "peptide": example.get("peptide") or "",
            "charge": int(example.get("charge") or 0),
            # Measured precursor m/z, used by the rescorer.
            "precursor": float(example.get("precursor") or 0.0),
        }

    rows = raw[split].map(prepare, remove_columns=raw[split].column_names,
                          num_proc=num_proc, desc="preprocess alignment pairs")
    before = len(rows)
    rows = rows.filter(lambda e: len(e["mz"]) > 0 and bool(e["peptide"]),
                       num_proc=num_proc, desc="drop empty pairs")
    dropped = before - len(rows)
    if dropped:
        oversized = sum(1 for n in raw[split]["mz"] if len(n) > max_peaks)
        print(f"[alignment] dropped {dropped:,} of {before:,} pairs "
              f"({100 * dropped / before:.1f}%); {oversized:,} were over max_peaks="
              f"{max_peaks}. Peak count tracks charge and peptide length, so this is a "
              f"biased loss, not a random one.", flush=True)

    peptides = sorted(set(rows["peptide"]))
    generator = np.random.default_rng(seed)
    held_out = set(generator.choice(
        peptides, size=max(1, int(len(peptides) * validation_fraction)), replace=False
    ).tolist())
    return {
        "train": rows.filter(lambda e: e["peptide"] not in held_out, desc="train split"),
        "validation": rows.filter(lambda e: e["peptide"] in held_out, desc="validation split"),
    }


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


# De novo sequencing: one spectrum and its peptide per row, peptides in MassIVE-KB notation.

# Nine-species modification spellings and their MassIVE-KB equivalents.
NINESPECIES_MODIFICATIONS = {
    "C(+57.02)": "C[+57.021]",
    "M(+15.99)": "M[+15.995]",
    "N(+.98)": "N[+0.984]",
    "Q(+.98)": "Q[+0.984]",
}


def ninespecies_to_massive_kb(peptide: str) -> str:
    """Rewrite a nine-species modified sequence in MassIVE-KB notation."""
    for theirs, ours in NINESPECIES_MODIFICATIONS.items():
        peptide = peptide.replace(theirs, ours)
    if "(" in peptide:
        raise ValueError(f"unmapped nine-species modification in {peptide!r}")
    return peptide


def top_peaks(mz, intensity, max_peaks: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Keep the max_peaks most intense peaks, in their original order."""
    mz = torch.as_tensor(np.asarray(mz, dtype=np.float32))
    intensity = torch.as_tensor(np.asarray(intensity, dtype=np.float32))
    if mz.numel() > max_peaks:
        keep = intensity.topk(max_peaks).indices.sort().values
        mz, intensity = mz[keep], intensity[keep]
    return mz, intensity


def preprocess_denovo_batch(
    mzs: list,
    intensities: list,
    peptides: list[str],
    charges: list[int],
    precursor_mzs: list[float] | None,
    *,
    processor: IonaProcessor,
    tokenizer,
    max_peptide_length: int,
    max_charge: int,
) -> dict[str, list]:
    """Process spectra and tokenize peptides, raising on anything the model cannot represent.

    Without measured precursor m/z values, the theoretical m/z is computed from the peptide.
    """
    encoded = tokenizer(peptides)["input_ids"]
    masses = tokenizer.token_masses
    out: dict[str, list] = {key: [] for key in (
        "mz", "log_intensity", "labels", "peptide", "precursor_mz", "precursor_charge", "length"
    )}
    for row, (peptide, ids) in enumerate(zip(peptides, encoded)):
        if tokenizer.unk_token_id in ids:
            raise ValueError(f"unknown token in {peptide!r}")
        residues = ids[:-1]
        if len(residues) > max_peptide_length:
            raise ValueError(f"{len(residues)} tokens exceeds max_peptide_length="
                             f"{max_peptide_length}: {peptide!r}")
        charge = int(charges[row])
        if not 1 <= charge <= max_charge:
            raise ValueError(f"precursor charge {charge} is outside [1, {max_charge}]: {peptide!r}")
        if precursor_mzs is None:
            neutral = sum(masses[index] for index in residues) + WATER_MASS
            precursor = (neutral + charge * PROTON_MASS) / charge
        else:
            precursor = float(precursor_mzs[row])

        mz, intensity = top_peaks(mzs[row], intensities[row], processor.max_peaks)
        values = processor(mz, intensity, padding=False)
        out["mz"].append(values["mz"])
        out["log_intensity"].append(values["log_intensity"])
        out["labels"].append(ids)
        out["peptide"].append(peptide)
        out["precursor_mz"].append(precursor)
        out["precursor_charge"].append(charge)
        out["length"].append(len(values["mz"]))
    return out


def build_massive_kb_denovo_split(split: Dataset, processor: IonaProcessor, tokenizer, *,
                                  max_peptide_length: int, max_charge: int,
                                  num_proc: int | None = None) -> Dataset:
    """Preprocess a MassIVE-KB split; precursor m/z is theoretical, since none is stored."""

    def prepare(batch):
        peptides, charges = zip(*(value.rsplit("_", 1) for value in batch["peptide_charge"]))
        return preprocess_denovo_batch(
            batch["m/z"], batch["int"], list(peptides), [int(c) for c in charges], None,
            processor=processor, tokenizer=tokenizer, max_peptide_length=max_peptide_length,
            max_charge=max_charge,
        )

    return split.map(prepare, batched=True, remove_columns=split.column_names,
                     num_proc=num_proc, desc="preprocess MassIVE-KB de novo")


def build_ninespecies_denovo_split(split: Dataset, processor: IonaProcessor, tokenizer, *,
                                   max_peptide_length: int, max_charge: int,
                                   num_proc: int | None = None) -> Dataset:
    """Preprocess a nine-species split, keeping its measured precursor m/z."""

    def prepare(batch):
        return preprocess_denovo_batch(
            batch["mz_array"], batch["intensity_array"],
            [ninespecies_to_massive_kb(p) for p in batch["modified_sequence"]],
            batch["precursor_charge"], batch["precursor_mz"],
            processor=processor, tokenizer=tokenizer, max_peptide_length=max_peptide_length,
            max_charge=max_charge,
        )

    return split.map(prepare, batched=True, remove_columns=split.column_names,
                     num_proc=num_proc, desc="preprocess nine-species de novo")
