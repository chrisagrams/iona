"""Deterministic synthetic corpora for the end-to-end tests. Nothing is stored on disk
in the repo: every fixture is regenerated from a seed, in the layout the real entry
points read, so the tests drive the same loaders the production runs do.

Spectra are built from the peptide's own b/y ions (residue masses from
msdelta.data.chemistry), jittered per replicate, plus random noise peaks. That gives
replicates of one analyte something genuinely in common, so contrastive and retrieval
numbers are not pure noise -- but the tests only assert plumbing, never quality.

Layouts produced
  pretraining  DatasetDict saved with save_to_disk, already preprocessed
               (--preprocessed_dataset_dir of msdelta.pretraining.train)
  grouped      a directory of train/validation/test parquet files, one ANALYTE per row
               (peptide, charge, precursor, analyte_id, consensus{mz,intensity},
               experimental[3 x {mz,intensity}]) -- the ms-contrastive-100k schema;
               `datasets.load_dataset(<dir>)` maps the file names to splits, so the
               entry points take the directory as --dataset_repo / --repo unchanged
  denoise      a directory of train/validation/test parquet files with mz, intensity,
               noise (bool per peak) -- the ms-denoise-100k schema
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

AMINO = "ACDEFGHIKLMNPQRSTVWY"
PROTON = 1.007276
WATER = 18.010565


def _residue_masses():
    from msdelta.data.chemistry import RESIDUE_MASSES
    return RESIDUE_MASSES


def peptides(n: int, seed: int = 0, min_len: int = 7, max_len: int = 14) -> list[str]:
    """n distinct tryptic-looking peptides (end in K/R); some carry a fixed-mod cysteine."""
    rng = np.random.default_rng(seed)
    out: list[str] = []
    seen = set()
    while len(out) < n:
        length = int(rng.integers(min_len, max_len + 1))
        body = "".join(rng.choice(list(AMINO.replace("K", "").replace("R", "")), length - 1))
        pep = body + str(rng.choice(["K", "R"]))
        if pep in seen:
            continue
        seen.add(pep)
        out.append(pep.replace("C", "C[57.0215]"))
    return out


def _plain(peptide: str) -> str:
    import re
    return re.sub(r"\[[^\]]*\]", "", peptide)


def neutral_mass(peptide: str) -> float:
    masses = _residue_masses()
    mod = 57.0215 * peptide.count("C[57.0215]")
    return sum(masses[a] for a in _plain(peptide)) + WATER + mod


def fragment_ions(peptide: str) -> np.ndarray:
    """Singly charged b and y ions."""
    masses = _residue_masses()
    seq = _plain(peptide)
    res = np.array([masses[a] + (57.0215 if a == "C" else 0.0) for a in seq])
    b = np.cumsum(res)[:-1] + PROTON
    y = np.cumsum(res[::-1])[:-1] + WATER + PROTON
    return np.concatenate([b, y])


def spectrum(peptide: str, rng, n_noise: int = 12, jitter_ppm: float = 10.0,
             drop: float = 0.15, max_mz: float = 1800.0):
    """(mz, intensity, is_noise) sorted by m/z; noise peaks are uniform in m/z."""
    ions = fragment_ions(peptide)
    ions = ions[(ions > 60) & (ions < max_mz)]
    keep = rng.random(len(ions)) > drop
    ions = ions[keep] if keep.any() else ions
    signal_mz = ions * (1 + rng.normal(0, jitter_ppm * 1e-6, len(ions)))
    signal_int = rng.uniform(0.2, 1.0, len(ions)) * 1e4
    noise_mz = rng.uniform(100, max_mz, n_noise)
    noise_int = rng.uniform(0.01, 0.3, n_noise) * 1e4
    mz = np.concatenate([signal_mz, noise_mz])
    intensity = np.concatenate([signal_int, noise_int])
    noise = np.concatenate([np.zeros(len(signal_mz), bool), np.ones(n_noise, bool)])
    order = np.argsort(mz)
    return mz[order].tolist(), intensity[order].tolist(), noise[order].tolist()


def pretraining_dataset(out_dir: Path, processor, n_train: int = 64, n_val: int = 16,
                        seed: int = 0) -> Path:
    """Preprocessed DatasetDict for msdelta.pretraining.train --preprocessed_dataset_dir."""
    from datasets import Dataset, DatasetDict

    from msdelta.data.data import build_preprocessed_dataset

    rng = np.random.default_rng(seed)
    peps = peptides(n_train + n_val, seed=seed)

    def raw(sub):
        rows = {"mz": [], "intensity": [], "peptide": [], "charge": []}
        for p in sub:
            mz, inten, _ = spectrum(p, rng)
            rows["mz"].append(mz); rows["intensity"].append(inten)
            rows["peptide"].append(p); rows["charge"].append(int(rng.integers(2, 4)))
        return Dataset.from_dict(rows)

    dd = DatasetDict({
        "train": build_preprocessed_dataset(raw(peps[:n_train]), processor, num_proc=None),
        "validation": build_preprocessed_dataset(raw(peps[n_train:]), processor, num_proc=None),
    })
    dd.save_to_disk(str(out_dir))
    return out_dir


def _write_parquet(table: dict, path: Path) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.table(table), path)


def grouped_corpus(out_dir: Path, sizes=(("train", 30), ("validation", 10), ("test", 12)),
                   seed: int = 1) -> Path:
    """ms-contrastive-100k-shaped parquet splits; peptide-disjoint across splits."""
    rng = np.random.default_rng(seed)
    total = sum(n for _, n in sizes)
    peps = peptides(total, seed=seed)
    start = 0
    for split, n in sizes:
        rows = {"analyte_id": [], "peptide": [], "charge": [], "precursor": [],
                "consensus": [], "experimental": []}
        for i, p in enumerate(peps[start:start + n]):
            z = int(rng.integers(2, 4))
            reps = []
            for _ in range(3):
                mz, inten, _ = spectrum(p, rng)
                reps.append({"mz": mz, "intensity": inten})
            cmz, cint, _ = spectrum(p, rng, n_noise=4, jitter_ppm=2.0, drop=0.0)
            rows["analyte_id"].append(f"{split}-{i}")
            rows["peptide"].append(p)
            rows["charge"].append(z)
            rows["precursor"].append((neutral_mass(p) + z * PROTON) / z)
            rows["consensus"].append({"mz": cmz, "intensity": cint})
            rows["experimental"].append(reps)
        _write_parquet(rows, out_dir / f"{split}.parquet")
        start += n
    return out_dir


def denoise_corpus(out_dir: Path, sizes=(("train", 48), ("validation", 16), ("test", 16)),
                   seed: int = 2) -> Path:
    """ms-denoise-100k-shaped parquet splits: mz, intensity, noise."""
    rng = np.random.default_rng(seed)
    peps = peptides(sum(n for _, n in sizes), seed=seed)
    start = 0
    for split, n in sizes:
        rows = {"mz": [], "intensity": [], "noise": []}
        for p in peps[start:start + n]:
            mz, inten, noise = spectrum(p, rng, n_noise=int(rng.integers(6, 20)))
            rows["mz"].append(mz); rows["intensity"].append(inten); rows["noise"].append(noise)
        _write_parquet(rows, out_dir / f"{split}.parquet")
        start += n
    return out_dir


def tiny_model_dir(out_dir: Path, max_peaks: int = 64) -> Path:
    """config.json + preprocessor_config.json for a 2-layer, hidden-32 MSDelta."""
    from msdelta.models.configuration_msdelta import MSDeltaConfig
    from msdelta.models.processing_msdelta import MSDeltaProcessor

    MSDeltaConfig(hidden_size=32, num_attention_heads=4, num_hidden_layers=2,
                  intermediate_size=64, delta_bias_n_freqs=8,
                  delta_bias_per_head_hidden=4).save_pretrained(str(out_dir))
    MSDeltaProcessor(max_peaks=max_peaks).save_pretrained(str(out_dir))
    return out_dir
