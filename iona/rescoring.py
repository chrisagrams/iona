"""Hand-built PSM features: the spectrum alone, the candidate alone, and the two matched."""

from __future__ import annotations

import math

import numpy as np

from iona.chemistry import PROTON_MASS, RESIDUE_MASSES, WATER_MASS

FEATURE_NAMES = (
    # spectrum
    "n_peaks", "log_tic", "spectrum_entropy", "base_peak_fraction", "mz_range",
    # candidate
    "peptide_length", "charge", "n_modifications", "missed_cleavages",
    "basic_fraction", "gravy", "precursor_mz", "mass_error_ppm", "abs_mass_error_ppm",
    # match -- most of the signal
    "matched_peaks", "frag_coverage_b", "frag_coverage_y", "frag_coverage_all",
    "explained_intensity", "longest_b_run", "longest_y_run", "median_frag_error_ppm",
)

# Kyte-Doolittle hydrophobicity.
_GRAVY = {"A": 1.8, "R": -4.5, "N": -3.5, "D": -3.5, "C": 2.5, "Q": -3.5, "E": -3.5,
          "G": -0.4, "H": -3.2, "I": 4.5, "L": 3.8, "K": -3.9, "M": 1.9, "F": 2.8,
          "P": -1.6, "S": -0.8, "T": -0.7, "W": -0.9, "Y": -1.3, "V": 4.2}


def split_peptide(peptide: str) -> tuple[list[str], list[float]]:
    """Residues and their modification masses from a ProForma-ish string."""
    residues, mods, index = [], [], 0
    while index < len(peptide):
        char = peptide[index]
        if char == "[":
            close = peptide.index("]", index)
            if residues:
                mods[-1] += float(peptide[index + 1 : close])
            index = close + 1
            continue
        if char.isalpha():
            residues.append(char)
            mods.append(0.0)
        index += 1
    return residues, mods


def peptide_mass(residues: list[str], mods: list[float]) -> float:
    return (sum(RESIDUE_MASSES.get(r, 0.0) for r in residues) + sum(mods) + WATER_MASS)


def theoretical_fragments(residues: list[str], mods: list[float]) -> tuple[np.ndarray, np.ndarray]:
    """Singly-charged b and y ions. Returns (b_ions, y_ions)."""
    masses = [RESIDUE_MASSES.get(r, 0.0) + m for r, m in zip(residues, mods)]
    prefix = np.cumsum(masses[:-1]) + PROTON_MASS if len(masses) > 1 else np.array([])
    suffix = (np.cumsum(masses[::-1][:-1]) + WATER_MASS + PROTON_MASS
              if len(masses) > 1 else np.array([]))
    return prefix, suffix


def _match(observed: np.ndarray, theoretical: np.ndarray, tolerance_ppm: float,
           da_floor: float = 0.0) -> np.ndarray:
    """Which theoretical ions appear in the scan, within max(ppm window, da_floor Da)."""
    if theoretical.size == 0 or observed.size == 0:
        return np.zeros(theoretical.size, dtype=bool)
    index = np.searchsorted(observed, theoretical)
    found = np.zeros(theoretical.size, dtype=bool)
    for side in (0, -1):
        probe = np.clip(index + side, 0, observed.size - 1)
        diff = np.abs(observed[probe] - theoretical)
        found |= (diff / np.maximum(theoretical, 1e-9) * 1e6 <= tolerance_ppm) | (diff <= da_floor)
    return found


def _longest_run(found: np.ndarray) -> int:
    """The longest consecutive run of matched ions."""
    best = current = 0
    for hit in found:
        current = current + 1 if hit else 0
        best = max(best, current)
    return best


def spectrum_features(mz: np.ndarray, intensity: np.ndarray) -> dict[str, float]:
    total = float(intensity.sum())
    if intensity.size == 0 or total <= 0:
        return {"n_peaks": 0.0, "log_tic": 0.0, "spectrum_entropy": 0.0,
                "base_peak_fraction": 0.0, "mz_range": 0.0}
    share = intensity / total
    entropy = float(-(share * np.log(np.maximum(share, 1e-12))).sum())
    return {"n_peaks": float(mz.size), "log_tic": math.log1p(total),
            "spectrum_entropy": entropy,
            "base_peak_fraction": float(intensity.max() / total),
            "mz_range": float(mz.max() - mz.min())}


def candidate_features(peptide: str, precursor_mz: float, charge: int) -> dict[str, float]:
    residues, mods = split_peptide(peptide)
    if not residues:
        return {k: 0.0 for k in FEATURE_NAMES[5:14]}
    neutral = peptide_mass(residues, mods)
    charge = max(int(charge), 1)
    theoretical_mz = (neutral + charge * PROTON_MASS) / charge
    error = (precursor_mz - theoretical_mz) / max(theoretical_mz, 1e-9) * 1e6
    basic = sum(1 for r in residues if r in "KRH") / len(residues)
    # Trypsin cuts after K/R except before P; an internal K/R is a missed cleavage.
    missed = sum(1 for i, r in enumerate(residues[:-1])
                 if r in "KR" and residues[i + 1] != "P")
    return {"peptide_length": float(len(residues)), "charge": float(charge),
            "n_modifications": float(sum(1 for m in mods if abs(m) > 1e-6)),
            "missed_cleavages": float(missed), "basic_fraction": basic,
            "gravy": float(np.mean([_GRAVY.get(r, 0.0) for r in residues])),
            "precursor_mz": float(precursor_mz), "mass_error_ppm": float(error),
            "abs_mass_error_ppm": abs(float(error))}


def match_features(peptide: str, mz: np.ndarray, intensity: np.ndarray,
                   tolerance_ppm: float = 20.0, da_floor: float = 0.0) -> dict[str, float]:
    residues, mods = split_peptide(peptide)
    keys = ("matched_peaks", "frag_coverage_b", "frag_coverage_y", "frag_coverage_all",
            "explained_intensity", "longest_b_run", "longest_y_run",
            "median_frag_error_ppm")
    if len(residues) < 2 or mz.size == 0:
        return dict.fromkeys(keys, 0.0)
    order = np.argsort(mz)
    observed, weights = mz[order], intensity[order]
    b_ions, y_ions = theoretical_fragments(residues, mods)
    b_found, y_found = (_match(observed, b_ions, tolerance_ppm, da_floor),
                        _match(observed, y_ions, tolerance_ppm, da_floor))

    matched_mz, errors = [], []
    for ions, found in ((b_ions, b_found), (y_ions, y_found)):
        for ion in ions[found]:
            nearest = int(np.argmin(np.abs(observed - ion)))
            matched_mz.append(nearest)
            errors.append((observed[nearest] - ion) / ion * 1e6)
    total = float(weights.sum())
    explained = float(weights[list(set(matched_mz))].sum() / total) if total > 0 and matched_mz else 0.0
    n = len(residues) - 1
    return {"matched_peaks": float(len(matched_mz)),
            "frag_coverage_b": float(b_found.sum()) / n,
            "frag_coverage_y": float(y_found.sum()) / n,
            "frag_coverage_all": float(b_found.sum() + y_found.sum()) / (2 * n),
            "explained_intensity": explained,
            "longest_b_run": float(_longest_run(b_found)),
            "longest_y_run": float(_longest_run(y_found)),
            "median_frag_error_ppm": float(np.median(np.abs(errors))) if errors else 0.0}


def extract_features(peptide: str, mz: np.ndarray, intensity: np.ndarray,
                     precursor_mz: float, charge: int,
                     tolerance_ppm: float = 20.0, da_floor: float = 0.0) -> np.ndarray:
    values = {**spectrum_features(mz, intensity),
              **candidate_features(peptide, precursor_mz, charge),
              **match_features(peptide, mz, intensity, tolerance_ppm, da_floor)}
    return np.array([values[name] for name in FEATURE_NAMES], dtype=np.float32)
