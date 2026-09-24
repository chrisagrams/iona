"""Percolator-style rescoring: features + a small classifier, ranked by the logit.

Why this and not embedding retrieval. The alignment tower ranks candidates by the cosine
between a spectrum embedding and a sequence embedding, and that single number reaches
AUROC 0.846 on true-vs-decoy pairs -- real signal, but it compresses a whole spectrum to
1280 numbers and asks whether two vectors are close. Fragment matching asks the physical
question instead: do the b and y ions this peptide would produce actually appear in this
scan. Rescoring on hand-built features has worked for two decades, so the embedding has
to EARN its place against that baseline rather than be assumed to replace it. Keeping
both in one feature vector makes the comparison a flag (`drop_embedding`) rather than a
separate experiment.

Three feature groups:

  spectrum   the scan alone -- is it even interpretable? A sparse or low-TIC scan should
             not yield a confident identification whatever the sequence.
  candidate  the peptide alone, plus precursor mass error, which is the strongest and
             cheapest classical feature: a candidate 50 ppm off the precursor is wrong
             however good its fragments look.
  match      the peptide against the scan. This carries most of the signal.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field

import numpy as np

from msdelta.chemistry import PROTON_MASS, RESIDUE_MASSES, WATER_MASS

FEATURE_NAMES = (
    # spectrum
    "n_peaks", "log_tic", "spectrum_entropy", "base_peak_fraction", "mz_range",
    # candidate
    "peptide_length", "charge", "n_modifications", "missed_cleavages",
    "basic_fraction", "gravy", "precursor_mz", "mass_error_ppm", "abs_mass_error_ppm",
    # match -- most of the signal
    "matched_peaks", "frag_coverage_b", "frag_coverage_y", "frag_coverage_all",
    "explained_intensity", "longest_b_run", "longest_y_run", "median_frag_error_ppm",
    # the model's contribution, one number among the rest
    "embedding_cosine",
)

# Kyte-Doolittle, for the hydrophobicity average. Retention time correlates with it, and
# a candidate whose hydrophobicity disagrees with when it eluted is suspect.
_GRAVY = {"A": 1.8, "R": -4.5, "N": -3.5, "D": -3.5, "C": 2.5, "Q": -3.5, "E": -3.5,
          "G": -0.4, "H": -3.2, "I": 4.5, "L": 3.8, "K": -3.9, "M": 1.9, "F": 2.8,
          "P": -1.6, "S": -0.8, "T": -0.7, "W": -0.9, "Y": -1.3, "V": 4.2}
_MOD = re.compile(r"\[([+-]?\d+\.?\d*)\]")


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
    """Singly-charged b and y ions. Returns (b_ions, y_ions).

    Only b/y: they dominate HCD spectra, and adding a/c/x/z series would raise the
    match rate for TRUE and DECOY candidates alike, which costs discrimination.
    """
    masses = [RESIDUE_MASSES.get(r, 0.0) + m for r, m in zip(residues, mods)]
    prefix = np.cumsum(masses[:-1]) + PROTON_MASS if len(masses) > 1 else np.array([])
    suffix = (np.cumsum(masses[::-1][:-1]) + WATER_MASS + PROTON_MASS
              if len(masses) > 1 else np.array([]))
    return prefix, suffix


def _match(observed: np.ndarray, theoretical: np.ndarray, tolerance_ppm: float,
           da_floor: float = 0.0) -> np.ndarray:
    """Which theoretical ions appear in the scan, within max(ppm window, da_floor Da).

    da_floor 0 (default) is the original ppm-only behaviour. Low-resolution ion-trap MS2
    needs a Da floor: the psm-rerank dataset card specifies max(250 ppm, 0.05 Da)."""
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
    """The longest consecutive ion series. A real identification produces runs.

    Six scattered matches across a 20-residue peptide is what a wrong candidate looks
    like; six consecutive ones is evidence, and a plain count cannot tell them apart.
    """
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
    # Shannon entropy: a proxy for how many peaks carry real signal, which a raw count
    # misses when one peak dominates.
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
                     embedding_cosine: float = 0.0,
                     tolerance_ppm: float = 20.0, da_floor: float = 0.0) -> np.ndarray:
    values = {**spectrum_features(mz, intensity),
              **candidate_features(peptide, precursor_mz, charge),
              **match_features(peptide, mz, intensity, tolerance_ppm, da_floor),
              "embedding_cosine": float(embedding_cosine)}
    return np.array([values[name] for name in FEATURE_NAMES], dtype=np.float32)


# ------------------------------------------------------------------ decoys

def pseudo_reverse(peptide: str) -> str:
    """Reverse the sequence, keeping the C-terminal residue fixed.

    The standard proteomics decoy. Holding the last residue preserves the tryptic K/R
    terminus and the precursor mass exactly, so the decoy is wrong in its FRAGMENTS and
    nowhere else -- which is the only thing we want the classifier to learn to see.
    """
    residues, mods = split_peptide(peptide)
    if len(residues) < 3:
        return peptide
    order = list(range(len(residues) - 2, -1, -1)) + [len(residues) - 1]
    return "".join(residues[i] + (f"[{mods[i]:+.4f}]" if abs(mods[i]) > 1e-6 else "")
                   for i in order)


def near_miss(peptide: str, rng: np.random.Generator) -> str:
    """Swap two adjacent residues: same mass, nearly the same fragments.

    Much harder than a reversal. It is also the realistic confusion -- a search engine's
    runner-up is usually a near-anagram of the truth, not a random sequence.
    """
    residues, mods = split_peptide(peptide)
    if len(residues) < 4:
        return peptide
    i = int(rng.integers(0, len(residues) - 2))
    order = list(range(len(residues)))
    order[i], order[i + 1] = order[i + 1], order[i]
    return "".join(residues[j] + (f"[{mods[j]:+.4f}]" if abs(mods[j]) > 1e-6 else "")
                   for j in order)


def il_equivalent(peptide: str) -> str | None:
    """Swap I and L. Identical mass, identical fragments -- undecidable by MS alone.

    Included as a control: a classifier that claims to separate these is overfitting,
    since no feature here can distinguish them even in principle.
    """
    residues, mods = split_peptide(peptide)
    if not any(r in "IL" for r in residues):
        return None
    swapped = ["L" if r == "I" else "I" if r == "L" else r for r in residues]
    return "".join(s + (f"[{m:+.4f}]" if abs(m) > 1e-6 else "")
                   for s, m in zip(swapped, mods))


@dataclass
class CandidateSet:
    """One spectrum's candidates: the truth, plus decoys of decreasing difficulty."""

    peptides: list[str] = field(default_factory=list)
    labels: list[int] = field(default_factory=list)
    kinds: list[str] = field(default_factory=list)


def build_candidates(peptide: str, rng: np.random.Generator,
                     mass_matched: list[str] | None = None,
                     n_near_miss: int = 2) -> CandidateSet:
    """Truth plus decoys. Order of difficulty: mass-matched > near-miss > reversal.

    Mass-matched decoys -- real peptides from the corpus within a few ppm of this
    precursor -- are the hard case and the realistic one: they are exactly the
    candidates a search engine would return, and the ones a reranker must separate.
    Random peptides of the wrong mass are free to reject and teach nothing.
    """
    out = CandidateSet([peptide], [1], ["true"])
    for _ in range(n_near_miss):
        alternative = near_miss(peptide, rng)
        if alternative != peptide and alternative not in out.peptides:
            out.peptides.append(alternative); out.labels.append(0)
            out.kinds.append("near_miss")
    reversed_peptide = pseudo_reverse(peptide)
    if reversed_peptide != peptide and reversed_peptide not in out.peptides:
        out.peptides.append(reversed_peptide); out.labels.append(0)
        out.kinds.append("reverse")
    for other in (mass_matched or []):
        if other not in out.peptides:
            out.peptides.append(other); out.labels.append(0)
            out.kinds.append("mass_matched")
    return out


# ------------------------------------------------------------- classifier

import torch  # noqa: E402
from torch import Tensor, nn  # noqa: E402


class RescoringClassifier(nn.Module):
    """A small MLP over the feature vector. The logit IS the score to rank by.

    Standardisation is inside the module, not a preprocessing step, so it travels with
    the weights. Without it the first epochs are spent undoing scale: `precursor_mz`
    runs to ~2000 while `frag_coverage_b` lives in [0, 1], and weight decay would then
    penalise the small-scale features for being small rather than for being useless.
    """

    def __init__(self, n_features: int, hidden_size: int = 128, layers: int = 2,
                 dropout: float = 0.1):
        super().__init__()
        self.register_buffer("center", torch.zeros(n_features))
        self.register_buffer("scale", torch.ones(n_features))
        body, width = [], n_features
        for _ in range(layers):
            body += [nn.Linear(width, hidden_size), nn.GELU(), nn.Dropout(dropout)]
            width = hidden_size
        self.body = nn.Sequential(*body)
        self.head = nn.Linear(width, 1)

    def fit_standardizer(self, features: Tensor) -> None:
        self.center.copy_(features.mean(0))
        self.scale.copy_(features.std(0).clamp_min(1e-6))

    def forward(self, features: Tensor) -> Tensor:
        return self.head(self.body((features - self.center) / self.scale)).squeeze(-1)


def train_rescorer(features: np.ndarray, labels: np.ndarray, groups: np.ndarray,
                   drop_embedding: bool = False, epochs: int = 40,
                   hidden_size: int = 128, lr: float = 1e-3, seed: int = 0,
                   validation_fraction: float = 0.2,
                   split_keys: np.ndarray | None = None) -> dict:
    """Train on (spectrum, candidate) pairs and report ranking quality.

    Splits by SPECTRUM, not by pair. Decoys are generated from the true peptide, so a
    pair-level split would put a spectrum's truth in train and its own decoy in
    validation, and the classifier would be scored on candidates it had already seen.
    """
    torch.manual_seed(seed)
    columns = [i for i, name in enumerate(FEATURE_NAMES)
               if not (drop_embedding and name == "embedding_cosine")]
    x = torch.tensor(features[:, columns], dtype=torch.float32)
    y = torch.tensor(labels, dtype=torch.float32)

    # Hold out by PEPTIDE when split_keys (the true peptide of each row's spectrum) is
    # given. A spectrum-level split put 100% of test spectra's peptides in training (audit
    # 8859654: each peptide has ~13 replicate spectra), and the peptide-level features
    # (length, charge, GRAVY, modifications, ...) let the classifier memorise them.
    keys = split_keys if split_keys is not None else groups
    unique = np.unique(keys)
    rng = np.random.default_rng(seed)
    held = set(rng.choice(unique, size=max(1, int(len(unique) * validation_fraction)),
                          replace=False).tolist())
    is_validation = np.array([k in held for k in keys])
    train_idx = torch.tensor(np.flatnonzero(~is_validation))
    valid_idx = torch.tensor(np.flatnonzero(is_validation))

    model = RescoringClassifier(len(columns), hidden_size=hidden_size)
    model.fit_standardizer(x[train_idx])
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    # Positives are outnumbered roughly 1:4 by decoys; without the weight the model can
    # score 80% by calling everything wrong.
    positive_weight = ((y[train_idx] == 0).sum() / (y[train_idx] == 1).sum().clamp_min(1))
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=positive_weight)

    model.train()
    for _ in range(epochs):
        order = torch.randperm(len(train_idx))
        for start in range(0, len(order), 256):
            batch = train_idx[order[start : start + 256]]
            optimizer.zero_grad()
            loss_fn(model(x[batch]), y[batch]).backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        scores = model(x[valid_idx]).numpy()
    truth = labels[valid_idx.numpy()]
    groups_v = groups[valid_idx.numpy()]

    from sklearn.metrics import roc_auc_score
    hits = ranked = 0
    for group in np.unique(groups_v):
        rows = groups_v == group
        if truth[rows].sum() != 1:
            continue
        ranked += 1
        hits += int(scores[rows].argmax() == truth[rows].argmax())
    return {"auroc": float(roc_auc_score(truth, scores)),
            "hit@1": hits / max(ranked, 1), "spectra": ranked,
            "pairs": int(len(truth)), "features": len(columns), "model": model}
