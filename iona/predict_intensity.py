"""Predict fragment-ion intensities for peptides with a trained ``IonaForIntensityPrediction``.

Takes Prosit's input CSV, one row per precursor:

    modified_sequence,collision_energy,precursor_charge
    PEPTIDEK,25,2
    M(ox)CPEPTIDER,30,3

Cysteine is always carbamidomethylated and ``M(ox)`` is oxidized methionine, as in
Prosit. MassIVE-KB (``M[+15.995]``), ProForma (``M[UNIMOD:35]``) and Casanovo
(``M+15.995``) spellings of the same two modifications are accepted too. Collision
energy is normalized collision energy (NCE) in percent, divided by 100 as Prosit does.

Writes a spectral library twice, in Prosit's two formats: ``PREFIX.msp`` and a
Spectronaut-style ``PREFIX.csv`` with one row per fragment ion. Intensities are
relative to the most intense ion of each spectrum. iRT is not predicted.

Example:
    iona-intensity peptides.csv --checkpoint runs/intensity-200m/final --output library
"""

from __future__ import annotations

import argparse
import logging
import re
import sys
from itertools import batched
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from accelerate import PartialState
from tqdm.auto import tqdm

from iona.chemistry import PROSIT_RESIDUE_MASSES, PROTON_MASS, WATER_MASS
from iona.data import normalize_annotation
from iona.intensity import IonaIntensityProcessor, prosit_fragment_ladder
from iona.modeling_iona import IonaForIntensityPrediction
from iona.predict_denovo import resolve_device
from iona.tokenization_iona import TOKEN_PATTERN

logger = logging.getLogger(__name__)

INPUT_COLUMNS = ("modified_sequence", "collision_energy", "precursor_charge")
CSV_COLUMNS = [
    "RelativeIntensity",
    "FragmentMz",
    "ModifiedPeptide",
    "LabeledPeptide",
    "StrippedPeptide",
    "PrecursorCharge",
    "PrecursorMz",
    "CollisionEnergy",
    "FragmentNumber",
    "FragmentType",
    "FragmentCharge",
    "FragmentLossType",
]

# Prosit's integer alphabet; 0 is padding and C is always carbamidomethylated.
PROSIT_ALPHABET = {residue: index for index, residue in enumerate("ACDEFGHIKLMNPQRSTVWY", 1)}
PROSIT_ALPHABET.update({"C[+57.021]": PROSIT_ALPHABET["C"], "M[+15.995]": 21})
PROSIT_TOKENS = {index: residue for residue, index in PROSIT_ALPHABET.items() if len(residue) == 1}
PROSIT_TOKENS[21] = "M(ox)"
SPECTRONAUT_TOKENS = {"C": "C[Carbamidomethyl (C)]", "M(ox)": "M[Oxidation (M)]"}
ION_TYPES = "yb"


def encode_peptide(peptide: str, max_length: int) -> list[int]:
    """Prosit integer encoding of one peptide; raises ValueError if the model cannot take it."""
    try:
        normalized = normalize_annotation(peptide.replace("M(ox)", "M[+15.995]"))
    except ValueError as error:
        raise ValueError(f"peptide {peptide!r}: {error}") from None
    tokens = re.findall(TOKEN_PATTERN, normalized)
    unsupported = sorted({token for token in tokens if token not in PROSIT_ALPHABET})
    if unsupported:
        raise ValueError(
            f"peptide {peptide!r}: {', '.join(unsupported)} not supported; "
            "only carbamidomethyl C and oxidized M are, as in Prosit"
        )
    if not 2 <= len(tokens) <= max_length:
        raise ValueError(f"peptide {peptide!r}: length {len(tokens)} is outside 2-{max_length}")
    return [PROSIT_ALPHABET[token] for token in tokens]


def read_peptides(path: Path, max_length: int, max_charge: int) -> pd.DataFrame:
    """Read and validate Prosit's input CSV."""
    table = pd.read_csv(path)
    missing = [column for column in INPUT_COLUMNS if column not in table.columns]
    if missing:
        raise ValueError(f"{path} is missing column(s) {', '.join(missing)}")
    errors = []
    encoded = []
    for line, row in enumerate(table.itertuples(index=False), start=2):
        try:
            encoded.append(encode_peptide(str(row.modified_sequence).strip(), max_length))
            charge, energy = int(row.precursor_charge), float(row.collision_energy)
            if not 1 <= charge <= max_charge:
                raise ValueError(f"precursor charge {charge} is outside 1-{max_charge}")
            if not 0 < energy <= 100:
                raise ValueError(f"collision energy {energy} is not an NCE in (0, 100]")
        except ValueError as error:
            errors.append(f"{path}:{line}: {error}")
    if errors:
        raise ValueError(
            "\n".join(errors[:20] + ([f"... {len(errors) - 20} more"] if len(errors) > 20 else []))
        )
    sequence_integer = np.zeros((len(table), max_length), dtype=np.int64)
    for row, tokens in enumerate(encoded):
        sequence_integer[row, : len(tokens)] = tokens
    return pd.DataFrame(
        {
            "sequence_integer": list(sequence_integer),
            "precursor_charge": table["precursor_charge"].astype(int),
            "collision_energy": table["collision_energy"].astype(float),
        }
    )


class IntensityPredictor:
    """Predict Prosit-layout intensities and fragment m/z for batches of peptides."""

    def __init__(
        self,
        model: IonaForIntensityPrediction,
        processor: IonaIntensityProcessor,
        *,
        batch_size: int = 1024,
    ):
        self.model = model.eval()
        self.device = next(model.parameters()).device
        self.processor = processor
        self.batch_size = batch_size

    @torch.inference_mode()
    def predict(self, peptides: pd.DataFrame, show_progress: bool = False):
        """Return ``(intensities, fragment_mz)``, each (peptides, 174); -1 marks impossible ions."""
        n_slots = self.model.config.num_ion_slots
        intensities = np.full((len(peptides), n_slots), -1.0, dtype=np.float32)
        fragment_mz = np.zeros((len(peptides), n_slots), dtype=np.float32)
        # Batch peptides of similar length together to limit padding.
        lengths = [int((sequence > 0).sum()) for sequence in peptides["sequence_integer"]]
        order = sorted(range(len(peptides)), key=lengths.__getitem__)
        with tqdm(total=len(order), unit="peptides", disable=not show_progress) as bar:
            for rows in batched(order, self.batch_size):
                rows = list(rows)
                features = [
                    {
                        "sequence_integer": peptides["sequence_integer"].iat[row],
                        "precursor_charge": peptides["precursor_charge"].iat[row],
                        # The model was trained on Prosit's collision_energy_aligned_normed.
                        "collision_energy": peptides["collision_energy"].iat[row] / 100.0,
                    }
                    for row in rows
                ]
                batch = self.processor(features)
                batch = {key: value.to(self.device) for key, value in batch.items()}
                output = self.model(**batch)
                intensities[rows] = output.intensities.float().cpu().numpy()
                mz, _, _ = prosit_fragment_ladder(
                    torch.stack([torch.as_tensor(f["sequence_integer"]) for f in features]),
                    torch.as_tensor([int(f["precursor_charge"]) for f in features]),
                    self.model.config.max_fragment_charge,
                )
                fragment_mz[rows] = mz.numpy()
                bar.update(len(rows))
        return intensities, fragment_mz


def fragment_annotations(n_slots: int, max_fragment_charge: int):
    """(type, number, charge) of each slot in Prosit's flat layout."""
    slots = np.arange(n_slots)
    per_ion = len(ION_TYPES) * max_fragment_charge
    number = slots // per_ion + 1
    ion_type = np.array(list(ION_TYPES))[(slots % per_ion) // max_fragment_charge]
    charge = slots % max_fragment_charge + 1
    return ion_type, number, charge


def precursor_mz(sequence_integer: np.ndarray, charge: int) -> float:
    neutral = sum(PROSIT_RESIDUE_MASSES[int(token)] for token in sequence_integer) + WATER_MASS
    return (neutral + charge * PROTON_MASS) / charge


def msp_mods(sequence: list[str]) -> tuple[str, str]:
    """Prosit's ``Mods`` and ``ModString`` fields."""
    mods = [
        (position, residue[0], "Carbamidomethyl" if residue == "C" else "Oxidation")
        for position, residue in enumerate(sequence)
        if residue in ("C", "M(ox)")
    ]
    if not mods:
        return "0", ""
    count = str(len(mods)) + "".join(f"/{p},{r},{name}" for p, r, name in mods)
    return count, "; ".join(f"{name}@{r}{p + 1}" for p, r, name in mods)


def precursor_table(peptides: pd.DataFrame) -> pd.DataFrame:
    """One row per peptide: its spellings, precursor m/z, and Prosit's MSP mod fields."""
    records = []
    for sequence_integer, charge, energy in zip(
        peptides["sequence_integer"], peptides["precursor_charge"], peptides["collision_energy"]
    ):
        residues = [PROSIT_TOKENS[int(token)] for token in sequence_integer if token]
        modified = "".join(residues)
        mods, mod_string = msp_mods(residues)
        records.append(
            {
                "ModifiedPeptide": "_"
                + "".join(SPECTRONAUT_TOKENS.get(r, r) for r in residues)
                + "_",
                "LabeledPeptide": modified,
                "StrippedPeptide": modified.replace("M(ox)", "M"),
                "PrecursorCharge": int(charge),
                "PrecursorMz": precursor_mz(sequence_integer, int(charge)),
                "CollisionEnergy": float(energy),
                "Mods": mods,
                "ModString": mod_string,
            }
        )
    return pd.DataFrame(records)


def fragment_table(
    precursors: pd.DataFrame,
    intensities: np.ndarray,
    fragment_mz: np.ndarray,
    max_fragment_charge: int,
    min_intensity: float,
) -> pd.DataFrame:
    """One row per kept fragment ion, grouped by spectrum and sorted by m/z within it."""
    spectrum, slot = np.nonzero(intensities > min_intensity)
    order = np.lexsort((fragment_mz[spectrum, slot], spectrum))
    spectrum, slot = spectrum[order], slot[order]
    ion_type, number, charge = fragment_annotations(intensities.shape[1], max_fragment_charge)
    fragments = precursors.iloc[spectrum].reset_index(drop=True)
    fragments["spectrum"] = spectrum
    fragments["RelativeIntensity"] = intensities[spectrum, slot]
    fragments["FragmentMz"] = fragment_mz[spectrum, slot].astype(np.float64).round(5)
    fragments["FragmentNumber"] = number[slot]
    fragments["FragmentType"] = ion_type[slot]
    fragments["FragmentCharge"] = charge[slot]
    fragments["FragmentLossType"] = "noloss"
    return fragments


def write_msp(path: Path | str, precursors: pd.DataFrame, fragments: pd.DataFrame) -> None:
    """Write the library in the layout of Prosit's MSP converter."""
    charge = fragments["FragmentCharge"].astype(str)
    peaks = (
        (
            fragments["FragmentMz"].map("{:.5f}".format)
            + "\t"
            + fragments["RelativeIntensity"].map("{:.6g}".format)
            + '\t"'
            + fragments["FragmentType"]
            + fragments["FragmentNumber"].astype(str)
            + ("^" + charge).where(fragments["FragmentCharge"] > 1, "")
            + '/0.0ppm"'
        ).tolist()
        if len(fragments)
        else []
    )
    ends = np.cumsum(np.bincount(fragments["spectrum"], minlength=len(precursors)))
    start = 0
    with open(path, "w", encoding="utf-8") as msp:
        for row, (precursor, end) in enumerate(zip(precursors.itertuples(index=False), ends)):
            name, z = precursor.StrippedPeptide, precursor.PrecursorCharge
            if row:
                msp.write("\n")
            msp.write(
                f"Name: {name}/{z}\nMW: {precursor.PrecursorMz}\n"
                f"Comment: Parent={precursor.PrecursorMz} "
                f"Collision_energy={precursor.CollisionEnergy:g} Mods={precursor.Mods} "
                f"ModString={name}//{precursor.ModString}/{z}\n"
                f"Num peaks: {end - start}\n"
            )
            msp.writelines(f"{peak}\n" for peak in peaks[start:end])
            start = end


def write_library(
    prefix: Path,
    peptides: pd.DataFrame,
    intensities: np.ndarray,
    fragment_mz: np.ndarray,
    max_fragment_charge: int,
    min_intensity: float,
) -> int:
    """Write ``prefix.msp`` and ``prefix.csv``; returns the number of fragment ions written."""
    precursors = precursor_table(peptides)
    fragments = fragment_table(
        precursors, intensities, fragment_mz, max_fragment_charge, min_intensity
    )
    write_msp(f"{prefix}.msp", precursors, fragments)
    table = fragments[CSV_COLUMNS].assign(PrecursorMz=fragments["PrecursorMz"].round(5))
    table.to_csv(f"{prefix}.csv", index=False)
    return len(fragments)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("input", type=Path, help="Prosit-style CSV of peptides")
    parser.add_argument(
        "--checkpoint", required=True, help="IonaForIntensityPrediction checkpoint path or Hub ID"
    )
    parser.add_argument("--processor", help="processor path; defaults to the checkpoint")
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="output prefix; writes PREFIX.msp and PREFIX.csv",
    )
    parser.add_argument(
        "--min-intensity",
        type=float,
        default=0.0,
        help="drop fragment ions at or below this relative intensity (default: 0)",
    )
    parser.add_argument("--batch-size", type=int, default=1024, help="peptides per model batch")
    parser.add_argument("--device", help="torch device (default: cuda, then xpu, then cpu)")
    parser.add_argument("--no-progress", action="store_true", help="disable the progress bar")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    device = resolve_device(args.device, PartialState())
    model = IonaForIntensityPrediction.from_pretrained(args.checkpoint, dtype="auto").to(device)
    processor = IonaIntensityProcessor.from_pretrained(args.processor or args.checkpoint)
    config = model.config
    try:
        peptides = read_peptides(
            args.input, config.max_sequence_length, config.max_precursor_charge
        )
    except ValueError as error:
        logger.error("%s", error)
        return 1
    intensities, fragment_mz = IntensityPredictor(
        model, processor, batch_size=args.batch_size
    ).predict(peptides, show_progress=not args.no_progress)
    ions = write_library(
        args.output,
        peptides,
        intensities,
        fragment_mz,
        config.max_fragment_charge,
        args.min_intensity,
    )
    logger.info(
        "wrote %d spectra, %d fragment ions to %s.{msp,csv}", len(peptides), ions, args.output
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
