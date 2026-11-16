"""Read spectra and their precursors from peak files for inference.

Only MGF is supported for now.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np
from pyteomics import mgf


@dataclass
class InputSpectrum:
    """One MS2 spectrum with what de novo sequencing needs to know about its precursor.

    `index` is the zero-based position of the spectrum in its file, as Casanovo reports it.
    """

    index: int
    title: str
    scan: str
    mz: np.ndarray
    intensity: np.ndarray
    precursor_mz: float
    charge: int
    retention_time: float | None
    annotation: str | None


def read_mgf(path: str | Path) -> Iterator[InputSpectrum]:
    """Stream the spectra of an MGF file in file order.

    Raises ValueError for a spectrum without a precursor m/z or without exactly one positive
    charge.
    """
    with mgf.read(
        str(path), use_index=False, convert_arrays=1, read_charges=True, read_ions=False
    ) as reader:
        for index, spectrum in enumerate(reader):
            params = spectrum["params"]
            title = params.get("title", "")
            where = f"{Path(path).name} spectrum {index} ({title!r})"
            if "pepmass" not in params:
                raise ValueError(f"{where} has no PEPMASS")
            charges = params.get("charge")
            if charges is None or len(charges) != 1 or charges[0] <= 0:
                raise ValueError(f"{where} needs exactly one positive CHARGE, got {charges}")
            retention_time = params.get("rtinseconds")
            yield InputSpectrum(
                index=index,
                title=title,
                scan=params.get("scans", ""),
                mz=spectrum["m/z array"],
                intensity=spectrum["intensity array"],
                precursor_mz=float(params["pepmass"][0]),
                charge=int(charges[0]),
                retention_time=float(retention_time) if retention_time is not None else None,
                annotation=params.get("seq"),
            )


def count_spectra(path: str | Path) -> int:
    """Count the spectra in an MGF file without parsing their peaks."""
    count = 0
    with open(path, "rb") as handle:
        for line in handle:
            count += line.startswith(b"BEGIN IONS")
    return count


def read_spectra(path: str | Path) -> Iterator[InputSpectrum]:
    """Stream the spectra of a supported peak file."""
    path = Path(path)
    if path.suffix.lower() != ".mgf":
        raise ValueError(f"{path}: only .mgf files are supported")
    return read_mgf(path)
