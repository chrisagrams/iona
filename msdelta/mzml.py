"""Stream spectra out of an mzML file and write them back with filtered peak arrays.

Everything outside ``<spectrum>`` elements is copied byte-for-byte, so run,
instrument, chromatogram, and precursor metadata survive unchanged. Only the
binary peak arrays and the peak-derived spectrum attributes are rewritten.
Indexed inputs are written as ordinary mzML files without their index wrapper.
"""

from __future__ import annotations

import base64
import re
import zlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import IO, Iterator

import numpy as np
from lxml import etree

MZ_ARRAY = "MS:1000514"
INTENSITY_ARRAY = "MS:1000515"
_PRECISION = {
    "MS:1000523": np.dtype("<f8"),  # 64-bit float
    "MS:1000521": np.dtype("<f4"),  # 32-bit float
    "MS:1000522": np.dtype("<i8"),  # 64-bit integer
    "MS:1000519": np.dtype("<i4"),  # 32-bit integer
}
_ZLIB = "MS:1000574"
_NO_COMPRESSION = "MS:1000576"
_PEAK_STATISTICS = {
    "MS:1000504": "base_peak_mz",
    "MS:1000505": "base_peak_intensity",
    "MS:1000285": "total_ion_current",
    "MS:1000528": "lowest_observed_mz",
    "MS:1000527": "highest_observed_mz",
}

_SPECTRUM_START = re.compile(rb"<spectrum[\s>]")
_SPECTRUM_END = b"</spectrum>"
_SPECTRUM_LIST_COUNT = re.compile(rb"<spectrumList\b[^>]*\bcount\s*=\s*['\"](\d+)['\"]")
_INDEXED_MZML_START = re.compile(rb"<indexedmzML(?:\s[^>]*)?>")
_INDEX_LIST_START = re.compile(rb"<indexList[\s>]")
_READ_SIZE = 4 << 20


def _local_name(element) -> str:
    return etree.QName(element).localname


def _children(element, name: str) -> Iterator[etree._Element]:
    """Yield the direct child elements of ``element`` whose local name is ``name``."""
    for child in element:
        if isinstance(child.tag, str) and _local_name(child) == name:
            yield child


def _cv_params(element) -> dict[str, etree._Element]:
    """Map cvParam accessions to their elements among ``element``'s direct children."""
    return {
        child.get("accession"): child
        for child in _children(element, "cvParam")
        if child.get("accession")
    }


def _find_child(element, name: str):
    return next(_children(element, name), None)


def decode_binary_array(array_element) -> tuple[np.ndarray, np.dtype, bool]:
    """Decode one ``<binaryDataArray>`` into values, storage dtype, and zlib flag."""
    params = _cv_params(array_element)
    dtypes = [_PRECISION[accession] for accession in params if accession in _PRECISION]
    if len(dtypes) != 1:
        raise ValueError("binaryDataArray must declare exactly one supported precision")
    if _ZLIB in params:
        compressed = True
    elif _NO_COMPRESSION in params:
        compressed = False
    else:
        raise ValueError("binaryDataArray uses an unsupported compression scheme")
    binary = _find_child(array_element, "binary")
    if binary is None:
        raise ValueError("binaryDataArray is missing its <binary> element")
    raw = base64.b64decode(binary.text or b"")
    if compressed and raw:
        raw = zlib.decompress(raw)
    return np.frombuffer(raw, dtype=dtypes[0]), dtypes[0], compressed


def encode_binary_array(values: np.ndarray, dtype: np.dtype, compressed: bool) -> bytes:
    """Encode ``values`` the way ``decode_binary_array`` found them stored."""
    raw = np.ascontiguousarray(values, dtype=dtype).tobytes()
    if compressed and raw:
        raw = zlib.compress(raw)
    return base64.b64encode(raw)


@dataclass
class Spectrum:
    """One ``<spectrum>`` element with its decoded peak arrays."""

    index: int
    id: str
    ms_level: int | None
    mz: np.ndarray
    intensity: np.ndarray
    element: etree._Element = field(repr=False)
    block: bytes = field(default=b"", repr=False)
    prefix: bytes = field(default=b"", repr=False)

    @property
    def peak_count(self) -> int:
        return int(self.mz.size)


def _parse_spectrum(index: int, block: bytes, prefix: bytes) -> Spectrum:
    element = etree.fromstring(block)
    params = _cv_params(element)
    level = params.get("MS:1000511")
    ms_level = int(level.get("value")) if level is not None else None
    mz = np.empty(0, dtype=np.float64)
    intensity = np.empty(0, dtype=np.float64)
    array_list = _find_child(element, "binaryDataArrayList")
    for array in _children(array_list, "binaryDataArray") if array_list is not None else ():
        accessions = _cv_params(array)
        if MZ_ARRAY in accessions:
            mz = decode_binary_array(array)[0]
        elif INTENSITY_ARRAY in accessions:
            intensity = decode_binary_array(array)[0]
    if mz.shape != intensity.shape:
        raise ValueError(f"spectrum {index} has mismatched m/z and intensity arrays")
    return Spectrum(
        index=index,
        id=element.get("id", ""),
        ms_level=ms_level,
        mz=mz,
        intensity=intensity,
        element=element,
        block=block,
        prefix=prefix,
    )


def filter_spectrum_peaks(spectrum: Spectrum, keep: np.ndarray) -> bytes:
    """Serialize ``spectrum`` with every peak array reduced to ``keep``."""
    keep = np.asarray(keep, dtype=bool)
    if keep.shape != spectrum.mz.shape:
        raise ValueError("keep mask must have one entry per peak")
    if keep.all():
        return spectrum.block
    element = spectrum.element

    n_kept = int(keep.sum())
    array_list = _find_child(element, "binaryDataArrayList")
    for array in _children(array_list, "binaryDataArray") if array_list is not None else ():
        values, dtype, compressed = decode_binary_array(array)
        if values.size != keep.size:
            # Arrays with a different length (e.g. per-scan metadata) are left alone.
            continue
        encoded = encode_binary_array(values[keep], dtype, compressed)
        array.set("encodedLength", str(len(encoded)))
        if array.get("arrayLength") is not None:
            array.set("arrayLength", str(n_kept))
        _find_child(array, "binary").text = encoded.decode("ascii")
    element.set("defaultArrayLength", str(n_kept))

    mz = spectrum.mz[keep]
    intensity = spectrum.intensity[keep]
    if n_kept:
        base = int(np.argmax(intensity))
        statistics = {
            "base_peak_mz": mz[base],
            "base_peak_intensity": intensity[base],
            "total_ion_current": intensity.sum(),
            "lowest_observed_mz": mz.min(),
            "highest_observed_mz": mz.max(),
        }
    else:
        statistics = dict.fromkeys(_PEAK_STATISTICS.values(), 0.0)
    for accession, param in _cv_params(element).items():
        name = _PEAK_STATISTICS.get(accession)
        if name is not None:
            param.set("value", str(float(statistics[name])))
    return etree.tostring(element)


class MzMLRewriter:
    """Copy an mzML file while replacing the peak arrays of its spectra.

    Iterate :meth:`spectra` to read spectra in file order, and call :meth:`write`
    for every spectrum in that same order, passing a boolean mask of the peaks to
    keep. Spectra may be buffered between reading and writing, but every spectrum
    must be written before the rewriter is closed. Pass ``destination=None`` on
    distributed workers that only need to read and validate the stream.
    """

    def __init__(self, source: str | Path, destination: str | Path | None):
        self.source = Path(source)
        self.destination = Path(destination) if destination is not None else None
        self._input: IO[bytes] | None = None
        self._output: IO[bytes] | None = None
        self.spectrum_count: int | None = None
        self._read_count = 0
        self._written_count = 0
        self._trailer: bytes | None = None

    def __enter__(self) -> MzMLRewriter:
        self._input = self.source.open("rb")
        if self.destination is not None:
            self._output = self.destination.open("wb")
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        try:
            if exc_type is None:
                self.close()
        finally:
            if self._input is not None:
                self._input.close()
            if self._output is not None:
                self._output.close()

    def _read_more(self, buffer: bytes) -> tuple[bytes, bool]:
        chunk = self._input.read(_READ_SIZE)
        return buffer + chunk, not chunk

    def spectra(self) -> Iterator[Spectrum]:
        """Yield each spectrum with the raw bytes that precede it in the file."""
        buffer = b""
        eof = False
        prefix = bytearray()
        while True:
            match = _SPECTRUM_START.search(buffer)
            if match is None:
                if eof:
                    break
                # Keep a partial "<spectrum" that might straddle the read boundary.
                safe = max(len(buffer) - 16, 0)
                prefix += buffer[:safe]
                buffer, eof = self._read_more(buffer[safe:])
                continue
            end = buffer.find(_SPECTRUM_END, match.start())
            if end < 0:
                if eof:
                    raise ValueError(f"{self.source}: unterminated <spectrum> element")
                buffer, eof = self._read_more(buffer)
                continue
            end += len(_SPECTRUM_END)
            prefix += buffer[: match.start()]
            if self._read_count == 0:
                count = _SPECTRUM_LIST_COUNT.search(prefix)
                if count is not None:
                    self.spectrum_count = int(count.group(1))
                prefix = bytearray(_INDEXED_MZML_START.sub(b"", prefix, count=1))
            spectrum = _parse_spectrum(self._read_count, buffer[match.start() : end], bytes(prefix))
            prefix = bytearray()
            buffer = buffer[end:]
            self._read_count += 1
            yield spectrum
        self._trailer = bytes(prefix + buffer)

    def write(self, spectrum: Spectrum, keep: np.ndarray | None = None) -> None:
        """Write ``spectrum`` keeping only the peaks flagged in ``keep``."""
        if spectrum.index != self._written_count:
            raise ValueError("spectra must be written in the order they were read")
        if self._output is not None:
            self._output.write(spectrum.prefix)
            if keep is None:
                keep = np.ones(spectrum.peak_count, dtype=bool)
            self._output.write(filter_spectrum_peaks(spectrum, keep))
        self._written_count += 1

    def close(self) -> None:
        """Copy the remainder of the file, omitting any indexed mzML wrapper."""
        if self._trailer is None:
            raise ValueError("all spectra must be read before closing the rewriter")
        if self._written_count != self._read_count:
            raise ValueError("every spectrum read must be written before closing")
        trailer = self._trailer
        if self._read_count == 0:
            trailer = _INDEXED_MZML_START.sub(b"", trailer, count=1)
        index_start = _INDEX_LIST_START.search(trailer)
        if index_start is not None:
            trailer = trailer[: index_start.start()]
        if self._output is not None:
            self._output.write(trailer)
        self._trailer = None
