"""Round-trip and scoring tests for mzML denoising."""

from __future__ import annotations

import base64
import hashlib
import re
import zlib

import numpy as np
import pytest
import torch
from lxml import etree

from msdelta.configuration_msdelta import MSDeltaConfig, MSDeltaDenoisingConfig
from msdelta.denoise_mzml import (
    SpectrumDenoiser,
    denoise_mzml,
    peak_budget_batches,
    select_model_peaks,
)
from msdelta.modeling_msdelta import MSDeltaForDenoising
from msdelta.mzml import MzMLRewriter
from msdelta.processing_msdelta import MSDeltaProcessor

_HEADER = """<?xml version="1.0" encoding="utf-8"?>
<indexedmzML xmlns="http://psi.hupo.org/ms/mzml">
  <mzML xmlns="http://psi.hupo.org/ms/mzml" id="test" version="1.1.0">
    <cvList count="1">
      <cv id="MS" fullName="PSI-MS" version="4.1.0" URI="https://example.org/psi-ms.obo"/>
    </cvList>
    <run id="test" defaultInstrumentConfigurationRef="IC1">
      <spectrumList count="{count}" defaultDataProcessingRef="dp">
"""

_SPECTRUM = """        <spectrum index="{index}" id="scan={scan}" defaultArrayLength="{n}">
          <cvParam cvRef="MS" accession="MS:1000511" name="ms level" value="{level}"/>
          <cvParam cvRef="MS" accession="MS:1000504" name="base peak m/z" value="{bpmz}"/>
          <cvParam cvRef="MS" accession="MS:1000505" name="base peak intensity" value="{bpi}"/>
          <cvParam cvRef="MS" accession="MS:1000285" name="total ion current" value="{tic}"/>
          <cvParam cvRef="MS" accession="MS:1001581" name="FAIMS compensation voltage" value="-45.0"/>
          <binaryDataArrayList count="2">
            <binaryDataArray encodedLength="{mz_len}">
              <cvParam cvRef="MS" accession="MS:1000523" name="64-bit float" value=""/>
              <cvParam cvRef="MS" accession="MS:1000574" name="zlib compression" value=""/>
              <cvParam cvRef="MS" accession="MS:1000514" name="m/z array" value=""/>
              <binary>{mz}</binary>
            </binaryDataArray>
            <binaryDataArray encodedLength="{int_len}">
              <cvParam cvRef="MS" accession="MS:1000521" name="32-bit float" value=""/>
              <cvParam cvRef="MS" accession="MS:1000576" name="no compression" value=""/>
              <cvParam cvRef="MS" accession="MS:1000515" name="intensity array" value=""/>
              <binary>{intensity}</binary>
            </binaryDataArray>
          </binaryDataArrayList>
        </spectrum>
"""

_TRAILER = """      </spectrumList>
      <chromatogramList count="1" defaultDataProcessingRef="dp">
        <chromatogram index="0" id="TIC" defaultArrayLength="0">
          <binaryDataArrayList count="0"/>
        </chromatogram>
      </chromatogramList>
    </run>
  </mzML>
"""


def _encode(values, dtype, compress):
    raw = np.asarray(values, dtype=dtype).tobytes()
    if compress:
        raw = zlib.compress(raw)
    return base64.b64encode(raw).decode()


def _write_indexed_mzml(path, spectra):
    """Write a minimal but valid indexedmzML file for ``(level, mz, intensity)`` triples."""
    body = _HEADER.format(count=len(spectra)).encode()
    offsets = []
    for index, (level, mz, intensity) in enumerate(spectra):
        mz_text = _encode(mz, "<f8", True)
        int_text = _encode(intensity, "<f4", False)
        base = int(np.argmax(intensity)) if len(intensity) else None
        block = _SPECTRUM.format(
            index=index,
            scan=index + 1,
            n=len(mz),
            level=level,
            bpmz=mz[base] if base is not None else 0.0,
            bpi=intensity[base] if base is not None else 0.0,
            tic=float(np.sum(intensity)),
            mz_len=len(mz_text),
            int_len=len(int_text),
            mz=mz_text,
            intensity=int_text,
        ).encode()
        offsets.append((f"scan={index + 1}", len(body) + block.index(b"<spectrum")))
        body += block
    trailer = _TRAILER.encode()
    chromatogram_offset = len(body) + trailer.index(b"<chromatogram ")
    body += trailer
    index_offset = len(body)
    index = '<indexList count="2">\n  <index name="spectrum">\n'
    index += "".join(f'    <offset idRef="{i}">{o}</offset>\n' for i, o in offsets)
    index += '  </index>\n  <index name="chromatogram">\n'
    index += f'    <offset idRef="TIC">{chromatogram_offset}</offset>\n  </index>\n</indexList>\n'
    body += index.encode()
    body += f"<indexListOffset>{index_offset}</indexListOffset>\n<fileChecksum>".encode()
    body += hashlib.sha1(body).hexdigest().encode() + b"</fileChecksum>\n</indexedmzML>\n"
    path.write_bytes(body)


def _example_spectra(rng, count=6, max_peaks=40):
    spectra = []
    for index in range(count):
        n = int(rng.integers(1, max_peaks + 1))
        mz = np.sort(rng.uniform(100, 1500, n))
        intensity = rng.uniform(1, 1e5, n).astype(np.float32)
        spectra.append((1 if index % 3 == 0 else 2, mz, intensity))
    spectra.append((2, np.empty(0), np.empty(0, dtype=np.float32)))
    return spectra


def _check_index_and_checksum(path):
    data = path.read_bytes()
    checksum_at = data.index(b"<fileChecksum>") + len(b"<fileChecksum>")
    expected = hashlib.sha1(data[:checksum_at]).hexdigest().encode()
    assert data[checksum_at : checksum_at + 40] == expected
    index_offset = int(re.search(rb"<indexListOffset>(\d+)</indexListOffset>", data).group(1))
    assert data[index_offset:].startswith(b"<indexList")
    for identifier, offset in re.findall(rb'<offset idRef="([^"]*)">(\d+)</offset>', data):
        tag = b"<chromatogram " if identifier == b"TIC" else b"<spectrum "
        assert data[int(offset) :].startswith(tag)
        assert b'id="' + identifier + b'"' in data[int(offset) : int(offset) + 200]


def _read_spectra(path):
    """Decode every spectrum of an mzML file independently of ``msdelta.mzml``."""
    ns = {"mz": "http://psi.hupo.org/ms/mzml"}
    tree = etree.parse(str(path))
    spectra = {}
    for element in tree.iterfind(".//mz:spectrum", ns):
        params = {
            param.get("name"): param.get("value") for param in element.iterfind("mz:cvParam", ns)
        }
        arrays = {}
        for array in element.iterfind("mz:binaryDataArrayList/mz:binaryDataArray", ns):
            names = {param.get("name") for param in array.iterfind("mz:cvParam", ns)}
            raw = base64.b64decode(array.findtext("mz:binary", "", ns))
            if "zlib compression" in names and raw:
                raw = zlib.decompress(raw)
            dtype = "<f8" if "64-bit float" in names else "<f4"
            values = np.frombuffer(raw, dtype=dtype)
            assert int(array.get("encodedLength")) == len(array.findtext("mz:binary", "", ns))
            assert values.size == int(element.get("defaultArrayLength"))
            arrays["m/z array" if "m/z array" in names else "intensity array"] = values
        spectra[element.get("id")] = {**params, **arrays}
    return spectra


def test_rewriter_round_trip_preserves_unfiltered_file(tmp_path):
    rng = np.random.default_rng(0)
    spectra = _example_spectra(rng)
    source = tmp_path / "source.mzML"
    destination = tmp_path / "copy.mzML"
    _write_indexed_mzml(source, spectra)

    with MzMLRewriter(source, destination) as rewriter:
        for spectrum in rewriter.spectra():
            assert spectrum.ms_level == spectra[spectrum.index][0]
            np.testing.assert_array_equal(spectrum.mz, spectra[spectrum.index][1])
            np.testing.assert_array_equal(spectrum.intensity, spectra[spectrum.index][2])
            rewriter.write(spectrum)
    assert destination.read_bytes() == source.read_bytes()


def test_rewriter_filters_arrays_and_rebuilds_index(tmp_path):
    rng = np.random.default_rng(1)
    spectra = _example_spectra(rng)
    source = tmp_path / "source.mzML"
    destination = tmp_path / "filtered.mzML"
    _write_indexed_mzml(source, spectra)

    masks = {}
    with MzMLRewriter(source, destination) as rewriter:
        buffered = list(rewriter.spectra())
        for spectrum in buffered:
            keep = rng.random(spectrum.peak_count) < 0.5
            if spectrum.index == 1:
                keep[:] = False
            masks[spectrum.id] = keep
            rewriter.write(spectrum, keep)

    _check_index_and_checksum(destination)
    written = _read_spectra(destination)
    for index, (level, mz, intensity) in enumerate(spectra):
        spectrum = written[f"scan={index + 1}"]
        keep = masks[f"scan={index + 1}"]
        np.testing.assert_array_equal(spectrum["m/z array"], mz[keep])
        np.testing.assert_allclose(spectrum["intensity array"], intensity[keep])
        assert int(spectrum["ms level"]) == level
        assert spectrum["FAIMS compensation voltage"] == "-45.0"
        if keep.any():
            base = int(np.argmax(intensity[keep]))
            assert float(spectrum["base peak m/z"]) == pytest.approx(mz[keep][base])
            assert float(spectrum["total ion current"]) == pytest.approx(intensity[keep].sum())
        else:
            assert float(spectrum["total ion current"]) == 0.0
    assert b"</indexedmzML>" in destination.read_bytes()[-20:]


def test_peak_budget_batches_respect_budget():
    lengths = [10, 3, 7, 10, 1, 5]
    batches = peak_budget_batches(lengths, peak_pair_budget=100)
    assert sorted(i for batch in batches for i in batch) == list(range(len(lengths)))
    for batch in batches:
        longest = max(lengths[i] for i in batch)
        assert len(batch) * longest**2 <= 100


def test_select_model_peaks_keeps_most_intense_in_order():
    intensity = np.array([5.0, 1.0, 9.0, 3.0, 7.0])
    np.testing.assert_array_equal(select_model_peaks(intensity, 3), [0, 2, 4])
    np.testing.assert_array_equal(select_model_peaks(intensity, 10), np.arange(5))


@pytest.fixture(scope="module")
def tiny_denoiser():
    torch.manual_seed(0)
    config = MSDeltaDenoisingConfig(
        encoder=MSDeltaConfig(
            hidden_size=16,
            num_attention_heads=2,
            num_hidden_layers=1,
            intermediate_size=32,
            delta_bias_n_freqs=4,
        ),
        head_hidden_size=8,
    )
    model = MSDeltaForDenoising(config)
    processor = MSDeltaProcessor(max_peaks=8)
    return model, processor


def test_denoise_mzml_end_to_end(tmp_path, tiny_denoiser):
    model, processor = tiny_denoiser
    rng = np.random.default_rng(3)
    spectra = _example_spectra(rng, count=5, max_peaks=12)
    source = tmp_path / "source.mzML"
    _write_indexed_mzml(source, spectra)

    denoiser = SpectrumDenoiser(model, processor, noise_threshold=0.5, peak_pair_budget=64 * 3)
    destination = tmp_path / "denoised.mzML"
    summary = denoise_mzml(source, destination, denoiser, ms_levels={2}, chunk_size=2)
    _check_index_and_checksum(destination)

    assert summary.spectra == len(spectra)
    assert summary.denoised_spectra == sum(level == 2 for level, _, _ in spectra)
    assert summary.peaks_in - summary.peaks_out == summary.noise_peaks + summary.unscored_peaks
    written = _read_spectra(destination)
    for index, (level, mz, intensity) in enumerate(spectra):
        spectrum = written[f"scan={index + 1}"]
        if level != 2 or not len(mz):
            np.testing.assert_array_equal(spectrum["m/z array"], mz)
            continue
        assert len(spectrum["m/z array"]) <= processor.max_peaks
        selected = select_model_peaks(intensity, processor.max_peaks)
        noise = denoiser.noise_probabilities([(mz[selected], intensity[selected])])[0]
        np.testing.assert_array_equal(spectrum["m/z array"], mz[selected][noise < 0.5])

    kept = SpectrumDenoiser(model, processor, keep_unscored=True, peak_pair_budget=64 * 3)
    keep_destination = tmp_path / "kept.mzML"
    keep_summary = denoise_mzml(source, keep_destination, kept, ms_levels={2})
    assert keep_summary.unscored_peaks == 0
    assert keep_summary.peaks_out >= summary.peaks_out
