"""data/prepare_massive_kb.py on tiny synthetic shards (CPU, seconds)."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch

REPO = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location("prepare_massive_kb",
                                               REPO / "data/prepare_massive_kb.py")
pk = importlib.util.module_from_spec(_spec)
sys.modules["prepare_massive_kb"] = pk          # so worker processes can pickle its functions
_spec.loader.exec_module(pk)


# ------------------------------------------------------------------ parsing

def test_split_peptide_charge():
    assert pk.split_peptide_charge("DVNAAIATIKTK_2") == ("DVNAAIATIKTK", 2)
    assert pk.split_peptide_charge("[-17.027]QSPLGFEVHDEITVTR_3") == \
        ("[-17.027]QSPLGFEVHDEITVTR", 3)
    for bad, reason in [("PEPTIDE", "bad_format"), ("PEPTIDE_x", "bad_charge"),
                        ("PEPTIDE_0", "bad_charge"), ("", "bad_format")]:
        with pytest.raises(pk.Drop) as err:
            pk.split_peptide_charge(bad)
        assert err.value.reason == reason


@pytest.mark.parametrize("source, ours", [
    ("YHQIGSGKC[+57.021]EIK", "YHQIGSGKC[57.0215]EIK"),
    ("M[+15.995]PEPTIDEK", "M[15.9949]PEPTIDEK"),
    ("PEN[+0.984]GQ[+0.984]K", "PEN[0.9840]GQ[0.9840]K"),
    ("[+42.011]AAAK", "[42.0106]AAAK"),
    ("[+43.006]AAAK", "[43.0058]AAAK"),
    ("[-17.027]QSPLGR", "[-17.0265]QSPLGR"),
    ("[+42.011]M[+15.995]C[+57.021]K", "[42.0106]M[15.9949]C[57.0215]K"),
    ("PEPTIDE", "PEPTIDE"),
])
def test_modification_mapping(source, ours):
    mapped, tokens = pk.map_modifications(source)
    assert mapped == ours
    assert len(tokens) == source.count("[")


@pytest.mark.parametrize("source, reason, detail", [
    ("S[+57.021]PEPK", "unknown_modification", "S[+57.021]"),      # right token, wrong residue
    ("PES[+79.966]K", "unknown_modification", "S[+79.966]"),       # phospho: not in the table
    ("AAC[+57.02]K", "unknown_modification", "C[+57.02]"),         # different precision
    ("[+43.006-17.027]QK", "unknown_modification", "^[+43.006-17.027]"),
    ("[+43.006][-17.027]QK", "unknown_modification", "^[43.0058][-17.027]"),
    ("M[+15.995][+0.984]K", "unknown_modification", "M[+0.984]"),
    ("PEPUK", "unknown_residue", "U"),
    ("pepK", "unknown_residue", "p"),
])
def test_unknown_modifications_are_dropped(source, reason, detail):
    with pytest.raises(pk.Drop) as err:
        pk.map_modifications(source)
    assert (err.value.reason, err.value.detail) == (reason, detail)


def test_masses_match_ours():
    from msdelta.data.chemistry import PROTON_MASS, RESIDUE_MASSES
    from msdelta.rescoring.reranking import peptide_neutral_mass
    assert abs(pk.PROTON - PROTON_MASS) < 1e-9
    for peptide in ["PEPTIDEK", "YHQIGSGKC[57.0215]EIK", "[-17.0265]QSPLGR",
                    "[42.0106]M[15.9949]C[57.0215]K", "PEN[0.9840]GQ[0.9840]K"]:
        assert pk.neutral_mass(peptide) == pytest.approx(peptide_neutral_mass(peptide), abs=1e-9)
    pk.neutral_mass("A")
    assert pk._RESIDUE_MASSES == RESIDUE_MASSES
    # theoretical m/z: (M + z * proton) / z
    assert pk.precursor_mz("PEPTIDEK", 2) == pytest.approx(
        (peptide_neutral_mass("PEPTIDEK") + 2 * PROTON_MASS) / 2)


def test_sequence_key():
    assert pk.sequence_key("[42.0106]M[15.9949]LIC[57.0215]K", False) == "MLICK"
    assert pk.sequence_key("[42.0106]M[15.9949]LIC[57.0215]K", True) == "MLLCK"
    assert pk.sequence_key("n[42.0106]AAIK", True) == "AALK"        # replicate-corpus N-term
    assert pk.sequence_key("[+42.011]AAIK", False) == "AAIK"        # source notation


def test_peptide_key_matches_ours():
    from msdelta.models.peptide_encoder import peptide_key
    assert pk.peptide_key("C[57.0215]AK", 3) == peptide_key("C[57.0215]AK", 3)


def test_assign_split_is_stable_and_proportional():
    keys = [f"PEPTIDE{i}K" for i in range(20000)]
    first = [pk.assign_split(k, 0.1, 0.05) for k in keys]
    assert first == [pk.assign_split(k, 0.1, 0.05) for k in keys]
    counts = {s: first.count(s) / len(keys) for s in pk.SPLITS}
    assert counts["validation"] == pytest.approx(0.1, abs=0.01)
    assert counts["test"] == pytest.approx(0.05, abs=0.01)


# ------------------------------------------------------------------ spectrum processing

def test_process_spectra_matches_processor():
    from msdelta.models.processing_msdelta import MSDeltaProcessor
    processor = MSDeltaProcessor(max_peaks=8)
    rng = np.random.default_rng(0)
    spectra = [(np.sort(rng.uniform(100, 1000, n)).astype(np.float32),
                rng.uniform(0, 1e5, n).astype(np.float32)) for n in (5, 8, 3, 1)]
    spectra.append((np.arange(9, dtype=np.float32) + 100, np.ones(9, np.float32)))  # oversize
    spectra.append((np.array([100.0, 200.0], np.float32), np.zeros(2, np.float32)))  # all-zero
    spectra.append((np.array([100.0], np.float32), np.array([-1.0], np.float32)))    # negative
    spectra.append((np.array([np.nan], np.float32), np.array([1.0], np.float32)))    # nan
    spectra.append((np.zeros(0, np.float32), np.zeros(0, np.float32)))              # empty
    offsets = np.concatenate([[0], np.cumsum([len(m) for m, _ in spectra])])
    mz = np.concatenate([m for m, _ in spectra])
    it = np.concatenate([i for _, i in spectra])
    status, off, out_mz, out_li = pk.process_spectra(offsets, mz, it, 8, "drop")
    assert list(status) == ["", "", "", "", "oversize"] + ["invalid_spectrum"] * 4
    for k in range(4):
        ref = processor(torch.as_tensor(spectra[k][0]), torch.as_tensor(spectra[k][1]),
                        padding=False)
        np.testing.assert_allclose(out_mz[off[k]:off[k + 1]], ref["mz"], rtol=0)
        np.testing.assert_allclose(out_li[off[k]:off[k + 1]], ref["log_intensity"],
                                   rtol=1e-6, atol=1e-7)
    with pytest.raises(ValueError):          # the processor rejects oversize too
        processor(torch.as_tensor(spectra[4][0]), torch.as_tensor(spectra[4][1]))
    # "top" keeps the most intense max_peaks, in m/z order
    m = np.arange(10, dtype=np.float32) + 100
    i = np.array([5, 1, 9, 2, 8, 3, 7, 4, 6, 0.5], np.float32)
    status, off, out_mz, _ = pk.process_spectra(np.array([0, 10]), m, i, 4, "top")
    assert list(status) == [""]
    assert out_mz.tolist() == [102.0, 104.0, 106.0, 108.0]      # intensities 9, 8, 7, 6


# ------------------------------------------------------------------ end to end

GOOD = "ACDEFGHK"


def _spectrum(n, seed):
    rng = np.random.default_rng(seed)
    return (np.sort(rng.uniform(100, 1500, n)).astype(np.float32).tolist(),
            rng.uniform(1, 1e4, n).astype(np.float32).tolist())


def _write_shard(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    mzs, ints, names = [], [], []
    for name, n, seed in rows:
        m, i = _spectrum(n, seed)
        mzs.append(m)
        ints.append(i)
        names.append(name)
    pq.write_table(pa.table({"m/z": pa.array(mzs, pa.list_(pa.float32())),
                             "int": pa.array(ints, pa.list_(pa.float32())),
                             "peptide_charge": names}), path, row_group_size=3)


@pytest.fixture
def world(tmp_path):
    """A 3-split source with one row per drop reason, plus a one-peptide eval set."""
    src = tmp_path / "src"
    _write_shard(src / "train/train_000.parquet", [
        (f"{GOOD}_2", 10, 0), (f"{GOOD}_2", 12, 1), (f"{GOOD}_3", 9, 2),
        ("C[+57.021]PEPK_2", 7, 3), ("C[+57.021]PEPK_2", 7, 4),
        ("S[+79.966]AAK_2", 5, 5),                   # unknown modification
        ("EVALPEPTLDEK_2", 6, 6),                    # eval peptide as I/L variant
        ("[+42.011]EVALPEPTIDEK_3", 6, 7),           # eval peptide, other mod form
        ("LARGEPEAK_2", 20, 8),                      # oversize (max_peaks 16)
        ("NOCHARGE", 5, 9),                          # bad format
    ])
    _write_shard(src / "train/train_001.parquet", [
        (f"{GOOD}_2", 10, 10), ("M[+15.995]NNK_2", 8, 11), ("M[+15.995]NNK_2", 8, 12),
    ])
    _write_shard(src / "val/val_000.parquet", [("VVVVK_2", 5, 13), ("VVVVK_2", 6, 14)])
    _write_shard(src / "test/test_000.parquet", [("TTTTK_2", 5, 15), ("PEPUK_2", 5, 16)])
    from datasets import Dataset
    Dataset.from_dict({"peptide": ["EVALPEPTIDEK", "OTHERK"], "charge": [2, 2]}) \
        .save_to_disk(str(tmp_path / "eval_a"))
    Dataset.from_dict({"peptide": ["[42.0106]EVALPEPTIDEK"], "charge": [2]}) \
        .save_to_disk(str(tmp_path / "eval_b"))
    assert pk.main(["exclusion", "--out", str(tmp_path / "excl"), "--no-defaults",
                    "--source", f"a={tmp_path / 'eval_a'}",
                    "--source", f"b={tmp_path / 'eval_b'}"]) == 0
    return tmp_path


def _prepare(world, out="out", *extra):
    return pk.main(["prepare", "--out", str(world / out), "--exclusion", str(world / "excl"),
                    "--source-dir", str(world / "src"), "--max-peaks", "16",
                    "--workers", "1", *extra])


def test_exclusion_counts(world):
    data = json.loads((world / "excl/exclusion.json").read_text())
    assert data["sources"]["a"]["peptides"] == 2
    assert data["sources"]["b"]["sequences"] == 1
    assert data["union"]["sequences_il"] == 2
    # refuse to overwrite the exclusion set too
    assert pk.main(["exclusion", "--out", str(world / "excl"), "--no-defaults",
                    "--source", f"a={world / 'eval_a'}"]) == 2


@pytest.mark.parametrize("workers", ["1", "2"])
def test_prepare_end_to_end(world, workers):
    assert pk.main(["prepare", "--out", str(world / "out"), "--exclusion", str(world / "excl"),
                    "--source-dir", str(world / "src"), "--max-peaks", "16",
                    "--workers", workers, "--split-policy", "source"]) == 0
    out = world / "out"
    m = json.loads((out / "manifest.json").read_text())
    assert m["source"]["revision"] == pk.REVISION
    assert m["rows_in"] == 17
    assert m["dropped_by_reason"] == {
        "bad_format": 1, "bad_charge": 0, "excluded_eval": 2, "unknown_residue": 1,
        "unknown_modification": 1, "oversize": 1, "invalid_spectrum": 0}
    assert m["unknown_tokens"] == {"S[+79.966]": 1, "residue:U": 1}
    assert m["kept_by_split"] == {"train": 8, "validation": 2, "test": 1}
    assert m["unaccounted_rows"] == 0
    assert m["modification_tokens_kept"] == {"C[+57.021]": 2, "M[+15.995]": 2}
    # groups by (peptide, charge)
    g = pq.read_table(out / "group_sizes.parquet").to_pylist()
    sizes = {(r["split"], r["analyte_id"]): r["count"] for r in g}
    assert sizes == {("train", f"{GOOD}_2"): 3, ("train", f"{GOOD}_3"): 1,
                     ("train", "C[57.0215]PEPK_2"): 2, ("train", "M[15.9949]NNK_2"): 2,
                     ("validation", "VVVVK_2"): 2, ("test", "TTTTK_2"): 1}
    assert m["groups"]["train"]["singleton_groups"] == 1
    # the overlap report: both eval sources, per source
    rep = json.loads((out / "overlap_report.json").read_text())
    assert rep["total_spectra_removed"] == 2
    assert rep["sources"]["a"]["massive_kb_spectra_removed"] == 2
    assert rep["sources"]["a"]["massive_kb_spectra_matching_without_il_collapse"] == 1
    assert rep["sources"]["b"]["eval_sequences_in_massive_kb"] == 1
    # rows carry our notation, processed values and the theoretical precursor
    rows = pq.read_table(out / "train").to_pylist()
    assert {r["peptide"] for r in rows} == {GOOD, "C[57.0215]PEPK", "M[15.9949]NNK"}
    for r in rows:
        assert r["analyte_id"] == f"{r['peptide']}_{r['charge']}"
        assert max(r["log_intensity"]) == pytest.approx(1.0)
        assert r["precursor"] == pytest.approx(pk.precursor_mz(r["peptide"], r["charge"]),
                                               abs=1e-3)
        assert r["source"] == "experimental"
    assert pk.main(["check", str(out)]) == 0


def test_il_flag_off_keeps_il_variant(world):
    assert _prepare(world, "out", "--no-il-collapse") == 0
    m = json.loads((world / "out/manifest.json").read_text())
    assert m["dropped_by_reason"]["excluded_eval"] == 1
    peptides = set()
    for split in pk.SPLITS:
        if (world / "out" / split).exists():
            peptides |= set(pq.read_table(world / "out" / split).column("peptide").to_pylist())
    assert "EVALPEPTLDEK" in peptides


def test_peptide_split_policy_is_disjoint(world):
    assert _prepare(world, "out", "--validation-fraction", "0.4",
                    "--test-fraction", "0.3") == 0
    seen = {}
    for split in pk.SPLITS:
        if not (world / "out" / split).exists():
            continue
        for p in pq.read_table(world / "out" / split).column("peptide").to_pylist():
            assert seen.setdefault(pk.sequence_key(p, True), split) == split
    assert pk.main(["check", str(world / "out")]) == 0


def test_refuses_to_overwrite_and_resumes(world):
    assert _prepare(world) == 0
    before = (world / "out/manifest.json").read_text()
    assert _prepare(world) == 2                                  # existing output
    assert (world / "out/manifest.json").read_text() == before
    # an interrupted run: a .partial with one shard done
    partial = world / "out2.partial"
    assert _prepare(world, "out2", "--shards", "1", "--splits", "train") == 0
    (world / "out2").rename(partial)
    (partial / "manifest.json").unlink()
    assert _prepare(world, "out2", "--shards", "1", "--splits", "train") == 2   # no --resume
    assert _prepare(world, "out2", "--shards", "1", "--splits", "train", "--resume") == 0
    assert _prepare(world, "out3", "--limit", "2") == 0
    m = json.loads((world / "out3/manifest.json").read_text())
    assert m["rows_in"] == 2 + 2 + 2 + 2


def test_resume_refuses_changed_settings(world):
    partial = world / "out.partial"
    (partial / "_shards").mkdir(parents=True)
    (partial / "_settings.json").write_text(json.dumps({"max_peaks": 1}))
    assert _prepare(world, "out", "--resume") == 2


def test_check_catches_tampering(world):
    assert _prepare(world) == 0
    out = world / "out"
    m = json.loads((out / "manifest.json").read_text())
    m["kept_by_split"]["train"] += 1
    (out / "manifest.json").write_text(json.dumps(m))
    assert pk.main(["check", str(out)]) == 1
