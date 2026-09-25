"""rerank_psm_fdr.load_ms2r_features: MS2Rescore all-ranks features joined back to our candidates."""

import pandas as pd
import pytest

from msdelta.rerank_psm_fdr import load_ms2r_features


def _write(root, run, cand, out):
    (root / "in" / run).mkdir(parents=True)
    (root / "fullall").mkdir(exist_ok=True)
    pd.DataFrame(cand).to_csv(root / "in" / run / "candidates.tsv", sep="\t", index=False)
    pd.DataFrame(out).to_csv(root / "fullall" / f"{run}.tsv", sep="\t", index=False)


def test_join_on_spectrum_and_peptidoform(tmp_path):
    cand = {"candidate_id": ["r:1:1", "r:1:2", "r:2:1"], "spectrum_id": ["r:scan=1", "r:scan=1", "r:scan=2"],
            "peptidoform": ["PEPTIDE/2", "C[Carbamidomethyl]AK/2", "PEPTIDE/2"],
            "search_rank": [1, 2, 1], "is_decoy": [False, True, False]}
    # MS2Rescore reorders rows and adds its own score; the join must use (spectrum, peptidoform)
    out = {"peptidoform": ["PEPTIDE/2", "PEPTIDE/2", "C[Carbamidomethyl]AK/2"],
           "spectrum_id": ["r:scan=2", "r:scan=1", "r:scan=1"], "score": [9.0, 8.0, 7.0],
           "rescoring:spec_pearson": [0.3, 0.1, 0.2], "rescoring:rt_diff": [3.0, 1.0, 2.0],
           "rescoring:const": [1.0, 1.0, 1.0]}
    _write(tmp_path, "r", cand, out)
    f, cols = load_ms2r_features(str(tmp_path), ["r"])
    assert cols == ("ms2r_spec_pearson", "ms2r_rt_diff")          # constant column dropped, score not used
    got = f.set_index("candidate")
    assert got.loc["r:1:1", "ms2r_spec_pearson"] == 0.1
    assert got.loc["r:1:2", "ms2r_rt_diff"] == 2.0
    assert got.loc["r:2:1", "ms2r_spec_pearson"] == 0.3


def test_refuses_missing_features(tmp_path):
    cand = {"candidate_id": ["r:1:1", "r:1:2"], "spectrum_id": ["r:scan=1", "r:scan=1"],
            "peptidoform": ["PEPTIDE/2", "AAK/2"], "search_rank": [1, 2], "is_decoy": [False, False]}
    out = {"peptidoform": ["PEPTIDE/2"], "spectrum_id": ["r:scan=1"], "rescoring:x": [0.5]}
    _write(tmp_path, "r", cand, out)
    with pytest.raises(SystemExit):
        load_ms2r_features(str(tmp_path), ["r"])
