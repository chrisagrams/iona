"""The handoff CLI (msdelta.psm_rerank): per-run and global rescoring, model save/load."""

import sys
from pathlib import Path

import numpy as np
import pytest

import msdelta.rerank_psm_fdr  # noqa: F401  -- module-level msdelta import, see FT33

sys.path.insert(0, str(Path(__file__).parent))
from test_rerank_r4 import _synthetic_run, _write  # noqa: E402

from msdelta.psm_rerank import (GlobalModel, load_table, main, score_perrun,  # noqa: E402
                                train_global, engine_score, summarize)
from msdelta.rerank_psm_fdr import WS_FEATURES  # noqa: E402


@pytest.fixture()
def runs(tmp_path):
    rng = np.random.default_rng(0)
    for run, ds in (("runA", "HEK293"), ("runB", "HEK293"), ("runC", "HCT116")):
        rows, lab, *_ = _synthetic_run(run, ds, 300, rng)
        _write(rows, tmp_path / "rows" / f"{run}.parquet")
        _write(lab, tmp_path / "features" / ds / f"{run}.parquet")
    return tmp_path


def test_perrun_beats_engine_on_informative_lab_feature(runs):
    df, spec, lab = load_table(str(runs / "rows"), str(runs / "features"))
    assert "cos_delta" in df.columns and set(lab) == {"feat__signal", "feat__noise"}
    s = score_perrun(df, spec, lab + WS_FEATURES, iters=3)
    _, base = summarize(df, spec, engine_score(df), "engine")
    _, ours = summarize(df, spec, s, "perrun")
    assert ours["psms_1pct"] > base["psms_1pct"]


def test_global_model_roundtrip_and_missing_columns(runs, tmp_path):
    df, spec, lab = load_table(str(runs / "rows"), str(runs / "features"))
    m = train_global(df, spec, lab + WS_FEATURES, epochs=2)
    m.save(tmp_path / "model")
    m2 = GlobalModel.load(str(tmp_path / "model"))
    assert np.allclose(m.score(df), m2.score(df), atol=1e-6)
    # a new run whose lab table lacks a trained column still scores (filled, not crashed)
    df2, _, _ = load_table(str(runs / "rows"), str(runs / "features"),
                           lab_columns=[c for c in m2.cols if c.startswith("feat__")] + ["feat__absent"])
    assert "feat__absent" in df2.columns


def test_cli_perrun_and_global(runs, tmp_path):
    out = tmp_path / "psms.parquet"
    assert main(["score", "--rows", str(runs / "rows"), "--labfeat", str(runs / "features"),
                 "--out", str(out)]) == 0
    import pandas as pd
    t = pd.read_parquet(out)
    assert len(t) == 900 and {"q_value", "is_decoy", "score"} <= set(t.columns)
    assert main(["train", "--rows", str(runs / "rows"), "--labfeat", str(runs / "features"),
                 "--out", str(tmp_path / "m")]) == 0
    assert main(["score", "--mode", "global", "--model", str(tmp_path / "m"), "--rows",
                 str(runs / "rows"), "--labfeat", str(runs / "features"),
                 "--out", str(tmp_path / "g.parquet")]) == 0
