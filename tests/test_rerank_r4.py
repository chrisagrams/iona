"""R4/R5 rescorer pieces: lab features, per-run Percolator regime, product-PCA vectors."""

import json

import numpy as np
import pytest

import msdelta.rerank_psm_fdr  # noqa: F401  -- module-level msdelta import, see FT33
from msdelta.rerank_psm_fdr import (ProductPCA, calibrate, fit_linear, load_lab_features,
                                    load_vectors, main, qvalues, top_per_spectrum)


def _write(table: dict, path):
    import pyarrow as pa
    import pyarrow.parquet as pq
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.table(table), path)


def test_lab_features_drop_empty_and_constant_fill_mod_counts(tmp_path):
    _write({"candidate_id": ["r1:1:1", "r1:1:2"], "feat__good": [0.1, 0.9],
            "feat__empty": [float("nan")] * 2, "feat__const": [3.0, 3.0],
            "feat__mod_count_ox": [1.0, 2.0], "label": [1, 0]},
           tmp_path / "features" / "HEK293" / "r1.parquet")
    _write({"candidate_id": ["r2:1:1", "r2:1:2"], "feat__good": [0.3, 0.2],
            "feat__empty": [float("nan")] * 2, "feat__const": [3.0, 3.0], "label": [0, 1]},
           tmp_path / "features" / "HCT116" / "r2.parquet")
    lab, cols = load_lab_features(str(tmp_path / "features"), ["r1", "r2"])
    assert set(cols) == {"feat__good", "feat__mod_count_ox"}       # label never a feature
    assert lab.set_index("candidate").loc["r2:1:1", "feat__mod_count_ox"] == 0.0
    with pytest.raises(SystemExit):
        load_lab_features(str(tmp_path / "features"), ["r3"])


def test_fit_linear_separates_and_warm_starts():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(400, 3)); y = (X[:, 0] + 0.1 * rng.normal(size=400) > 0).astype(float)
    w = fit_linear(X, y)
    assert w[0] > abs(w[1]) and w[0] > abs(w[2])
    w2 = fit_linear(X, y, w0=w, steps=5)
    assert np.allclose(w, w2, atol=0.5)


def test_calibrate_puts_the_one_percent_threshold_at_zero():
    rng = np.random.default_rng(0)
    n = 2000
    decoy = rng.random(n) < 0.3
    s = np.where(decoy, rng.normal(0, 1, n), rng.normal(4, 1, n))
    spec = np.arange(n)
    out = calibrate(s, decoy, spec, s)
    q = qvalues(s, decoy)
    thr = s[(q <= 0.01) & ~decoy].min()
    assert out[s == thr][0] == pytest.approx(0.0)
    assert np.median(out[decoy]) == pytest.approx(-1.0)


def test_product_pca_is_fitted_on_given_rows_only():
    rng = np.random.default_rng(0)
    s = rng.normal(size=(20, 16)).astype(np.float16)
    p = rng.normal(size=(60, 16)).astype(np.float16)
    owner = np.repeat(np.arange(20), 3)
    pca = ProductPCA(s, p, owner, k=4).fit(np.arange(30))
    z = pca.transform(np.arange(60))
    assert z.shape == (60, 4)
    assert np.allclose(z[:30].mean(0), 0, atol=1e-4)        # centred on the fit rows
    null = pca.transform(np.arange(60), owner=np.roll(owner, 3))
    assert not np.allclose(null, z)


def test_load_vectors_aligns_by_candidate(tmp_path):
    for run, n_s in (("a", 2), ("b", 3)):
        d = tmp_path / run; d.mkdir()
        np.save(d / "spectrum.npy", np.full((n_s, 2), ord(run), np.float16))
        cands = [f"{run}:{i}" for i in range(n_s)]
        np.save(d / "peptide.npy", np.arange(n_s * 2, dtype=np.float16).reshape(n_s, 2))
        _write({"candidate": cands, "owner": np.arange(n_s), "null_owner":
                np.roll(np.arange(n_s), 1)}, d / "index.parquet")
    s, p, o, no = load_vectors(str(tmp_path), ["b:2", "a:0", "b:0"])
    assert s.shape == (5, 2)
    assert s[o].tolist() == [[98, 98], [97, 97], [98, 98]]      # owners map into run b/a
    assert p.tolist() == [[4, 5], [0, 1], [0, 1]]
    assert o.tolist() == [4, 0, 2]
    with pytest.raises(SystemExit):
        load_vectors(str(tmp_path), ["c:0"])


def _synthetic_run(run, dataset, n_spec, rng, k=4, dim=8):
    """Rows + lab features + vectors for one run. The true target (rank 1 for most
    spectra) has a higher e-value, a higher lab feature and a vector aligned with its
    spectrum; decoys are random."""
    rows = {c: [] for c in ("spectrum_id", "run_id", "dataset", "charge", "n_peaks",
                            "candidate", "peptide", "sequence", "is_decoy", "label",
                            "length", "cosine", "cosine_null", "msfragger_hyperscore",
                            "search_rank", "search_delta_score", "search_neglog10_evalue",
                            "num_matched_ions", "tot_num_ions", "massdiff", "num_tol_term",
                            "num_missed_cleavages")}
    lab = {"candidate_id": [], "feat__signal": [], "feat__noise": []}
    spec = rng.normal(size=(n_spec, dim)); spec /= np.linalg.norm(spec, axis=1, keepdims=True)
    peps, owners = [], []
    for i in range(n_spec):
        true = rng.random() < 0.6
        for r in range(k):
            is_t = true and r == 0
            decoy = (not is_t) and rng.random() < 0.5
            v = spec[i] + (0.2 if is_t else 2.0) * rng.normal(size=dim)
            v /= np.linalg.norm(v)
            cand = f"{run}:{i}:{r + 1}"
            vals = dict(spectrum_id=f"{run}:scan={i}", run_id=run, dataset=dataset,
                        charge=2, n_peaks=50, candidate=cand, peptide=f"P{i}_{r}",
                        sequence=f"P{i}R{r}", is_decoy=decoy, label=int(is_t), length=9,
                        cosine=float(v @ spec[i]), cosine_null=float(rng.normal() * 0.1),
                        msfragger_hyperscore=float(rng.normal() + 2 * is_t),
                        search_rank=r + 1, search_delta_score=0.1,
                        search_neglog10_evalue=float(rng.normal() + (3 if is_t else 0)),
                        num_matched_ions=5, tot_num_ions=10, massdiff=0.001,
                        num_tol_term=2, num_missed_cleavages=0)
            for c, x in vals.items():
                rows[c].append(x)
            lab["candidate_id"].append(cand)
            lab["feat__signal"].append(float(rng.normal() + 2 * is_t))
            lab["feat__noise"].append(float(rng.normal()))
            peps.append(v); owners.append(i)
    return rows, lab, spec, np.array(peps), np.array(owners)


@pytest.mark.parametrize("regime", ["global", "perrun"])
def test_end_to_end_regimes_with_lab_and_vectors(tmp_path, regime):
    rng = np.random.default_rng(0)
    for run, ds in (("runA", "HEK293"), ("runB", "HEK293"), ("runC", "HCT116")):
        rows, lab, spec, peps, owners = _synthetic_run(run, ds, 150, rng)
        _write(rows, tmp_path / "rows" / f"{run}.parquet")
        _write(lab, tmp_path / "features" / ds / f"{run}.parquet")
        d = tmp_path / "vectors" / run; d.mkdir(parents=True)
        np.save(d / "spectrum.npy", spec.astype(np.float16))
        np.save(d / "peptide.npy", peps.astype(np.float16))
        _write({"candidate": rows["candidate"], "owner": owners,
                "null_owner": (owners + 7) % len(spec)}, d / "index.parquet")
    out = tmp_path / "out.json"
    rc = main(["--rows", str(tmp_path / "rows"), "--out", str(out), "--models", "linear",
               "--labfeat", str(tmp_path / "features"), "--vectors", str(tmp_path / "vectors"),
               "--pca-k", "3", "--regime", regime, "--iters", "3", "--sets", "ms,lab",
               "--runs", "runA,runB,runC"])
    assert rc == 0
    rep = json.loads(out.read_text())
    m = rep["methods"]["all"]
    prefix = "perrun/linear" if regime == "perrun" else "linear"
    for s in ("ms", "lab"):
        for v in ("", "+emb", "+embws", "+embvec", "+nullvec"):
            assert f"{prefix}:{s}{v}" in m
    # the informative lab feature helps; the real vectors help; the null ones do not
    assert m[f"{prefix}:lab"]["psms_1pct"] >= m[f"{prefix}:ms"]["psms_1pct"]
    assert m[f"{prefix}:ms+embvec"]["psms_1pct"] > m[f"{prefix}:ms"]["psms_1pct"]
    assert m[f"{prefix}:ms+nullvec"]["psms_1pct"] <= m[f"{prefix}:ms+embvec"]["psms_1pct"]


def test_runs_filter_refuses_missing(tmp_path):
    rng = np.random.default_rng(1)
    rows, *_ = _synthetic_run("runA", "HEK293", 20, rng)
    _write(rows, tmp_path / "rows" / "runA.parquet")
    with pytest.raises(SystemExit):
        main(["--rows", str(tmp_path / "rows"), "--out", str(tmp_path / "o.json"),
              "--runs", "runA,runZ"])
