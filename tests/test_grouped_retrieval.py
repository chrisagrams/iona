"""ms-contrastive-100k support: the chunked top-k metric and the analyte flattening."""

import numpy as np
import pytest
import torch

from msdelta.contrastive import retrieval_metrics_exact, retrieval_metrics_topk
from msdelta.grouped_retrieval import flatten_analyte

KEYS = ("Hit@1", "Precision@1", "R-Precision", "MAP@R", "R@5", "MAP@100")


def _random_problem(seed, n_groups, sizes=(1, 2, 3, 4, 5), dim=16, noise=1.0):
    rng = np.random.default_rng(seed)
    groups = np.concatenate([[g] * rng.choice(sizes) for g in range(n_groups)])
    centers = rng.normal(size=(n_groups, dim))
    x = centers[groups] + noise * rng.normal(size=(len(groups), dim))
    return torch.tensor(x, dtype=torch.float32), groups


@pytest.mark.parametrize("seed", range(5))
@pytest.mark.parametrize("chunk", [1, 7, 64, 10_000])
def test_topk_matches_exact(seed, chunk):
    emb, groups = _random_problem(seed, n_groups=60)
    exact = retrieval_metrics_exact(emb, groups)
    topk = retrieval_metrics_topk(emb, groups, k=100, chunk=chunk)
    for key in KEYS:
        assert topk[key] == pytest.approx(exact[key], abs=1e-6), key


def test_topk_matches_exact_when_k_exceeds_n():
    emb, groups = _random_problem(1, n_groups=8)          # n < 100
    exact = retrieval_metrics_exact(emb, groups)
    topk = retrieval_metrics_topk(emb, groups, k=100, chunk=3)
    for key in KEYS:
        assert topk[key] == pytest.approx(exact[key], abs=1e-6), key


def test_topk_counts_only_scorable_queries():
    emb, groups = _random_problem(2, n_groups=40, sizes=(1, 3))
    counts = np.bincount(groups)
    out = retrieval_metrics_topk(emb, groups)
    assert out["queries"] == float((counts[groups] > 1).sum())


def test_topk_refuses_r_above_k():
    emb, groups = _random_problem(3, n_groups=3, sizes=(6,))
    with pytest.raises(ValueError):
        retrieval_metrics_topk(emb, groups, k=4)


def test_topk_perfect_and_separated():
    groups = np.repeat(np.arange(20), 4)
    emb = torch.eye(20)[groups] + 1e-3 * torch.randn(80, 20)
    out = retrieval_metrics_topk(emb, groups, chunk=9)
    assert out["MAP@R"] == pytest.approx(1.0)
    assert out["Hit@1"] == pytest.approx(1.0)


def _analyte():
    spec = lambda v: {"mz": [100.0 + v, 200.0 + v], "intensity": [1.0, 2.0]}
    return {"analyte_id": "a1", "peptide": "PEPTIDEK", "charge": 2, "precursor": 450.2,
            "consensus": spec(0.5),
            "experimental": [dict(spec(i), spectrum_id=f"s{i}") for i in range(3)]}


def test_flatten_experimental_only():
    out = flatten_analyte(_analyte(), include_consensus=False)
    assert out["source"] == ["experimental"] * 3
    assert out["mz"] == [[100.0 + i, 200.0 + i] for i in range(3)]
    assert out["peptide"] == ["PEPTIDEK"] * 3 and out["charge"] == [2] * 3
    assert out["precursor"] == [450.2] * 3


def test_flatten_with_consensus_puts_it_first():
    out = flatten_analyte(_analyte(), include_consensus=True)
    assert out["source"] == ["consensus"] + ["experimental"] * 3
    assert out["mz"][0] == [100.5, 200.5]
    assert len({len(v) for v in out.values()}) == 1   # parallel lists


def test_group_batch_sampler_members_unchanged_by_argsort():
    """The argsort construction must reproduce the old flatnonzero-per-group members
    exactly, or every existing seed would draw different batches."""
    from msdelta.contrastive import GroupBatchSampler

    rng = np.random.default_rng(0)
    groups = rng.integers(0, 50, size=700)
    sampler = GroupBatchSampler(groups, groups_per_batch=4, replicates=3, seed=1)
    old = {int(g): np.flatnonzero(groups == g) for g in np.unique(groups)}
    assert list(sampler.members) == list(old)
    for g in old:
        np.testing.assert_array_equal(sampler.members[g], old[g])


@pytest.mark.parametrize("consensus,k,ok", [(False, 3, True), (False, 4, False),
                                            (True, 4, True), (True, 5, False)])
def test_grouped_refuses_k_above_group_size(consensus, k, ok, monkeypatch):
    """K > spectra per analyte would pair a spectrum with itself as a positive."""
    import msdelta.grouped_retrieval as gr
    from msdelta.finetune_contrastive import (ContrastiveDataArguments,
                                              load_contrastive_datasets)

    class Loaded(Exception):
        pass

    def fake_load(*a, **k):
        raise Loaded
    monkeypatch.setattr(gr, "load_spectrum_datasets", fake_load)
    args = ContrastiveDataArguments(dataset_format="grouped", include_consensus=consensus,
                                    replicates=k)
    with pytest.raises(Loaded if ok else ValueError):
        load_contrastive_datasets(args, processor=None)


def test_binned_embeddings_bins_log_intensity():
    from msdelta.eval_grouped_retrieval import binned_embeddings

    rows = {"mz": [[100.2, 100.7, 250.0], [1999.9, 2500.0]],
            "log_intensity": [[0.25, 0.5, 1.0], [1.0, 1.0]]}
    out = binned_embeddings(rows, width=1.0)
    assert out.shape == (2, 2000)
    assert out[0, 100].item() == pytest.approx(0.75)   # both peaks land in bin 100
    assert out[0, 250].item() == pytest.approx(1.0)
    assert out[1, 1999].item() == pytest.approx(1.0)   # 2500 is out of range
    assert out[1].sum().item() == pytest.approx(1.0)
