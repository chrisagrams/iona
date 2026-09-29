"""C27-C: --consensus_weight in GroupBatchSampler (notes/C27_consensus_weight_card.md).

The K members of a group are drawn WITHOUT replacement, the consensus with weight w and each
experimental spectrum with weight 1; w = inf always draws the consensus. w = 1 must leave
every batch bit-identical to the sampler before C27.
"""

from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np
import pytest

from msdelta.finetuning.contrastive.contrastive import GroupBatchSampler
from msdelta.finetuning.contrastive.finetune_contrastive import (
    ContrastiveDataArguments, check_consensus_weight, consensus_rows)


class _PreC27Sampler(GroupBatchSampler):
    """GroupBatchSampler.__iter__ exactly as it was before C27 (commit e6636ded)."""

    def __iter__(self):
        rng = np.random.default_rng([self.seed, self.epoch])
        if self.group_masses is None:
            order = rng.permutation(list(self.members))
        elif self.random_fraction:
            order = self._mixed_order(rng)
        else:
            keys = np.array(list(self.members))
            mass = np.array([self.group_masses[int(k)] for k in keys], dtype=np.float64)
            by_mass = keys[np.argsort(mass + rng.uniform(-self.mass_jitter, self.mass_jitter,
                                                          len(keys)), kind="stable")]
            n = len(by_mass) // self.groups_per_batch
            blocks = [by_mass[i * self.groups_per_batch:(i + 1) * self.groups_per_batch]
                      for i in range(n)]
            order = (np.concatenate([blocks[j] for j in rng.permutation(n)])
                     if n else by_mass)
        for start in range(0, len(order) - self.groups_per_batch + 1,
                           self.groups_per_batch):
            batch: list[int] = []
            for group in order[start : start + self.groups_per_batch]:
                pool = self.members[int(group)]
                take = rng.choice(pool, size=self.replicates,
                                  replace=len(pool) < self.replicates)
                batch.extend(int(i) for i in take)
            yield batch
        self.epoch += 1


def _corpus(n_groups=400, seed=0, sizes=(4,)):
    """Rows grouped like the flattened corpus: consensus first, then experimental, shuffled."""
    rng = np.random.default_rng(seed)
    groups, mask = [], []
    for g in range(n_groups):
        size = int(rng.choice(sizes))
        groups += [g] * size
        mask += [True] + [False] * (size - 1)
    perm = rng.permutation(len(groups))
    groups, mask = np.array(groups)[perm], np.array(mask)[perm]
    masses = {g: float(rng.uniform(500, 3000)) for g in range(n_groups)}
    return groups, mask, masses


def _epochs(sampler, n=3):
    return [list(sampler) for _ in range(n)]


@pytest.mark.parametrize("mode", [
    {},
    {"mass": True},
    {"mass": True, "random_fraction": 0.25, "random_mix": "within"},
    {"mass": True, "random_fraction": 0.25, "random_mix": "between"},
    {"mass": True, "random_fraction": 0.25, "random_mix": "regions"},
])
@pytest.mark.parametrize("replicates", [2, 3, 4])
@pytest.mark.parametrize("with_mask", [False, True])
def test_default_weight_reproduces_pre_c27_batches(mode, replicates, with_mask):
    # sizes include 1..2 so groups smaller than K (drawn with replacement) are covered too
    groups, mask, masses = _corpus(sizes=(1, 2, 3, 4, 5))
    kw = dict(groups_per_batch=8, replicates=replicates, seed=7,
              group_masses=masses if mode.get("mass") else None,
              random_fraction=mode.get("random_fraction", 0.0),
              random_mix=mode.get("random_mix", "within"))
    old = _PreC27Sampler(groups, **kw)
    new = GroupBatchSampler(groups, consensus_mask=mask if with_mask else None,
                            consensus_weight=1.0, **kw)
    assert _epochs(new) == _epochs(old)


@pytest.mark.parametrize("weight", [0.25, 1.0, 3.0, 10.0, math.inf])
@pytest.mark.parametrize("replicates", [2, 3, 4])
def test_no_member_repeated_within_a_group_draw(weight, replicates):
    groups, mask, masses = _corpus(n_groups=200)
    sampler = GroupBatchSampler(groups, groups_per_batch=10, replicates=replicates, seed=1,
                                group_masses=masses, consensus_mask=mask,
                                consensus_weight=weight)
    for batch in (b for epoch in _epochs(sampler, 5) for b in epoch):
        assert len(batch) == 10 * replicates
        for j in range(0, len(batch), replicates):
            draw = batch[j:j + replicates]
            assert len(set(draw)) == replicates
            assert len({int(groups[i]) for i in draw}) == 1


def _p_pair_has_consensus(w):
    """Exact, K = 2 from {c, e1, e2, e3}, successive weighted sampling without replacement."""
    if math.isinf(w):
        return 1.0
    return w / (w + 3) + 3 / (w + 3) * w / (w + 2)


@pytest.mark.parametrize("weight,expected", [(1.0, 0.5), (3.0, 0.8), (math.inf, 1.0)])
def test_consensus_frequency_matches_formula(weight, expected):
    assert _p_pair_has_consensus(weight) == pytest.approx(expected)
    groups, mask, _ = _corpus(n_groups=2000, seed=3)
    sampler = GroupBatchSampler(groups, groups_per_batch=50, replicates=2, seed=0,
                                consensus_mask=mask, consensus_weight=weight)
    hits = total = 0
    for batch in (b for epoch in _epochs(sampler, 20) for b in epoch):
        for j in range(0, len(batch), 2):
            hits += bool(mask[batch[j]] or mask[batch[j + 1]])
            total += 1
    # 40,000 pairs: binomial sd <= 0.0025, so 0.01 is 4 sd
    assert hits / total == pytest.approx(expected, abs=0.01)


def test_other_weights_match_formula_too():
    groups, mask, _ = _corpus(n_groups=2000, seed=4)
    for w in (0.5, 6.0):
        sampler = GroupBatchSampler(groups, groups_per_batch=50, replicates=2, seed=2,
                                    consensus_mask=mask, consensus_weight=w)
        pairs = [(b[j], b[j + 1]) for e in _epochs(sampler, 20) for b in e
                 for j in range(0, len(b), 2)]
        freq = np.mean([mask[a] or mask[b] for a, b in pairs])
        assert freq == pytest.approx(_p_pair_has_consensus(w), abs=0.01)


def test_always_at_k3_is_consensus_plus_two_distinct_experimental():
    groups, mask, _ = _corpus(n_groups=300, seed=5)
    sampler = GroupBatchSampler(groups, groups_per_batch=10, replicates=3, seed=0,
                                consensus_mask=mask, consensus_weight=math.inf)
    seen = set()
    for batch in (b for e in _epochs(sampler, 3) for b in e):
        for j in range(0, len(batch), 3):
            draw = batch[j:j + 3]
            assert sum(bool(mask[i]) for i in draw) == 1
            seen |= {i for i in draw if not mask[i]}
    assert len(seen) > 0.9 * (~mask).sum()      # experimental rows are all reachable


def test_group_without_consensus_row_is_uniform():
    # a consensus dropped by max_peaks: the group falls back to uniform without replacement
    groups = np.array([0, 0, 0, 0, 1, 1, 1])
    mask = np.array([True, False, False, False, False, False, False])
    sampler = GroupBatchSampler(groups, groups_per_batch=2, replicates=2, seed=0,
                                consensus_mask=mask, consensus_weight=math.inf)
    for batch in (b for e in _epochs(sampler, 50) for b in e):
        for j in range(0, 4, 2):
            assert len(set(batch[j:j + 2])) == 2


def test_weight_without_consensus_is_an_error():
    groups, mask, _ = _corpus(n_groups=20)
    with pytest.raises(ValueError, match="consensus_mask"):
        GroupBatchSampler(groups, replicates=2, consensus_weight=3.0)
    with pytest.raises(ValueError, match="no row is marked"):
        GroupBatchSampler(groups, replicates=2, consensus_mask=np.zeros_like(mask),
                          consensus_weight=3.0)
    for bad in (0.0, -1.0, float("nan")):
        with pytest.raises(ValueError, match="> 0"):
            GroupBatchSampler(groups, replicates=2, consensus_mask=mask, consensus_weight=bad)

    def args(**kw):
        base = dict(consensus_weight=1.0, include_consensus=False, dataset_format="grouped")
        return SimpleNamespace(**(base | kw))

    check_consensus_weight(args())                                   # default: fine
    check_consensus_weight(args(include_consensus=True))             # C20-style: fine
    check_consensus_weight(args(consensus_weight=3.0, include_consensus=True))
    with pytest.raises(ValueError, match="include_consensus"):
        check_consensus_weight(args(consensus_weight=3.0))
    with pytest.raises(ValueError, match="include_consensus"):
        check_consensus_weight(args(consensus_weight=math.inf, include_consensus=True,
                                    dataset_format="replicate"))
    with pytest.raises(ValueError, match="pair"):
        check_consensus_weight(args(consensus_weight=3.0, include_consensus=True),
                               pair_loss=True)
    with pytest.raises(ValueError, match="> 0"):
        check_consensus_weight(args(consensus_weight=0.0, include_consensus=True))


def test_consensus_rows_reads_the_source_column():
    train = {"source": ["consensus", "experimental", "experimental", "consensus"]}
    assert consensus_rows(train, SimpleNamespace(consensus_weight=1.0)) is None
    assert consensus_rows(train, SimpleNamespace(consensus_weight=3.0)).tolist() == [
        True, False, False, True]


def test_parser_accepts_inf_and_defaults_to_one():
    from transformers import HfArgumentParser
    parser = HfArgumentParser(ContrastiveDataArguments)
    (default,) = parser.parse_args_into_dataclasses(args=[])
    assert default.consensus_weight == 1.0
    (inf,) = parser.parse_args_into_dataclasses(args=["--consensus_weight", "inf"])
    assert math.isinf(inf.consensus_weight) and inf.consensus_weight > 0
    (three,) = parser.parse_args_into_dataclasses(args=["--consensus_weight", "3"])
    assert three.consensus_weight == 3.0


def test_c27_arms_parse_and_pass_the_check():
    """Every generated C27 arm parses to the weight its name promises and passes the guard."""
    from transformers import HfArgumentParser
    from tests.conftest import REPO

    expected = {"cons_w1": 1.0, "cons_w3": 3.0, "cons_always": math.inf, "cons_w3_kl0": 3.0}
    arms = sorted((REPO / "configs" / "sweep-c27").glob("*/training.args"))
    assert len(arms) == 12
    listed = (REPO / "sweeps" / "arms" / "c27.txt").read_text().split()
    assert sorted(listed) == sorted(p.parent.name for p in arms)
    parser = HfArgumentParser(ContrastiveDataArguments)
    for path in arms:
        data, _rest = parser.parse_args_into_dataclasses(
            args=path.read_text().split(), return_remaining_strings=True)
        arm = path.parent.name.removeprefix("s050m_ck540k_").rsplit("_seed", 1)[0]
        assert data.include_consensus is True and data.dataset_format == "grouped"
        assert data.consensus_weight == expected[arm]
        assert data.groups_per_batch == 128 and data.replicates == 2
        check_consensus_weight(data)
