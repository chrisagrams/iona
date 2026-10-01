"""K189-P: length-grouped global batches (msdelta.pretraining.length_grouping)."""

from types import SimpleNamespace

import numpy as np
import pytest
from datasets import Dataset

from msdelta.pretraining.length_grouping import GlobalLengthGroupedSampler, spectrum_lengths
from msdelta.pretraining.train import MSDeltaTrainer


def _lengths(n=10_000, seed=0):
    # Roughly the MSConsensus-100M shape at cap 512 (median ~200, long right tail).
    return np.clip(np.random.default_rng(seed).lognormal(5.3, 0.45, n), 1, 512).astype(np.int32)


def test_every_index_once_per_epoch():
    lengths = _lengths()
    order = GlobalLengthGroupedSampler(lengths, global_batch=96, megabatches=5, seed=1).order()
    assert sorted(order.tolist()) == list(range(len(lengths)))


def test_deterministic_per_seed_and_epoch():
    lengths = _lengths()
    a = GlobalLengthGroupedSampler(lengths, 96, 5, seed=1)
    b = GlobalLengthGroupedSampler(lengths, 96, 5, seed=1)
    assert np.array_equal(a.order(), b.order())  # every rank builds the same order
    b.set_epoch(1)
    assert not np.array_equal(a.order(), b.order())
    a.set_epoch(1)
    assert np.array_equal(a.order(), b.order())


def test_global_batches_are_length_homogeneous():
    lengths, gb = _lengths(), 96
    order = GlobalLengthGroupedSampler(lengths, gb, megabatches=20, seed=0).order()
    full = order[: (len(order) // gb) * gb].reshape(-1, gb)
    spread = (lengths[full].max(1) - lengths[full].min(1)).mean()
    rand = np.random.default_rng(0).permutation(len(lengths))[: full.size].reshape(-1, gb)
    rand_spread = (lengths[rand].max(1) - lengths[rand].min(1)).mean()
    assert spread < 0.2 * rand_spread
    # Padding saved: sum over batches of the longest spectrum, grouped vs random.
    assert lengths[full].max(1).sum() < 0.75 * lengths[rand].max(1).sum()


def test_round_robin_ranks_share_a_global_batch():
    """accelerate deals per-device batches round-robin: rank r of W gets batches r, r + W, ... With
    accumulation A each step consumes W x A consecutive per-device batches = one global batch."""
    lengths, per_device, world, accum = _lengths(), 4, 6, 2
    gb = per_device * world * accum
    order = GlobalLengthGroupedSampler(lengths, gb, megabatches=10, seed=3).order()
    batches = [order[i:i + per_device] for i in range(0, len(order), per_device)]
    steps = len(batches) // (world * accum)
    for s in range(steps):
        step = batches[s * world * accum:(s + 1) * world * accum]
        ids = np.concatenate(step)
        assert np.array_equal(np.sort(ids), np.sort(order[s * gb:(s + 1) * gb]))


def test_longest_global_batch_first():
    lengths, gb = _lengths(), 96
    order = GlobalLengthGroupedSampler(lengths, gb, megabatches=5, seed=2).order()
    assert lengths[order[:gb]].max() == lengths.max()


def test_short_tail_and_tiny_dataset():
    lengths = _lengths(n=250)
    order = GlobalLengthGroupedSampler(lengths, global_batch=96, megabatches=50).order()
    assert sorted(order.tolist()) == list(range(250))
    assert len(GlobalLengthGroupedSampler(lengths, 96)) == 250


def test_spectrum_lengths_plain_and_selected():
    ds = Dataset.from_dict({"mz": [[1.0] * n for n in (3, 1, 4, 1, 5)]})
    assert spectrum_lengths(ds).tolist() == [3, 1, 4, 1, 5]
    assert spectrum_lengths(ds.select([4, 0, 2])).tolist() == [5, 3, 4]


def test_trainer_uses_sampler_only_when_enabled():
    ds = Dataset.from_dict({"mz": [[1.0] * n for n in range(1, 41)]})
    args = SimpleNamespace(length_grouped_batches=True, per_device_train_batch_size=2,
                           gradient_accumulation_steps=2, world_size=3, length_group_megabatches=2, seed=0)
    stub = SimpleNamespace(args=args, train_dataset=ds)
    sampler = MSDeltaTrainer._get_train_sampler(stub)
    assert isinstance(sampler, GlobalLengthGroupedSampler) and sampler.global_batch == 12
    assert sorted(sampler) == list(range(40))


@pytest.mark.parametrize("bad", [0, -1])
def test_rejects_bad_sizes(bad):
    with pytest.raises(ValueError):
        GlobalLengthGroupedSampler(_lengths(10), global_batch=bad)
