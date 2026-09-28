"""Sharded teacher-cache build == the unsharded one, row for row."""

from types import SimpleNamespace

import numpy as np
import torch

# Imported at module level on purpose: collecting a test file ALONE on an Aurora compute
# node segfaults (rc 139, no traceback) unless msdelta is imported during collection --
# test_grouped_retrieval does this and is the only single file that collects cleanly
# (jobs 8860999, 8861019; same family as FT26). The full suite is unaffected because an
# earlier module imports msdelta first. pbs/precompute_align_sharded.pbs runs this file alone.
import msdelta.precompute_align  # noqa: F401


def _tiny_teacher():
    from msdelta.configuration_msdelta import MSDeltaConfig
    from msdelta.modeling_msdelta import MSDeltaForPreTraining
    torch.manual_seed(0)
    return MSDeltaForPreTraining(MSDeltaConfig(
        hidden_size=32, num_attention_heads=4, num_hidden_layers=2, intermediate_size=64,
        delta_bias_n_freqs=8, delta_bias_per_head_hidden=4, hidden_dropout_prob=0.0,
        attention_probs_dropout_prob=0.0))


def _rows(n, seed):
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        k = int(rng.integers(3, 9))
        out.append({"mz": sorted(rng.uniform(100, 1500, k).tolist()),
                    "log_intensity": rng.uniform(0, 1, k).tolist(),
                    "peptide": "PEPTIDE"[: 3 + i % 4] + "K", "charge": 2,
                    "precursor": 500.0 + i})
    return out


def test_sharded_merge_equals_unsharded(tmp_path, monkeypatch):
    from datasets import Dataset
    import msdelta.precompute_align as pa
    from msdelta.reranking import attach_teacher_embeddings

    teacher = _tiny_teacher()
    # precompute_align loads its teacher through load_strict (K94-P); stub that.
    monkeypatch.setattr(pa, "load_strict", lambda cls, *a, **k: teacher)
    splits = {"train": Dataset.from_list(_rows(23, 0)),
              "validation": Dataset.from_list(_rows(7, 1))}
    for name, split in splits.items():
        split.save_to_disk(str(tmp_path / "_flat" / name))

    model_args = SimpleNamespace(pretrained_path="x", pooling="mean+max",
                                 max_peptide_length=64)
    data_args = SimpleNamespace(max_peaks=512, validation_fraction=0.1,
                                dataset_repo="r", dataset_format="grouped",
                                include_consensus=False, exclude_replicate_peptides=True)
    n = 4
    for i in range(n):
        pa._shard(tmp_path, model_args,
                  SimpleNamespace(num_shards=n, shard_index=i, batch_size=3), "cpu")
    pa._merge(tmp_path, model_args, data_args,
              SimpleNamespace(num_shards=n, seed=0, batch_size=3))

    from datasets import load_from_disk
    reference = attach_teacher_embeddings(splits, teacher, "mean+max", batch_size=5,
                                          device="cpu")
    for name in splits:
        merged = load_from_disk(str(tmp_path / name))
        assert merged["peptide"] == reference[name]["peptide"]
        assert merged["precursor"] == reference[name]["precursor"]
        np.testing.assert_allclose(np.array(merged["target"]),
                                   np.array(reference[name]["target"]), atol=1e-5)
    assert (tmp_path / "MANIFEST.txt").exists()


def test_fixed_width_padding_does_not_change_targets():
    """pad_spectra_to only adds masked positions: the targets must not move, so a cache
    built partly with and partly without it is one consistent cache."""
    from datasets import Dataset
    from msdelta.reranking import attach_teacher_embeddings

    teacher = _tiny_teacher()
    split = {"train": Dataset.from_list(_rows(11, 2))}
    a = attach_teacher_embeddings(split, teacher, "mean+max", batch_size=4, device="cpu")
    b = attach_teacher_embeddings(split, teacher, "mean+max", batch_size=4, device="cpu",
                                  pad_spectra_to=16)
    np.testing.assert_allclose(np.array(a["train"]["target"]),
                               np.array(b["train"]["target"]), atol=1e-5)
