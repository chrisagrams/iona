"""msdelta.pretraining.eval_mlm: the masked-peak validation loss of saved checkpoints.

Tiny CPU models (a transformer and a Pairformer, randomly initialised) on a tiny synthetic
preprocessed split. What must hold for the tool's numbers to be comparable across models:
the masks are a function of (seed, row) only, the loss is exactly the trainer's eval_loss,
and neither the batch size nor the padding width moves it beyond float noise.
"""

from __future__ import annotations

import json

import pytest
import torch
from datasets import Dataset, DatasetDict

from msdelta.models.configuration_msdelta import MSDeltaConfig
from msdelta.models.modeling_msdelta import MSDeltaForPreTraining
from msdelta.models.processing_msdelta import (MSDeltaDataCollatorForPreTraining,
                                               MSDeltaProcessor)
from msdelta.pretraining import eval_mlm
from msdelta.pretraining.eval_mlm import (IndexedDataset, SeededMaskingCollator, batch_digests,
                                          build_batches, evaluate_model)

N_SPECTRA = 11          # not a multiple of any batch size used below: the last batch is partial
MAX_PEAKS = 24

TINY = dict(hidden_size=32, num_attention_heads=4, num_hidden_layers=2, intermediate_size=64,
            delta_bias_n_freqs=8, delta_bias_per_head_hidden=4)
PAIR = dict(architecture="pairformer", pair_channels=8, pair_tri_channels=8,
            pair_use_triangle_attention=True, pair_tri_attn_heads=2, pair_tri_attn_dim=4,
            pair_tri_attn_chunk=5, pair_opm_channels=4, pair_mass_defect_n_freqs=4)


def _split(n: int = N_SPECTRA, seed: int = 0) -> Dataset:
    """Preprocessed rows as msdelta.preprocess writes them (processor output + extra columns)."""
    g = torch.Generator().manual_seed(seed)
    processor = MSDeltaProcessor(max_peaks=MAX_PEAKS)
    rows = {"mz": [], "log_intensity": [], "labels": [], "charge": [], "precursor_mz": []}
    for _ in range(n):
        k = int(torch.randint(3, MAX_PEAKS + 1, (1,), generator=g))
        mz = torch.sort(torch.rand(k, generator=g) * 1500 + 100).values
        intensity = torch.rand(k, generator=g) * 1e4 + 1
        values = processor(mz.tolist(), intensity.tolist(), padding=False, return_labels=True)
        for name in ("mz", "log_intensity", "labels"):
            rows[name].append(values[name])
        rows["charge"].append(2)
        rows["precursor_mz"].append(500.0)
    return Dataset.from_dict(rows)


def _model(architecture: str, seed: int = 0) -> MSDeltaForPreTraining:
    torch.manual_seed(seed)
    extra = PAIR if architecture == "pairformer" else {}
    # Default dropout (0.1) on purpose: evaluation must switch it off.
    return MSDeltaForPreTraining(MSDeltaConfig(**TINY, **extra))


def _collator(seed: int = 0, **kwargs) -> SeededMaskingCollator:
    return SeededMaskingCollator(MSDeltaDataCollatorForPreTraining(mask_ratio=0.5, **kwargs),
                                 seed=seed)


ARCHS = ["transformer", "pairformer"]


class TestMasks:
    def test_masks_depend_only_on_seed_and_row(self):
        split = _split()
        a = batch_digests(build_batches(split, _collator(), 4))
        torch.manual_seed(1234)                      # global RNG state must not matter
        torch.rand(17)
        b = batch_digests(build_batches(split, _collator(), 3))
        c = batch_digests(build_batches(split, _collator(pad_to_multiple_of=32), 1))
        assert a == {**b, "max_peaks_in_data": a["max_peaks_in_data"]} == \
            {**c, "max_peaks_in_data": a["max_peaks_in_data"]}
        assert a["n_spectra"] == N_SPECTRA

    def test_seed_changes_masks(self):
        split = _split()
        a = batch_digests(build_batches(split, _collator(seed=0), 4))
        b = batch_digests(build_batches(split, _collator(seed=1), 4))
        assert a["mask_digest"] != b["mask_digest"]
        assert a["input_digest"] == b["input_digest"]

    def test_prefix_keeps_its_masks(self):
        """--max_spectra N scores the first N rows with the masks of the full run."""
        split = _split()
        full = build_batches(split, _collator(), N_SPECTRA)[0]["mask_positions"]
        head = build_batches(split.select(range(5)), _collator(), 5)[0]["mask_positions"]
        width = head.shape[1]
        assert torch.equal(full[:5, :width], head)
        assert not full[:5, width:].any()

    def test_masking_rule_is_the_collators(self):
        """Same count per row as the trainer's collator: min(len, max(1, round(len * 0.5)))."""
        split = _split()
        batch = build_batches(split, _collator(), N_SPECTRA)[0]
        for row in range(N_SPECTRA):
            length = len(split[row]["mz"])
            assert int(batch["mask_positions"][row].sum()) == min(length, max(1, round(length * 0.5)))
            assert not batch["mask_positions"][row, length:].any()

    def test_global_rng_is_left_alone(self):
        torch.manual_seed(7)
        expected = torch.rand(3)
        torch.manual_seed(7)
        build_batches(_split(), _collator(), 4)
        assert torch.equal(torch.rand(3), expected)


@pytest.mark.parametrize("architecture", ARCHS)
class TestLoss:
    def test_deterministic(self, architecture):
        split = _split()
        first = evaluate_model(_model(architecture), build_batches(split, _collator(), 4),
                               "cpu", "fp32")
        second = evaluate_model(_model(architecture), build_batches(split, _collator(), 4),
                                "cpu", "fp32")
        assert first["eval_loss"] == second["eval_loss"]
        assert first["eval_loss"] > 0

    def test_batching_and_padding_do_not_change_it(self, architecture):
        split = _split()
        model = _model(architecture)
        losses = [evaluate_model(model, build_batches(split, _collator(**kw), bs), "cpu",
                                 "fp32")["eval_loss"]
                  for bs, kw in ((1, {}), (4, {}), (N_SPECTRA, {}), (3, {"pad_to_multiple_of": 40}))]
        assert losses == pytest.approx([losses[0]] * len(losses), rel=1e-5, abs=1e-7)

    def test_equals_model_loss_on_one_batch(self, architecture):
        """One batch holding every spectrum: the tool's number IS the model's own loss."""
        split = _split()
        model = _model(architecture).eval()
        batch = build_batches(split, _collator(), N_SPECTRA)[0]
        with torch.no_grad():
            expected = model(**batch).loss.item()
        assert evaluate_model(model, [batch], "cpu", "fp32")["eval_loss"] == \
            pytest.approx(expected, rel=1e-6)

    @pytest.mark.parametrize("bf16", [False, True])
    def test_equals_trainer_eval_loss(self, architecture, bf16, tmp_path):
        """MSDeltaTrainer.evaluate() (the loop msdelta.train logs eval_loss from) with the
        same collator gives the same number, partial last batch included."""
        from msdelta.pretraining.train import MSDeltaTrainer
        from msdelta.pretraining.training_args import MSDeltaTrainingArguments

        split = _split()
        model = _model(architecture)
        args = MSDeltaTrainingArguments(
            output_dir=str(tmp_path), per_device_eval_batch_size=4, use_cpu=True, bf16=bf16,
            report_to=[], remove_unused_columns=False, dataloader_num_workers=0,
            probe_execution="off", logarithmic_eval_start_step=None)
        trainer = MSDeltaTrainer(model=model, args=args, eval_dataset=IndexedDataset(split),
                                 data_collator=_collator())
        trainer.can_return_loss = True          # as msdelta.pretraining.train does
        metrics = trainer.evaluate()
        ours = evaluate_model(model, build_batches(split, _collator(), 4), "cpu",
                              "bf16" if bf16 else "fp32", batch_size=4)
        assert ours["eval_loss"] == pytest.approx(metrics["eval_loss"], rel=1e-6)
        assert set(ours) <= set(metrics)


class TestCli:
    def test_end_to_end(self, tmp_path):
        data_dir = tmp_path / "preprocessed"
        DatasetDict({"train": _split(3, seed=1), "validation": _split()}).save_to_disk(str(data_dir))
        paths = []
        for architecture in ARCHS:
            path = tmp_path / architecture
            _model(architecture).save_pretrained(path)
            MSDeltaProcessor(max_peaks=16 if architecture == "transformer" else 512) \
                .save_pretrained(path)
            paths.append(str(path))
        out = tmp_path / "out" / "eval.json"
        argv = ["--checkpoints", "+".join(paths), "--dataset", str(data_dir), "--out", str(out),
                "--batch_size", "4", "--precision", "fp32", "--device", "cpu", "--max_spectra", "9"]
        assert eval_mlm.main(argv) == 0
        result = json.loads(out.read_text())
        assert result["n_spectra"] == 9 and result["seed"] == 0 and result["split"] == "validation"
        assert result["code"]
        by_arch = {c["architecture"]: c for c in result["checkpoints"]}
        assert set(by_arch) == set(ARCHS)
        for architecture, entry in by_arch.items():
            assert entry["n_params"] == sum(p.numel() for p in _model(architecture).parameters())
            assert entry["n_spectra"] == 9 and entry["n_masked_tokens"] == result["n_masked_tokens"]
            assert entry["eval_loss"] > 0
        # max_peaks 16 < the longest spectrum: recorded, not changed.
        assert "warning" in by_arch["transformer"] and "warning" not in by_arch["pairformer"]

        # Same run again: identical numbers.
        again = tmp_path / "again.json"
        assert eval_mlm.main(argv[:5] + [str(again)] + argv[6:]) == 0
        repeat = json.loads(again.read_text())
        assert repeat["mask_digest"] == result["mask_digest"]
        assert [c["eval_loss"] for c in repeat["checkpoints"]] == \
            [c["eval_loss"] for c in result["checkpoints"]]

    def test_bad_checkpoint_is_reported_not_fatal(self, tmp_path):
        data_dir = tmp_path / "preprocessed"
        DatasetDict({"validation": _split(4)}).save_to_disk(str(data_dir))
        good = tmp_path / "good"
        _model("transformer").save_pretrained(good)
        out = tmp_path / "eval.json"
        rc = eval_mlm.main(["--checkpoints", f"{tmp_path / 'missing'},{good}", "--dataset",
                            str(data_dir), "--out", str(out), "--device", "cpu",
                            "--precision", "fp32"])
        assert rc == 1
        entries = json.loads(out.read_text())["checkpoints"]
        assert "error" in entries[0] and entries[1]["eval_loss"] > 0

    def test_split_checkpoints(self):
        assert eval_mlm.split_checkpoints("a+b,c") == ["a", "b", "c"]
