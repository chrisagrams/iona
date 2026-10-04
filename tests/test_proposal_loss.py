"""K195a-P: the proposal masked-intensity loss and its logging next to today's loss.

What must hold: power 1 IS today's loss (both architectures); the trainer optimises the chosen objective; the logs keep
today's loss under 'loss' / 'eval_loss' and add 'proposal_loss' / 'eval_proposal_loss'; without the option nothing
changes; and msdelta.pretraining.train.main runs end to end with it.
"""

from __future__ import annotations

import math

import pytest
import torch
from datasets import DatasetDict
from transformers import TrainerCallback

from msdelta.models.processing_msdelta import MSDeltaProcessor
from msdelta.pretraining.eval_mlm import IndexedDataset, build_batches
from msdelta.pretraining.proposal_loss import tempered_intensity_kl
from msdelta.pretraining.train import MSDeltaTrainer
from msdelta.pretraining.training_args import MSDeltaTrainingArguments
from test_eval_mlm import ARCHS, N_SPECTRA, TINY, _collator, _model, _split


def _batch():
    return build_batches(_split(), _collator(), N_SPECTRA)[0]


@pytest.mark.parametrize("architecture", ARCHS)
def test_power_one_is_todays_loss(architecture):
    model, batch = _model(architecture).eval(), _batch()
    with torch.no_grad():
        out = model(**batch)
    ours = tempered_intensity_kl(out.logits, batch["labels"], batch["mask_positions"], 1.0)
    assert ours.item() == pytest.approx(out.loss.item(), rel=1e-6)


def test_power_half_matches_a_direct_computation():
    model, batch = _model("transformer").eval(), _batch()
    with torch.no_grad():
        logits = model(**batch).logits
    ours = tempered_intensity_kl(logits, batch["labels"], batch["mask_positions"], 0.5).item()
    total = 0.0
    for row in range(logits.shape[0]):
        m = batch["mask_positions"][row].bool()
        t = batch["labels"][row][m].double().sqrt()
        t = t / t.sum()
        logp = torch.log_softmax(logits[row][m].double(), dim=-1)
        total += float((t * (t.clamp_min(1e-300).log() - logp)).sum())
    assert ours == pytest.approx(total / logits.shape[0], rel=1e-5)
    assert ours != pytest.approx(model(**batch).loss.item(), rel=1e-3)  # it is a different loss


def test_no_masked_peaks_gives_zero():
    logits, labels = torch.randn(2, 5), torch.rand(2, 5)
    assert tempered_intensity_kl(logits, labels, torch.zeros(2, 5, dtype=torch.bool), 0.5).item() == 0.0


class _Logs(TrainerCallback):
    def __init__(self):
        self.logs = []

    def on_log(self, args, state, control, logs=None, **kwargs):
        self.logs.append(dict(logs or {}))


def _train(tmp_path, power, on_proposal, steps=4):
    args = MSDeltaTrainingArguments(
        output_dir=str(tmp_path), use_cpu=True, report_to=[], remove_unused_columns=False,
        dataloader_num_workers=0, probe_execution="off", logarithmic_eval_start_step=None,
        per_device_train_batch_size=4, per_device_eval_batch_size=4, max_steps=steps, logging_steps=1,
        eval_strategy="steps", eval_steps=steps, save_strategy="no", learning_rate=1e-3, seed=0,
        proposal_intensity_power=power, train_on_proposal_loss=on_proposal)
    trainer = MSDeltaTrainer(model=_model("transformer"), args=args, train_dataset=IndexedDataset(_split(16, seed=1)),
                             eval_dataset=IndexedDataset(_split()), data_collator=_collator())
    trainer.can_return_loss = True
    rec = _Logs()
    trainer.add_callback(rec)
    trainer.train()
    train_logs = [l for l in rec.logs if "loss" in l and "eval_loss" not in l]
    eval_logs = [l for l in rec.logs if "eval_loss" in l]
    return trainer, train_logs, eval_logs


@pytest.mark.parametrize("on_proposal", [False, True])
def test_trainer_logs_both_losses_and_optimises_the_chosen_one(tmp_path, on_proposal):
    _, train_logs, eval_logs = _train(tmp_path, 0.5, on_proposal)
    assert len(train_logs) == 4 and len(eval_logs) == 1
    for l in train_logs:
        assert {"loss", "proposal_loss", "objective_loss"} <= set(l)
        assert l["loss"] != pytest.approx(l["proposal_loss"], rel=1e-3)
        # objective_loss is the Trainer's own number for what was optimised
        assert l["objective_loss"] == pytest.approx(l["proposal_loss"] if on_proposal else l["loss"], abs=2e-4)
    e = eval_logs[0]
    assert e["eval_loss_check"] == pytest.approx(e["eval_loss"], rel=1e-5)
    assert math.isfinite(e["eval_proposal_loss"]) and e["eval_proposal_loss"] > 0


def test_objective_changes_the_updates(tmp_path):
    a, _, _ = _train(tmp_path / "a", 0.5, False, steps=2)
    b, _, _ = _train(tmp_path / "b", 0.5, True, steps=2)
    pa, pb = (torch.cat([p.detach().flatten() for p in t.model.parameters()]) for t in (a, b))
    assert not torch.allclose(pa, pb)


def test_default_is_unchanged(tmp_path):
    _, train_logs, eval_logs = _train(tmp_path, None, False, steps=2)
    assert all("proposal_loss" not in l and "objective_loss" not in l for l in train_logs)
    assert "eval_proposal_loss" not in eval_logs[0]


@pytest.mark.parametrize("kwargs", [dict(proposal_intensity_power=0.0), dict(train_on_proposal_loss=True)])
def test_rejects_bad_settings(tmp_path, kwargs):
    with pytest.raises(ValueError):
        MSDeltaTrainingArguments(output_dir=str(tmp_path), use_cpu=True, report_to=[], **kwargs)


def test_main_end_to_end(tmp_path):
    """The real entry point (8884395/8884396 died in main() although unit tests passed)."""
    from msdelta.models.configuration_msdelta import MSDeltaConfig
    from msdelta.pretraining import train

    data = tmp_path / "pre"
    DatasetDict({"train": _split(16, seed=1), "validation": _split()}).save_to_disk(str(data))
    cfg = tmp_path / "cfg"
    MSDeltaConfig(**TINY).save_pretrained(cfg)
    MSDeltaProcessor(max_peaks=24).save_pretrained(cfg)
    out = tmp_path / "run"
    argv = ["--config_name", str(cfg), "--processor_name_or_path", str(cfg), "--output_dir", str(out),
            "--dataset_repo_id", "unused", "--preprocessed_dataset_dir", str(data), "--use_cpu", "true",
            "--report_to", "none", "--remove_unused_columns", "false", "--probe_execution", "off",
            "--bias_curve_steps", "0", "--per_device_train_batch_size", "4", "--per_device_eval_batch_size", "4",
            "--max_steps", "2", "--logging_steps", "1", "--eval_strategy", "steps", "--eval_steps", "2",
            "--save_strategy", "no", "--dataloader_num_workers", "0", "--mask_ratio", "0.5",
            "--proposal_intensity_power", "0.5", "--train_on_proposal_loss", "true"]
    assert train.main(argv) == 0
    assert (out / "final" / "model.safetensors").exists()
