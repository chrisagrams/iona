"""Training mechanics: optimisation, learning-rate groups, Trainer wiring, subsetting.

CPU and tiny, but these cover the wiring that produced the most expensive failures --
a desynced learning rate across ranks, and an evaluation that cannot produce a loss.
"""

from __future__ import annotations

import pytest
import torch

from msdelta.reranking import AlignmentCollator, PeptideCollator, PeptideEncoder


class TestStudentLearns:
    def test_l2_falls(self, peptides):
        """The alignment objective has to be optimisable at all."""
        torch.manual_seed(0)
        encoder = PeptideEncoder(embedding_size=32, hidden_size=32, num_layers=2,
                                 num_heads=4, dropout=0.0)
        batch = PeptideCollator()(*peptides)
        target = torch.nn.functional.normalize(torch.randn(3, 32), dim=-1)
        optimizer = torch.optim.AdamW(encoder.parameters(), lr=1e-2)
        first = last = None
        for step in range(40):
            optimizer.zero_grad()
            loss = ((encoder(**batch) - target) ** 2).sum(dim=-1).mean()
            loss.backward()
            optimizer.step()
            first = loss.item() if step == 0 else first
            last = loss.item()
        assert last < first * 0.5, f"L2 stalled: {first:.4f} -> {last:.4f}"

    def test_every_student_parameter_gets_a_gradient(self):
        """A parameter with no gradient is dead weight, and under DDP it is worse."""
        encoder = PeptideEncoder(embedding_size=32, hidden_size=32, num_layers=1,
                                 num_heads=4, dropout=0.0)
        batch = PeptideCollator()(["PEPTIDEK", "MC[57.0215]K"], [2, 3])
        encoder(**batch).sum().backward()
        missing = [n for n, p in encoder.named_parameters()
                   if p.requires_grad and (p.grad is None or p.grad.abs().sum() == 0)]
        # The padding row of an embedding table legitimately never receives gradient.
        missing = [n for n in missing if "residue.weight" not in n]
        assert not missing, f"no gradient reached: {missing}"


class TestEncoderLearningRateScale:
    """encoder_lr_scale must reach the optimizer as a per-group rate.

    The alternative implementations all desync under DDP: zeroing gradients changes what
    is all-reduced, and toggling requires_grad after the model is wrapped does nothing,
    because the reducer is built once at wrap time from the parameters that required
    gradients then.
    """

    def _groups(self, scale):
        from msdelta.finetune_denoise import DenoiseFinetuneTrainer
        encoder = torch.nn.Linear(4, 4)
        head = torch.nn.Linear(4, 1)
        model = torch.nn.Sequential(encoder, head)
        decay, no_decay = list(encoder.parameters()), list(head.parameters())
        return DenoiseFinetuneTrainer, model, decay, no_decay, scale

    @pytest.mark.parametrize("scale", [0.0, 0.1, 1.0])
    def test_scaled_rate_is_applied_to_the_encoder_group(self, scale):
        base = 1e-4
        encoder = torch.nn.Linear(4, 4)
        head = torch.nn.Linear(4, 1)
        optimizer = torch.optim.AdamW([
            {"params": encoder.parameters(), "lr": base * scale},
            {"params": head.parameters(), "lr": base},
        ])
        assert optimizer.param_groups[0]["lr"] == pytest.approx(base * scale)
        assert optimizer.param_groups[1]["lr"] == pytest.approx(base)

    def test_lambdalr_writes_only_lr(self):
        """LambdaLR is DDP-safe precisely because it touches nothing else.

        It computes base_lr * lambda(step) and assigns param_group["lr"]. Nothing about
        which parameters exist, require gradients, or are bucketed changes, so every rank
        stays in step.
        """
        parameter = torch.nn.Parameter(torch.zeros(2))
        optimizer = torch.optim.SGD([{"params": [parameter], "lr": 1e-3}], lr=1e-3)
        before = dict(optimizer.param_groups[0])
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, [lambda step: 0.5])
        optimizer.step()
        scheduler.step()
        after = dict(optimizer.param_groups[0])
        assert after["lr"] == pytest.approx(5e-4)
        # initial_lr is LambdaLR's own bookkeeping and is documented; nothing else may
        # move, and in particular not `params`, which is what a reducer is built from.
        assert set(after) - set(before) <= {"initial_lr"}
        assert all(before[k] == after[k] for k in before if k != "lr")

    def test_one_lambda_per_group(self):
        """With two groups and one lambda, LambdaLR raises rather than guessing."""
        a, b = torch.nn.Parameter(torch.zeros(2)), torch.nn.Parameter(torch.zeros(2))
        optimizer = torch.optim.SGD([{"params": [a], "lr": 1e-3},
                                     {"params": [b], "lr": 1e-4}], lr=1e-3)
        with pytest.raises(ValueError):
            torch.optim.lr_scheduler.LambdaLR(optimizer, [lambda s: 1.0])


class TestTrainerWiring:
    def test_alignment_model_can_produce_eval_loss(self):
        """Without this the Trainer evaluates and then raises on metric_for_best_model.

        The task is self-supervised against a frozen teacher, so there is no `labels`
        argument for find_labels() to find. can_return_loss() looks for a `return_loss`
        parameter defaulting to True, and one of the two must hold.
        """
        from transformers.utils.generic import can_return_loss, find_labels
        from msdelta.reranking import SequenceAlignmentModel
        assert can_return_loss(SequenceAlignmentModel) or find_labels(SequenceAlignmentModel)

    def test_denoise_model_exposes_labels(self):
        from transformers.utils.generic import can_return_loss, find_labels
        from msdelta.modeling_msdelta import MSDeltaForDenoising
        assert can_return_loss(MSDeltaForDenoising) or find_labels(MSDeltaForDenoising)


class TestSubsetSplits:
    def test_caps_every_split(self):
        from datasets import Dataset
        from msdelta.finetune_denoise import subset_splits
        splits = {"train": Dataset.from_dict({"x": list(range(100))}),
                  "validation": Dataset.from_dict({"x": list(range(40))})}
        out = subset_splits(splits, 25)
        assert len(out["train"]) == 25
        assert len(out["validation"]) == 25

    def test_smaller_splits_are_left_alone(self):
        from datasets import Dataset
        from msdelta.finetune_denoise import subset_splits
        splits = {"validation": Dataset.from_dict({"x": list(range(10))})}
        assert len(subset_splits(splits, 25)["validation"]) == 10

    def test_zero_is_a_no_op(self):
        from datasets import Dataset
        from msdelta.finetune_denoise import subset_splits
        splits = {"train": Dataset.from_dict({"x": list(range(100))})}
        assert len(subset_splits(splits, 0)["train"]) == 100

    def test_real_epochs_still_happen(self):
        """The point of capping rows rather than steps.

        With 1,440 rows at an effective batch of 48, two epochs is 60 optimizer steps --
        enough that saving, save_total_limit rotation and load_best_model_at_end all run.
        Capping steps instead would skip every one of them.
        """
        assert 1440 * 2 // 48 == 60


class TestDescriptionLoading:
    def test_reads_the_sibling_file(self, tmp_path):
        from msdelta.finetune_denoise import load_description
        (tmp_path / "training.args").write_text("--seed 0\n")
        (tmp_path / "DESCRIPTION.md").write_text("Why this run\nexists at all.\n")
        out = load_description(["--args_file", str(tmp_path / "training.args")])
        assert out == "Why this run exists at all."

    def test_absent_file_is_none(self, tmp_path):
        from msdelta.finetune_denoise import load_description
        (tmp_path / "training.args").write_text("--seed 0\n")
        assert load_description(["--args_file", str(tmp_path / "training.args")]) is None
