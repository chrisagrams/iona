"""Regression coverage for nested retrieval training isolation."""

import random
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import torch
from accelerate.utils import DistributedType
from datasets import Dataset
from transformers import Trainer, TrainingArguments

from msdelta.callbacks import RetrievalProbeCallback
from msdelta.configuration_msdelta import MSDeltaConfig
from msdelta.modeling_msdelta import MSDeltaForPreTraining
from msdelta.retrieval import run_retrieval_probe
from msdelta.train import MSDeltaTrainer
from msdelta.training_args import MSDeltaTrainingArguments


class ProbeIsolationTests(unittest.TestCase):
    def test_plugins_cover_each_enabled_probe(self):
        for denoise, retrieval in [(False, False), (True, False), (False, True), (True, True)]:
            for pretrain in [None, object()]:
                with self.subTest(denoise=denoise, retrieval=retrieval, pretrain=pretrain):
                    trainer = object.__new__(MSDeltaTrainer)
                    trainer.use_denoising_probe = denoise
                    trainer.use_retrieval_probe = retrieval
                    trainer.args = SimpleNamespace(deepspeed={})
                    with (
                        patch.object(
                            Trainer,
                            "_build_accelerator_args",
                            return_value={"deepspeed_plugin": pretrain},
                        ),
                        patch("msdelta.train.DeepSpeedPlugin", side_effect=lambda **kw: object()),
                    ):
                        plugins = trainer._build_accelerator_args()["deepspeed_plugin"]
                    if pretrain is None or not (denoise or retrieval):
                        self.assertIs(plugins, pretrain)
                    else:
                        expected = {"pretrain"}
                        if denoise:
                            expected.add("denoise")
                        if retrieval:
                            expected.add("retrieval")
                        self.assertEqual(set(plugins), expected)
                        self.assertIs(plugins["pretrain"], pretrain)
                        self.assertEqual(len({id(p) for p in plugins.values()}), len(expected))

    def test_callback_restores_pretrain_plugin_even_on_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = MSDeltaTrainingArguments(output_dir=tmp, use_cpu=True, report_to=[])
            callback = RetrievalProbeCallback(
                torch.nn.Linear(1, 1), 1, {"train": [], "validation": []}, None, args, Path(tmp)
            )
            for fails in [False, True]:
                with self.subTest(fails=fails):
                    state = Mock(
                        distributed_type=DistributedType.DEEPSPEED,
                        deepspeed_plugins={"pretrain": object(), "retrieval": object()},
                    )

                    def probe(*args, **kwargs):
                        self.assertEqual(
                            state.select_deepspeed_plugin.call_args.args, ("retrieval",)
                        )
                        self.assertIs(kwargs["training_args"], callback.probe_training_args)
                        if fails:
                            raise RuntimeError("probe failed")
                        return {}

                    callback.last_step = -1
                    with (
                        patch("msdelta.callbacks.AcceleratorState", return_value=state),
                        patch("msdelta.callbacks.run_retrieval_probe", side_effect=probe),
                        patch("msdelta.callbacks.TrainingArguments", side_effect=AssertionError),
                    ):
                        outer_state = SimpleNamespace(global_step=1, is_world_process_zero=False)
                        if fails:
                            with self.assertRaisesRegex(RuntimeError, "probe failed"):
                                callback.on_step_end(args, outer_state, None)
                        else:
                            callback.on_step_end(args, outer_state, None)
                    self.assertEqual(
                        [call.args[0] for call in state.select_deepspeed_plugin.call_args_list],
                        ["retrieval", "pretrain"],
                    )

    def test_repeated_real_probe_preserves_encoder_and_rng(self):
        model = MSDeltaForPreTraining(
            MSDeltaConfig(
                hidden_size=8,
                num_attention_heads=2,
                num_hidden_layers=1,
                intermediate_size=16,
                delta_bias_n_freqs=4,
                delta_bias_per_head_hidden=4,
            )
        )
        model.train()
        original = {name: tensor.clone() for name, tensor in model.state_dict().items()}
        dataset = Dataset.from_list(
            [
                {
                    "mz": [[100.0 + i, 200.0 + j] for j in range(4)],
                    "log_intensity": [[0.5, 1.0] for _ in range(4)],
                }
                for i in range(2)
            ]
        )
        with tempfile.TemporaryDirectory() as tmp:
            args = TrainingArguments(
                output_dir=tmp,
                use_cpu=True,
                report_to=[],
                num_train_epochs=1,
                per_device_train_batch_size=2,
                per_device_eval_batch_size=2,
                save_strategy="no",
                logging_strategy="no",
                remove_unused_columns=False,
                label_names=["group_ids"],
                prediction_loss_only=True,
                disable_tqdm=True,
            )
            python_rng = random.getstate()
            numpy_rng = np.random.get_state()
            torch_rng = torch.get_rng_state().clone()
            for step in range(2):
                destination = Path(tmp) / f"step-{step}"
                metrics = run_retrieval_probe(
                    model,
                    dataset,
                    dataset,
                    output_dir=destination,
                    processor=None,
                    replicate_repo_id=None,
                    projection_hidden_size=8,
                    embedding_size=4,
                    dropout=0.0,
                    temperature=0.07,
                    validation_analytes=2,
                    training_args=args,
                )
                self.assertTrue(np.isfinite(metrics["retrieval/loss"]))
                self.assertTrue((destination / "config.json").is_file())
                self.assertEqual(args.output_dir, tmp)
                self.assertTrue(model.training)
                self.assertTrue(all(p.requires_grad for p in model.parameters()))
                for name, tensor in model.state_dict().items():
                    torch.testing.assert_close(tensor, original[name], rtol=0, atol=0)
                self.assertEqual(random.getstate(), python_rng)
                np.testing.assert_equal(np.random.get_state(), numpy_rng)
                self.assertTrue(torch.equal(torch.get_rng_state(), torch_rng))
                # The live encoder remains available for the next pretraining backward pass.
                model.zero_grad()
                batch = torch.tensor([[100.0, 200.0]])
                model.msdelta(
                    mz=batch, log_intensity=torch.ones_like(batch)
                ).last_hidden_state.square().sum().backward()
                self.assertTrue(any(p.grad is not None for p in model.msdelta.parameters()))
                torch.set_rng_state(torch_rng)


if __name__ == "__main__":
    unittest.main()
