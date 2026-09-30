"""Check the masked-intensity pretraining loss on edge-case masks."""

import unittest

import torch

from iona.configuration_iona import IonaConfig
from iona.modeling_iona import IonaForPreTraining


def tiny_model() -> IonaForPreTraining:
    torch.manual_seed(0)
    config = IonaConfig(
        hidden_size=32,
        num_attention_heads=2,
        num_hidden_layers=1,
        intermediate_size=64,
        delta_bias_n_freqs=8,
        delta_bias_per_head_hidden=4,
    )
    return IonaForPreTraining(config).eval()


def batch(mask_positions: torch.Tensor) -> dict[str, torch.Tensor]:
    batch_size, length = mask_positions.shape
    return {
        "mz": torch.rand(batch_size, length) * 1000.0,
        "log_intensity": torch.rand(batch_size, length),
        "attention_mask": torch.ones(batch_size, length, dtype=torch.long),
        "mask_positions": mask_positions,
        "labels": torch.rand(batch_size, length),
    }


class PretrainingLossTests(unittest.TestCase):
    def assert_finite_grads(self, model: IonaForPreTraining) -> None:
        for name, parameter in model.named_parameters():
            if parameter.grad is not None:
                self.assertTrue(torch.isfinite(parameter.grad).all(), name)

    def test_empty_mask_gives_zero_loss_and_finite_grads(self):
        model = tiny_model()
        mask = torch.zeros(2, 5, dtype=torch.bool)
        loss = model(**batch(mask)).loss
        self.assertEqual(loss.item(), 0.0)
        loss.backward()
        self.assert_finite_grads(model)

    def test_row_without_masked_peaks_gives_finite_grads(self):
        model = tiny_model()
        mask = torch.tensor([[True, False, True, False, False], [False] * 5])
        loss = model(**batch(mask)).loss
        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(loss.item(), 0.0)
        loss.backward()
        self.assert_finite_grads(model)

    def test_loss_compiles_without_graph_breaks(self):
        model = tiny_model()
        inputs = batch(torch.tensor([[True, False, True, False, False]]))
        compiled = torch.compile(model, backend="eager", fullgraph=True)
        torch.testing.assert_close(compiled(**inputs).loss, model(**inputs).loss)


if __name__ == "__main__":
    unittest.main()
