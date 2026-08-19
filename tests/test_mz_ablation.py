import unittest

import torch

from msdelta.model import DeltaMZBias, MSEncoder, ModelConfig
from msdelta.train import MSDeltaForPretraining, parameter_counts


class MZRepresentationAblationTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.mz = torch.tensor([[101.0, 203.5, 307.25, 411.0],
                                [79.0, 155.5, 222.0, 0.0]])
        self.log_int = torch.tensor([[-0.1, -0.5, -1.0, -1.5],
                                     [-0.2, -0.7, -1.2, 0.0]])
        self.padding = torch.tensor([[False, False, False, False],
                                     [False, False, False, True]])
        self.masked = torch.tensor([[True, True, False, False],
                                    [False, True, False, False]])

    @staticmethod
    def model(absolute: bool, delta: bool) -> MSEncoder:
        return MSEncoder(ModelConfig(
            d_model=32, n_heads=4, n_layers=2, dropout=0.0,
            use_absolute_mz=absolute, use_delta_mz_bias=delta,
        )).eval()

    def test_all_variants_are_permutation_equivariant(self):
        perm = torch.tensor([2, 0, 3, 1])
        for absolute in (False, True):
            for delta in (False, True):
                with self.subTest(absolute=absolute, delta=delta):
                    model = self.model(absolute, delta)
                    with torch.no_grad():
                        y = model(self.mz, self.log_int, self.padding, self.masked)
                        yp = model(self.mz[:, perm], self.log_int[:, perm],
                                   self.padding[:, perm], self.masked[:, perm])
                    torch.testing.assert_close(yp, y[:, perm], rtol=1e-5, atol=1e-6)

    def test_model_a_is_independent_of_mz(self):
        model = self.model(False, False)
        with torch.no_grad():
            y1 = model(self.mz, self.log_int, self.padding, self.masked)
            y2 = model(self.mz * 3.7 + 19.0, self.log_int, self.padding, self.masked)
        torch.testing.assert_close(y1, y2)

    def test_model_b_does_not_instantiate_delta_bias(self):
        model = self.model(True, False)
        self.assertIsNone(model.bias_module)
        self.assertFalse(any(isinstance(m, DeltaMZBias) for m in model.modules()))
        counts = parameter_counts(MSDeltaForPretraining(model.cfg), "B")
        self.assertEqual(counts["delta_bias_parameters"], 0)
        self.assertGreater(counts["absolute_mz_parameters"], 0)

    def test_model_c_is_translation_invariant_in_mz(self):
        model = self.model(False, True)
        with torch.no_grad():
            y1 = model(self.mz, self.log_int, self.padding, self.masked)
            y2 = model(self.mz + 16.0, self.log_int, self.padding, self.masked)
        torch.testing.assert_close(y1, y2, rtol=1e-5, atol=1e-6)

    def test_masked_peak_identity(self):
        for absolute, expected_equal in ((False, True), (True, False)):
            embed = self.model(absolute, False).embed
            with torch.no_grad():
                tokens = embed(self.mz[:1], self.log_int[:1], self.masked[:1])
            are_equal = torch.equal(tokens[0, 0], tokens[0, 1])
            self.assertEqual(are_equal, expected_equal)

    def test_padding_mask_is_retained_without_delta_bias(self):
        model = self.model(True, False)
        changed_padding = self.log_int.clone()
        changed_padding[1, 3] = 100.0
        with torch.no_grad():
            y1 = model(self.mz, self.log_int, self.padding, self.masked)
            y2 = model(self.mz, changed_padding, self.padding, self.masked)
        torch.testing.assert_close(y1[1, :3], y2[1, :3])


if __name__ == "__main__":
    unittest.main()
