import pytest
import torch

from msdelta.model import DeltaBiasConfig, DeltaMZBias


def test_delta_mz_bias_is_zero_initialized_and_has_expected_shape():
    module = DeltaMZBias(3, DeltaBiasConfig())
    mz = torch.tensor([[100.0, 100.05, 157.02]])

    bias = module(mz)

    assert bias.shape == (1, 3, 3, 3)
    assert torch.count_nonzero(bias) == 0


def test_delta_mz_bias_uses_signed_quantized_buckets():
    module = DeltaMZBias(
        2, DeltaBiasConfig(n_buckets=16, resolution=0.5, max_distance=20.0))
    dm = torch.tensor([0.49, 0.51, -0.51, 100.0])

    buckets = module._bucket(dm)

    assert buckets[0] == 9
    assert buckets[1] == 9
    assert buckets[2] == 1
    assert buckets[3] == 15  # distances beyond max_distance saturate


@pytest.mark.parametrize(
    "config",
    [
        DeltaBiasConfig(n_buckets=3),
        DeltaBiasConfig(n_buckets=15),
        DeltaBiasConfig(resolution=0.0),
        DeltaBiasConfig(max_distance=0.0),
        DeltaBiasConfig(n_buckets=16, resolution=1.0, max_distance=4.0),
    ],
)
def test_delta_mz_bias_rejects_invalid_bucket_configs(config):
    with pytest.raises(ValueError):
        DeltaMZBias(2, config)
