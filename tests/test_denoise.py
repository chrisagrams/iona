import torch

from msdelta.data import PreprocessConfig
from msdelta.denoise import (
    DenoiseCollator,
    MSDeltaForDenoising,
    parse_args,
    preprocess_labeled_spectrum,
)
from msdelta.model import MSEncoder, ModelConfig


def test_preprocessing_keeps_top_n_labels_aligned():
    row = {
        "mz": [100.0, 200.0, 300.0],
        "intensity": [1.0, 10.0, 5.0],
        "signal": [False, True, False],
    }
    result = preprocess_labeled_spectrum(row, PreprocessConfig(0.0, 2))
    assert result["mz"] == [200.0, 300.0]
    assert result["labels"] == [1.0, 0.0]


def test_only_classifier_receives_gradients():
    cfg = ModelConfig(d_model=16, n_heads=2, n_layers=1, ffn_mult=2, max_peaks=4)
    model = MSDeltaForDenoising(MSEncoder(cfg), hidden_dim=4)
    batch = DenoiseCollator()([
        {"mz": [100.0, 200.0], "log_int": [1.0, 0.5], "labels": [1.0, 0.0]},
        {"mz": [150.0], "log_int": [1.0], "labels": [1.0]},
    ])
    output = model(**batch)
    output["loss"].backward()

    assert output["logits"].shape == batch["labels"].shape
    assert all(parameter.grad is None for parameter in model.encoder.parameters())
    assert any(parameter.grad is not None for parameter in model.classifier.parameters())


def test_run_name_argument():
    args = parse_args([
        "--encoder-config", "encoder.yaml",
        "--encoder-checkpoint", "checkpoint",
        "--run-name", "v15-denoise-test",
    ])
    assert args.run_name == "v15-denoise-test"
