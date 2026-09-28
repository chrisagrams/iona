"""Opt-in per-block timing (K121-P, ``msdelta/utils/block_timing.py``). Tiny CPU models only."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from transformers import HfArgumentParser, Trainer

from msdelta.models.configuration_msdelta import MSDeltaConfig
from msdelta.models.modeling_msdelta import MSDeltaForPreTraining
from msdelta.models.processing_msdelta import MSDeltaDataCollatorForPreTraining
from msdelta.pretraining.training_args import (DataArguments, ModelArguments,
                                               MSDeltaTrainingArguments)
from msdelta.utils.block_timing import BlockTimer, BlockTimingCallback

REPO = Path(__file__).resolve().parents[1]

TINY = dict(hidden_size=32, num_attention_heads=4, num_hidden_layers=2, intermediate_size=64,
            hidden_dropout_prob=0.0, attention_probs_dropout_prob=0.0, delta_bias_n_freqs=8,
            delta_bias_per_head_hidden=4)
TINY_PAIR = dict(architecture="pairformer", pair_channels=8, pair_tri_channels=8,
                 pair_use_triangle_attention=True, pair_tri_attn_heads=2, pair_tri_attn_dim=4,
                 pair_tri_attn_chunk=5, pair_opm_channels=4)


def _rows(n_rows: int = 16, seed: int = 0) -> list[dict]:
    g = torch.Generator().manual_seed(seed)
    rows = []
    for i in range(n_rows):
        k = 8 + i % 5
        intensity = torch.rand(k, generator=g) + 0.01
        rows.append({
            "mz": torch.sort(torch.rand(k, generator=g) * 1500 + 100).values.tolist(),
            "log_intensity": (intensity.log1p() / intensity.log1p().max()).tolist(),
            "labels": (intensity / intensity.sum()).tolist(),
        })
    return rows


def _batch(seed: int = 0) -> dict[str, torch.Tensor]:
    torch.manual_seed(seed)
    return MSDeltaDataCollatorForPreTraining(mask_ratio=0.5)(_rows(4, seed))


def _train(tmp_path: Path, config: MSDeltaConfig, *, timing_steps: int, gc: bool) -> Path:
    parser = HfArgumentParser((MSDeltaTrainingArguments,))
    (args,) = parser.parse_args_into_dataclasses(args=[
        "--output_dir", str(tmp_path), "--use_cpu", "true", "--max_steps", "4",
        "--per_device_train_batch_size", "2", "--gradient_accumulation_steps", "2",
        "--report_to", "none", "--save_strategy", "no", "--eval_strategy", "no",
        "--probe_execution", "off", "--logging_steps", "1", "--remove_unused_columns", "false",
        "--gradient_checkpointing", str(gc).lower(), "--dataloader_num_workers", "0",
        "--block_timing_steps", str(timing_steps), "--block_timing_start_step", "2",
    ])
    model = MSDeltaForPreTraining(config)
    trainer = Trainer(model=model, args=args, train_dataset=_rows(),
                      data_collator=MSDeltaDataCollatorForPreTraining(mask_ratio=0.5))
    callback = BlockTimingCallback(model, args.block_timing_steps, args.block_timing_start_step,
                                   tmp_path / "block_timing.json")
    trainer.add_callback(callback)
    trainer.train()
    assert not callback.timer.attached, "hooks must be removed after the timed window"
    return tmp_path / "block_timing.json"


@pytest.mark.parametrize("gc", [False, True])
def test_pairformer_breakdown(tmp_path, gc):
    out = _train(tmp_path, MSDeltaConfig(**TINY, **TINY_PAIR), timing_steps=2, gc=gc)
    result = json.loads(out.read_text())
    assert result["timed_steps"] == 2 and len(result["step_ms"]) == 2
    assert (result["first_step"], result["last_step"]) == (2, 3)
    assert result["architecture"] == "pairformer" and result["gradient_checkpointing"] is gc
    groups = result["groups_ms_per_step"]
    expected = {"embed", "pair_features", "z_init_proj", "pair_layer_total", "a_writeback",
                "b_tri_mul_out", "c_tri_mul_in", "d_tri_attn_start", "e_tri_attn_end",
                "f_pair_transition", "g_bias_readout", "block_total", "i_single_transition",
                "final_norm", "head"}
    assert expected <= set(groups)
    assert "delta_bias" not in groups and "attention" not in groups
    for name in expected:
        assert groups[name]["fwd"] > 0 and groups[name]["bwd"] > 0, name
    # 2 micro-steps per optimizer step; one call per micro-step for single-module groups.
    assert groups["embed"]["calls_per_step"] == pytest.approx(4)  # fwd + bwd, x2
    assert (groups["b_tri_mul_out"]["recompute"] > 0) is gc
    assert len(result["per_layer_ms_per_step"]["b_tri_mul_out"]["fwd"]) == 2
    assert "h_single_attention" in result["derived_ms_per_step"]
    assert len(result["micro_batch_shapes"]) == 4


def test_transformer_breakdown(tmp_path):
    out = _train(tmp_path, MSDeltaConfig(**TINY), timing_steps=1, gc=False)
    groups = json.loads(out.read_text())["groups_ms_per_step"]
    assert {"embed", "delta_bias", "block_total", "attention", "ffn", "final_norm",
            "head"} <= set(groups)
    assert "pair_layer_total" not in groups


def test_off_by_default_writes_nothing(tmp_path):
    (args,) = HfArgumentParser((MSDeltaTrainingArguments,)).parse_args_into_dataclasses(
        args=["--output_dir", str(tmp_path), "--use_cpu", "true"])
    assert args.block_timing_steps == 0
    with pytest.raises(ValueError):
        BlockTimingCallback(MSDeltaForPreTraining(MSDeltaConfig(**TINY)), 0, 1,
                            tmp_path / "x.json")


@pytest.mark.parametrize("config", [TINY, {**TINY, **TINY_PAIR}], ids=["transformer", "pairformer"])
def test_hooks_do_not_change_numerics(config):
    """Loss and gradients are identical with the timing hooks attached (gc on)."""
    grads = []
    for attach in (False, True):
        torch.manual_seed(0)
        model = MSDeltaForPreTraining(MSDeltaConfig(**config))
        model.gradient_checkpointing_enable()
        model.train()
        timer = BlockTimer(model)
        if attach:
            timer.attach()
        loss = model(**_batch()).loss
        loss.backward()
        timer.detach()
        grads.append((loss.detach(), [p.grad.clone() for p in model.parameters()]))
        if attach:
            assert timer.ms, "no block was timed"
    (loss0, g0), (loss1, g1) = grads
    assert torch.equal(loss0, loss1)
    assert all(torch.allclose(a, b) for a, b in zip(g0, g1))


@pytest.mark.parametrize("arm", ["transformer", "pairformer"])
def test_stage0_args_keep_timing_off(arm):
    """The Stage 0 args files ship with timing off (K121-P not approved)."""
    parser = HfArgumentParser((ModelArguments, DataArguments, MSDeltaTrainingArguments))
    _, _, args = parser.parse_args_into_dataclasses(
        args=["--args_file", str(REPO / f"configs/stage0/{arm}/training.args"),
              "--output_dir", "/nonexistent-unused", "--use_cpu", "true"],
        args_file_flag="--args_file")
    assert args.block_timing_steps == 0
    assert args.gradient_checkpointing is (arm == "pairformer")
