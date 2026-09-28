"""Checkpoint loading fails loudly on any mismatch (K94-P). (regression)

A Pairformer checkpoint read by code that only builds the transformer came back from
`from_pretrained` as a mostly random transformer (41 missing / 143 unexpected keys, a
warning only) and was evaluated as if trained. These tests pin that every load path we own
refuses such a checkpoint, and still loads a correct one unchanged.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import load_file, save_file

from msdelta.models.configuration_msdelta import MSDeltaConfig, MSDeltaDenoisingConfig
from msdelta.models.loading import (CheckpointMismatchError, check_architecture, check_keys,
                                    load_strict)
from msdelta.models.modeling_msdelta import MSDeltaForDenoising, MSDeltaForPreTraining

REPO = Path(__file__).resolve().parents[1]


def _tiny_config(**extra) -> MSDeltaConfig:
    return MSDeltaConfig(hidden_size=32, num_attention_heads=4, num_hidden_layers=2,
                         intermediate_size=64, delta_bias_n_freqs=8,
                         delta_bias_per_head_hidden=4, **extra)


def _save(model, path: Path) -> Path:
    model.save_pretrained(str(path))
    return path


def _edit_weights(path: Path, fn) -> None:
    state = load_file(str(path / "model.safetensors"))
    fn(state)
    save_file(state, str(path / "model.safetensors"), metadata={"format": "pt"})


def _edit_config(path: Path, **fields) -> None:
    cfg = json.loads((path / "config.json").read_text())
    cfg.update(fields)
    (path / "config.json").write_text(json.dumps(cfg))


def _encoder_key(state) -> str:
    return next(k for k in state if k.startswith("msdelta.blocks.0."))


@pytest.fixture
def pretrained(tmp_path) -> Path:
    torch.manual_seed(0)
    return _save(MSDeltaForPreTraining(_tiny_config()), tmp_path / "pre")


def _same_weights(a, b) -> bool:
    sa, sb = a.state_dict(), b.state_dict()
    return sa.keys() == sb.keys() and all(torch.equal(sa[k], sb[k]) for k in sa)


# --- the helper --------------------------------------------------------------------------

class TestLoadStrict:
    def test_correct_checkpoint_loads_identically(self, pretrained):
        strict = load_strict(MSDeltaForPreTraining, pretrained)
        plain = MSDeltaForPreTraining.from_pretrained(str(pretrained))
        assert isinstance(strict, MSDeltaForPreTraining)
        assert _same_weights(strict, plain)

    def test_missing_encoder_key_raises(self, pretrained):
        removed = []
        _edit_weights(pretrained, lambda s: removed.append(_encoder_key(s)) or s.pop(removed[0]))
        with pytest.raises(CheckpointMismatchError, match=re.escape(removed[0])):
            load_strict(MSDeltaForPreTraining, pretrained)

    def test_unexpected_key_raises(self, pretrained):
        _edit_weights(pretrained, lambda s: s.__setitem__("pairformer.pair_stack.w", torch.zeros(2)))
        with pytest.raises(CheckpointMismatchError, match="unexpected.*pairformer.pair_stack.w"):
            load_strict(MSDeltaForPreTraining, pretrained)

    def test_shape_mismatch_raises(self, pretrained):
        def grow(s):
            k = _encoder_key(s)
            s[k] = torch.zeros(*(d + 1 for d in s[k].shape))
        _edit_weights(pretrained, grow)
        with pytest.raises((CheckpointMismatchError, RuntimeError)):
            load_strict(MSDeltaForPreTraining, pretrained)

    def test_unexpected_architecture_raises_before_loading(self, pretrained):
        _edit_config(pretrained, architecture="pairformer")
        with pytest.raises(CheckpointMismatchError, match="pairformer"):
            load_strict(MSDeltaForPreTraining, pretrained)

    def test_transformer_architecture_field_is_accepted(self, pretrained):
        _edit_config(pretrained, architecture="transformer")
        load_strict(MSDeltaForPreTraining, pretrained)

    def test_pairformer_shaped_failure_is_refused(self, pretrained):
        """The actual incident: architecture set AND keys that do not match."""
        _edit_config(pretrained, architecture="pairformer")
        _edit_weights(pretrained, lambda s: [s.pop(k) for k in list(s) if ".blocks." in k])
        with pytest.raises(CheckpointMismatchError):
            load_strict(MSDeltaForPreTraining, pretrained)

    def test_allowed_new_head_does_not_raise(self, pretrained):
        """A head absent from the checkpoint is fine only when the call site says so."""
        _edit_weights(pretrained, lambda s: [s.pop(k) for k in list(s)
                                             if k.startswith("intensity_head.")])
        with pytest.raises(CheckpointMismatchError, match="intensity_head"):
            load_strict(MSDeltaForPreTraining, pretrained)
        model = load_strict(MSDeltaForPreTraining, pretrained, allow_missing=("intensity_head.*",))
        assert isinstance(model, MSDeltaForPreTraining)

    def test_allow_list_is_narrow(self, pretrained):
        """Allowing the head does not excuse an encoder key."""
        def drop(s):
            s.pop(_encoder_key(s))
            for k in [k for k in s if k.startswith("intensity_head.")]:
                s.pop(k)
        _edit_weights(pretrained, drop)
        with pytest.raises(CheckpointMismatchError, match="msdelta.blocks.0"):
            load_strict(MSDeltaForPreTraining, pretrained, allow_missing=("intensity_head.*",))

    def test_allowed_unexpected_head(self, tmp_path):
        """A pretraining checkpoint's intensity head read into a model without one."""
        torch.manual_seed(0)
        path = _save(MSDeltaForPreTraining(_tiny_config()), tmp_path / "pre")
        from msdelta.models.modeling_msdelta import MSDeltaModel
        # MSDeltaModel's own keys have no "msdelta." prefix; transformers strips the base
        # prefix, so the only leftover is the head.
        with pytest.raises(CheckpointMismatchError, match="intensity_head"):
            load_strict(MSDeltaModel, path)
        load_strict(MSDeltaModel, path, allow_unexpected=("intensity_head.*",))


class TestCheckArchitecture:
    def test_absent_and_transformer_pass(self):
        check_architecture(_tiny_config())
        check_architecture(_tiny_config(architecture="transformer"))

    def test_unknown_value_raises(self):
        with pytest.raises(CheckpointMismatchError, match="pairformer"):
            check_architecture(_tiny_config(architecture="pairformer"))

    def test_nested_encoder_config_is_checked(self):
        cfg = MSDeltaDenoisingConfig(encoder=_tiny_config(architecture="pairformer"))
        with pytest.raises(CheckpointMismatchError, match="pairformer"):
            check_architecture(cfg)

    def test_config_class_that_declares_the_field_is_trusted(self):
        """Once the code itself knows `architecture` (the Pairformer branch), its config
        class validates the value and builds the matching encoder."""
        class Declaring(MSDeltaConfig):
            def __init__(self, architecture: str = "transformer", **kw):
                super().__init__(**kw)
                self.architecture = architecture
        check_architecture(Declaring(architecture="pairformer", hidden_size=32,
                                     num_attention_heads=4))

    def test_check_keys_error_type(self):
        with pytest.raises(SystemExit):
            check_keys(["a"], [], error=SystemExit)
        check_keys(["x.mask_token"], [], allow_missing=("*mask_token",))


# --- every entry point's loader -----------------------------------------------------------

def _denoise_args(random_init=False):
    return SimpleNamespace(random_init=random_init, head_hidden_size=16, head_dropout=0.1)


class TestEntryPoints:
    def test_denoise_builder_loads_correct_checkpoint(self, pretrained):
        from msdelta.finetuning.denoise.finetune_denoise import build_denoising_model
        model, source = build_denoising_model(str(pretrained), _denoise_args())
        plain = MSDeltaForPreTraining.from_pretrained(str(pretrained))
        assert _same_weights(source, plain)
        assert isinstance(model, MSDeltaForDenoising)   # fresh head, no allow-list needed

    def test_denoise_builder_refuses_missing_key(self, pretrained):
        from msdelta.finetuning.denoise.finetune_denoise import build_denoising_model
        _edit_weights(pretrained, lambda s: s.pop(_encoder_key(s)))
        with pytest.raises(CheckpointMismatchError):
            build_denoising_model(str(pretrained), _denoise_args())

    @pytest.mark.parametrize("random_init", [False, True])
    def test_denoise_builder_refuses_foreign_architecture(self, pretrained, random_init):
        """random_init too: a transformer 'control' of a Pairformer run is no control."""
        from msdelta.finetuning.denoise.finetune_denoise import build_denoising_model
        _edit_config(pretrained, architecture="pairformer")
        with pytest.raises(CheckpointMismatchError, match="pairformer"):
            build_denoising_model(str(pretrained), _denoise_args(random_init))

    def test_eval_checkpoint_load_weights(self, tmp_path, tiny_denoising_config):
        from msdelta.eval.eval_checkpoint import load_weights
        torch.manual_seed(0)
        path = _save(MSDeltaForDenoising(tiny_denoising_config), tmp_path / "den")
        target = MSDeltaForDenoising(tiny_denoising_config)
        load_weights(target, path)
        # mask_token may be absent (allowed); anything else may not.
        _edit_weights(path, lambda s: s.pop("msdelta.embed.mask_token"))
        load_weights(target, path)
        _edit_weights(path, lambda s: s.pop(_encoder_key(s)))
        with pytest.raises(SystemExit, match="missing"):
            load_weights(target, path)

    def test_denoise_checkpoint_mask_token_allowance(self, tmp_path, tiny_denoising_config):
        """eval_denoise_length's allow-list: the frozen, never-read mask token only."""
        torch.manual_seed(0)
        path = _save(MSDeltaForDenoising(tiny_denoising_config), tmp_path / "den")
        _edit_weights(path, lambda s: s.pop("msdelta.embed.mask_token"))
        with pytest.raises(CheckpointMismatchError):
            load_strict(MSDeltaForDenoising, path)
        load_strict(MSDeltaForDenoising, path, allow_missing=("msdelta.embed.mask_token",))

    def test_contrastive_final_encoder_loads(self, tmp_path):
        """finetune_contrastive saves the inner MSDeltaForPreTraining as final/; every
        retrieval eval reads it back through load_strict."""
        from msdelta.finetuning.contrastive.contrastive import MSDeltaForContrastive
        torch.manual_seed(0)
        model = MSDeltaForContrastive(MSDeltaForPreTraining(_tiny_config()), None, kl_weight=0)
        path = _save(model.model, tmp_path / "final")
        assert _same_weights(load_strict(MSDeltaForPreTraining, path), model.model)

    def test_peptide_encoder_standard_layout_is_strict(self, tmp_path):
        from msdelta.models.peptide_encoder import PeptideEncoderConfig, PeptideEncoderModel
        torch.manual_seed(0)
        cfg = PeptideEncoderConfig(embedding_size=16, hidden_size=32, num_layers=1, num_heads=4)
        path = _save(PeptideEncoderModel(cfg), tmp_path / "pep")
        assert isinstance(load_strict(PeptideEncoderModel, path), PeptideEncoderModel)
        PeptideEncoderModel.from_pretrained(path)
        _edit_weights(path, lambda s: s.pop(next(k for k in s if ".encoder.layers.0." in k)))
        with pytest.raises(CheckpointMismatchError, match="missing"):
            PeptideEncoderModel.from_pretrained(path)

    def test_no_unguarded_model_loads_remain(self):
        """Every MSDelta model load in the package goes through load_strict. A new bare
        `MSDeltaFor*.from_pretrained(` call would bring back the silent-random failure."""
        offenders = []
        for py in (REPO / "msdelta").rglob("*.py"):
            for n, line in enumerate(py.read_text().splitlines(), 1):
                if re.search(r"MSDelta(For\w+|Model)\.from_pretrained\(", line):
                    offenders.append(f"{py.relative_to(REPO)}:{n}: {line.strip()}")
        assert not offenders, "\n".join(offenders)


# --- real checkpoints (CPU, 25-400M params; skipped where the paths are absent) ------------

REAL_PRETRAINED = Path("/flare/UIC-HPC/khuss/msdelta/pretrained/"
                       "msdelta-25m-production-01-checkpoint-540423")
REAL_CONTRASTIVE = Path("/lus/flare/projects/UIC-HPC/khuss/msdelta/runs/"
                        "sweep-s050m_ck540k_supcon_mass_seed0-8872141/final")
HF_HOME = Path("/lus/flare/projects/UIC-HPC/khuss/msdelta/huggingface")


@pytest.mark.parametrize("path", [REAL_PRETRAINED, REAL_CONTRASTIVE], ids=["pretrained", "contrastive"])
def test_real_spectrum_checkpoints_load(path):
    if not os.access(path / "model.safetensors", os.R_OK):
        pytest.skip(f"not readable: {path}")
    model = load_strict(MSDeltaForPreTraining, path)
    assert sum(p.numel() for p in model.parameters()) > 1_000_000


def test_real_peptide_encoder_loads(monkeypatch):
    snapshots = HF_HOME / "hub" / "models--Gaolaboratory--iona-peptide-embedder-400m" / "snapshots"
    if not snapshots.is_dir():
        pytest.skip(f"not cached: {snapshots}")
    from msdelta.models.peptide_encoder import PeptideEncoderModel
    path = sorted(snapshots.iterdir())[-1]
    model = PeptideEncoderModel.from_pretrained(path)
    assert sum(p.numel() for p in model.parameters()) > 1_000_000
