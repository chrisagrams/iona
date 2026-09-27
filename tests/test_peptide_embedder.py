"""The peptide embedder as a standard model (msdelta.models.peptide_embedder), and that it reads
every layout a trained peptide embedder has been saved in so far."""

from __future__ import annotations

import json

import pytest
import torch
from safetensors.torch import save_file

PEPTIDES = ["PEPTIDEK", "AC[57.0215]M[15.9949]K", "[42.0106]SAMPLER", "NQ[0.9840]GLYK"]
CHARGES = [2, 3, 2, 4]


def _config(readout="pool", num_heads=8):
    from msdelta.models.peptide_embedder import PeptideEmbedderConfig
    return PeptideEmbedderConfig(embedding_size=24, hidden_size=16, num_layers=2, num_heads=num_heads,
                                 max_length=32, n_charges=8, mod_n_freqs=4, readout=readout)


def _model(readout="pool", num_heads=8):
    from msdelta.models.peptide_embedder import PeptideEmbedderModel
    torch.manual_seed(0)
    return PeptideEmbedderModel(_config(readout, num_heads)).eval()


def test_old_import_paths_give_the_same_classes():
    from msdelta.models import peptide_embedder as new
    from msdelta.rescoring import reranking
    import msdelta.reranking as old
    for name in ("PeptideEncoder", "PeptideCollator", "parse_peptide", "pool_sequence", "student_readout"):
        assert getattr(old, name) is getattr(new, name) is getattr(reranking, name)


def test_embed_is_unit_norm_and_deterministic():
    m = _model()
    a, b = m.embed(PEPTIDES, CHARGES), m.embed(PEPTIDES, CHARGES)
    assert a.shape == (4, 24)
    assert torch.allclose(a.norm(dim=-1), torch.ones(4), atol=1e-5)
    assert torch.equal(a, b)


def test_save_and_load_round_trip(tmp_path):
    from msdelta.models.peptide_embedder import PeptideEmbedderModel
    m = _model()
    m.save_pretrained(tmp_path)
    assert json.loads((tmp_path / "config.json").read_text())["model_type"] == "msdelta-peptide-embedder"
    again = PeptideEmbedderModel.from_pretrained(tmp_path).eval()
    assert torch.equal(m.embed(PEPTIDES, CHARGES), again.embed(PEPTIDES, CHARGES))


@pytest.mark.parametrize("readout", ["pool", "cls", "attn"])
def test_reads_an_alignment_run_final_dir(tmp_path, readout):
    """final/ of an alignment run: raw weights under sequence_encoder.*, no config.json."""
    from msdelta.models.peptide_embedder import PeptideEmbedderModel
    m = _model(readout)
    save_file({k: v.contiguous() for k, v in m.state_dict().items()}, str(tmp_path / "model.safetensors"))
    loaded = PeptideEmbedderModel.from_pretrained(tmp_path).eval()
    assert loaded.config.readout == readout
    assert loaded.config.embedding_size == 24 and loaded.config.num_layers == 2
    assert torch.equal(m.embed(PEPTIDES, CHARGES), loaded.embed(PEPTIDES, CHARGES))


def test_alignment_dir_with_other_head_count_needs_the_override(tmp_path):
    """num_heads is not in the weights: 8 is assumed, a different count must be passed."""
    from msdelta.models.peptide_embedder import PeptideEmbedderModel
    m = _model(num_heads=2)
    save_file({k: v.contiguous() for k, v in m.state_dict().items()}, str(tmp_path / "model.safetensors"))
    loaded = PeptideEmbedderModel.from_pretrained(tmp_path, num_heads=2).eval()
    assert torch.equal(m.embed(PEPTIDES, CHARGES), loaded.embed(PEPTIDES, CHARGES))


def test_reads_the_first_hub_release_layout(tmp_path):
    """Gaolaboratory/iona-peptide-embedder-400m as first released: model_type iona-peptide-embedder."""
    from msdelta.models.peptide_embedder import PeptideEmbedderModel
    m = _model()
    save_file({k: v.contiguous() for k, v in m.state_dict().items()}, str(tmp_path / "model.safetensors"))
    (tmp_path / "config.json").write_text(json.dumps({
        "model_type": "iona-peptide-embedder", "embedding_size": 24, "hidden_size": 16, "num_layers": 2,
        "num_heads": 8, "max_length": 32, "n_charges": 8, "mod_n_freqs": 4, "pooling": "mean+max",
        "readout": "pool", "spectrum_model": "Gaolaboratory/iona-contrastive-400m"}))
    loaded = PeptideEmbedderModel.from_pretrained(tmp_path).eval()
    assert loaded.config.spectrum_model == "Gaolaboratory/iona-contrastive-400m"
    assert torch.equal(m.embed(PEPTIDES, CHARGES), loaded.embed(PEPTIDES, CHARGES))


def test_rejects_a_directory_without_peptide_weights(tmp_path):
    from msdelta.models.peptide_embedder import PeptideEmbedderModel
    save_file({"encoder.weight": torch.zeros(2, 2)}, str(tmp_path / "model.safetensors"))
    with pytest.raises(ValueError):
        PeptideEmbedderModel.from_pretrained(tmp_path)


def test_alignment_training_saves_the_standard_layout(tmp_path):
    """finetune_align.save_peptide_embedder: the trained student -> a loadable standard model."""
    from types import SimpleNamespace

    from msdelta.finetuning.alignment.finetune_align import save_peptide_embedder
    from msdelta.models.peptide_embedder import PeptideEmbedderModel
    m = _model()
    wrapper = SimpleNamespace(sequence_encoder=m.sequence_encoder)
    args = SimpleNamespace(sequence_hidden_size=16, sequence_num_layers=2, sequence_num_heads=8,
                           max_peptide_length=32, sequence_dropout=0.1, pooling="mean+max",
                           sequence_readout="pool", pretrained_path="/spectrum/encoder")
    save_peptide_embedder(wrapper, args, tmp_path / "peptide_embedder")
    loaded = PeptideEmbedderModel.from_pretrained(tmp_path / "peptide_embedder").eval()
    assert loaded.config.spectrum_model == "/spectrum/encoder"
    assert torch.equal(m.embed(PEPTIDES, CHARGES), loaded.embed(PEPTIDES, CHARGES))
