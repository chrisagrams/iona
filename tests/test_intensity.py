"""Check the Prosit ion ladder, the intensity head, and its data utilities."""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from pyteomics import mass
from transformers import TrainingArguments

from msdelta.configuration_msdelta import MSDeltaConfig, MSDeltaIntensityPredictionConfig
from msdelta.intensity import (
    IntensityTrainer,
    MSDeltaIntensityProcessor,
    PrositIntensityDataset,
    fit_intensity_prior,
    prosit_fragment_ladder,
)
from msdelta.modeling_msdelta import MSDeltaForIntensityPrediction, MSDeltaModel

ALPHABET = "ACDEFGHIKLMNPQRSTVWY"
ENCODER = MSDeltaConfig(
    hidden_size=32,
    num_attention_heads=4,
    num_hidden_layers=2,
    intermediate_size=64,
    delta_bias_n_freqs=16,
    delta_bias_per_head_hidden=8,
)


def encode(peptide: str) -> list[int]:
    return [ALPHABET.index(residue) + 1 for residue in peptide] + [0] * (30 - len(peptide))


def make_rows(peptides: list[str], charges: list[int]) -> dict[str, torch.Tensor]:
    """Prosit-format rows, as stored on disk, with random labels on possible ions."""
    generator = torch.Generator().manual_seed(0)
    sequence = torch.tensor([encode(peptide) for peptide in peptides])
    charge = torch.tensor(charges)
    _, valid, _ = prosit_fragment_ladder(sequence, charge)
    labels = torch.rand(valid.shape, generator=generator).masked_fill(~valid, -1.0)
    return {
        "sequence_integer": sequence,
        "precursor_charge": charge,
        "collision_energy": torch.full((len(peptides),), 0.3),
        "labels": labels,
    }


def as_features(rows: dict[str, torch.Tensor]) -> list[dict[str, torch.Tensor]]:
    return [{name: values[i] for name, values in rows.items()} for i in range(len(rows["labels"]))]


def random_processor(seed: int = 0) -> MSDeltaIntensityProcessor:
    generator = torch.Generator().manual_seed(seed)
    return MSDeltaIntensityProcessor(torch.rand((6, 30, 174), generator=generator))


def write_split(directory: Path, batch: dict[str, torch.Tensor]) -> None:
    directory.mkdir(parents=True)
    np.save(directory / "sequence_integer.npy", batch["sequence_integer"].numpy().astype(np.int8))
    np.save(directory / "precursor_charge.npy", batch["precursor_charge"].numpy().astype(np.int8))
    np.save(directory / "collision_energy.npy", batch["collision_energy"].numpy())
    np.save(directory / "intensities.npy", batch["labels"].numpy())


class FragmentIonLadderTests(unittest.TestCase):
    def test_matches_pyteomics_and_prosit_validity(self):
        peptide = "PEPTIDEK"
        mz, valid, length = prosit_fragment_ladder(
            torch.tensor([encode(peptide)]), torch.tensor([2])
        )
        self.assertEqual(int(length), len(peptide))
        slots = mz.view(29, 2, 3)
        for number in range(1, len(peptide)):
            for charge in (1, 2):
                y = mass.fast_mass(peptide[-number:], ion_type="y", charge=charge)
                b = mass.fast_mass(peptide[:number], ion_type="b", charge=charge)
                self.assertAlmostEqual(slots[number - 1, 0, charge - 1].item(), y, places=3)
                self.assertAlmostEqual(slots[number - 1, 1, charge - 1].item(), b, places=3)
        # Seven cleavage sites x {y, b} x fragment charges 1-2.
        self.assertEqual(int(valid.sum()), 7 * 2 * 2)
        self.assertTrue((mz[~valid] == 0).all())


class IntensityCollatorTests(unittest.TestCase):
    def setUp(self):
        self.rows = make_rows(["PEPTIDEK", "ACDEFGHIKLMNPQR", "MKWVTFISLLR"], [2, 3, 4])
        self.batch = random_processor()(as_features(self.rows))

    def test_packs_only_possible_ions_with_their_slots(self):
        mz, valid, _ = prosit_fragment_ladder(
            self.rows["sequence_integer"], self.rows["precursor_charge"]
        )
        present = self.batch["attention_mask"].bool()
        self.assertEqual(present.sum(dim=-1).tolist(), valid.sum(dim=-1).tolist())
        for row in range(3):
            slots = self.batch["ion_slots"][row][present[row]]
            self.assertTrue(valid[row, slots].all())
            self.assertTrue(torch.equal(self.batch["mz"][row][present[row]], mz[row, slots]))
        self.assertEqual(self.batch["peptide_length"].tolist(), [8, 15, 11])
        self.assertAlmostEqual(self.batch["log_intensity"][present].max().item(), 1.0, places=6)

    def test_prior_survives_save_and_load(self):
        processor = random_processor()
        with tempfile.TemporaryDirectory() as directory:
            processor.save_pretrained(directory)
            self.assertTrue((Path(directory) / "preprocessor_config.json").is_file())
            loaded = MSDeltaIntensityProcessor.from_pretrained(directory)
        self.assertTrue(np.array_equal(processor.intensity_prior, loaded.intensity_prior))
        batch = loaded(as_features(self.rows))
        for name, values in self.batch.items():
            self.assertTrue(torch.equal(values, batch[name]), name)


class IntensityPredictionModelTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.config = MSDeltaIntensityPredictionConfig(encoder=ENCODER, head_hidden_size=16)
        self.rows = make_rows(["PEPTIDEK", "ACDEFGHIKLMNPQR", "MKWVTFISLLR"], [2, 3, 4])
        self.batch = random_processor()(as_features(self.rows))

    def test_uniform_encoder_input_is_degenerate_but_prior_input_is_not(self):
        encoder = MSDeltaModel(ENCODER).eval()
        inputs = {name: values[:1] for name, values in self.batch.items()}
        count = int(inputs["attention_mask"].sum())
        with torch.no_grad():
            uniform = encoder(
                inputs["mz"], torch.ones_like(inputs["log_intensity"]), inputs["attention_mask"]
            ).last_hidden_state[0, :count]
            primed = encoder(
                inputs["mz"], inputs["log_intensity"], inputs["attention_mask"]
            ).last_hidden_state[0, :count]
        self.assertLess((uniform - uniform[0]).abs().max().item(), 1e-5)
        self.assertGreater((primed - primed[0]).abs().max().item(), 1e-3)

    def test_frozen_encoder_trains_only_the_head(self):
        model = MSDeltaForIntensityPrediction(
            self.config, encoder=MSDeltaModel(ENCODER), freeze_encoder=True
        )
        model.train()
        self.assertFalse(model.msdelta.training)
        output = model(**self.batch)
        self.assertTrue(torch.isfinite(output.loss))
        output.loss.backward()
        self.assertTrue(all(p.grad is None for p in model.msdelta.parameters()))
        self.assertTrue(any(p.grad is not None for p in model.intensity_head.parameters()))

        valid = self.rows["labels"] >= 0
        self.assertTrue((output.intensities[~valid] == -1).all())
        self.assertTrue(torch.allclose(output.intensities.amax(dim=-1), torch.ones(3)))

    def test_finetuning_trains_every_used_encoder_parameter(self):
        for checkpointing in (False, True):
            with self.subTest(gradient_checkpointing=checkpointing):
                model = MSDeltaForIntensityPrediction(self.config, encoder=MSDeltaModel(ENCODER))
                if checkpointing:
                    model.gradient_checkpointing_enable()
                model.train()
                self.assertTrue(model.msdelta.training)
                model(**self.batch).loss.backward()
                mask_token = model.msdelta.embed.mask_token
                self.assertFalse(mask_token.requires_grad)
                # Every other trainable parameter is reached, so DDP can run
                # with find_unused_parameters=False.
                missing = [
                    name
                    for name, p in model.named_parameters()
                    if p.requires_grad and p.grad is None
                ]
                self.assertEqual(missing, [])

    def test_encoder_gets_its_own_learning_rate(self):
        model = MSDeltaForIntensityPrediction(self.config, encoder=MSDeltaModel(ENCODER))
        with tempfile.TemporaryDirectory() as directory:
            trainer = IntensityTrainer(
                model=model,
                args=TrainingArguments(
                    output_dir=directory, learning_rate=1e-3, weight_decay=0.01, report_to=[]
                ),
                ion_pair_budget=1024,
                encoder_learning_rate=1e-5,
            )
            optimizer = trainer.create_optimizer()
        encoder_ids = {id(p) for p in model.msdelta.parameters()}
        seen = set()
        for group in optimizer.param_groups:
            in_encoder = {id(p) in encoder_ids for p in group["params"]}
            self.assertEqual(len(in_encoder), 1, "groups must not mix encoder and head")
            self.assertEqual(group["lr"], 1e-5 if in_encoder.pop() else 1e-3)
            seen.update(id(p) for p in group["params"])
        trainable = {id(p) for p in model.parameters() if p.requires_grad}
        self.assertEqual(seen, trainable)

    def test_encoder_free_control_ignores_the_encoder(self):
        config = MSDeltaIntensityPredictionConfig(
            encoder=ENCODER, head_hidden_size=16, use_encoder_states=False
        )
        model = MSDeltaForIntensityPrediction(config, encoder=MSDeltaModel(ENCODER)).eval()
        before = model(**self.batch).intensities
        with torch.no_grad():
            for parameter in model.msdelta.parameters():
                parameter.add_(1.0)
        self.assertTrue(torch.equal(before, model(**self.batch).intensities))


class PrositDataTests(unittest.TestCase):
    def test_dataset_preserves_order_and_prior_is_finite(self):
        batch = make_rows(["PEPTIDEK", "ACDEFGHIKLMNPQR", "MKWVTFISLLR"], [2, 3, 2])
        with tempfile.TemporaryDirectory() as directory:
            write_split(Path(directory) / "train", batch)
            dataset = PrositIntensityDataset(Path(directory) / "train")
            items = dataset.__getitems__([2, 0, 1])
            for item, row in zip(items, (2, 0, 1)):
                self.assertTrue(torch.equal(item["labels"], batch["labels"][row]))
                self.assertEqual(int(item["precursor_charge"]), int(batch["precursor_charge"][row]))
            prior = fit_intensity_prior(dataset, MSDeltaIntensityPredictionConfig())
        self.assertTrue(torch.isfinite(prior).all())
        # Unseen (charge, length) cells fall back to the charge-level mean. At
        # charge 2, ion numbers 8-10 exist only in the 11-mer, so there the
        # fallback must equal the 11-mer's own cell.
        only_11mer = slice(6 * 7, 6 * 10)
        self.assertTrue(torch.allclose(prior[1, 29, only_11mer], prior[1, 10, only_11mer]))
        self.assertGreater(prior[1, 7].sum().item(), 0)


if __name__ == "__main__":
    unittest.main()
