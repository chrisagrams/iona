"""Check the Prosit ion ladder, the intensity head, and its data utilities."""

import json
import os
import socket
import tempfile
import unittest
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path
from unittest import mock

import numpy as np
import torch
from accelerate.state import AcceleratorState
from pyteomics import mass
from safetensors.torch import load_file, save_file
from transformers import HfArgumentParser, TrainingArguments

from iona.configuration_iona import IonaConfig, IonaIntensityPredictionConfig
from iona.denoising import PeakBudgetBatchSampler
from iona.finetune_intensity import (
    IntensityDataArguments,
    IntensityModelArguments,
    IntensityTrainingArguments,
    load_encoder,
    main,
)
from iona.intensity import (
    IntensityTrainer,
    IonaIntensityProcessor,
    PrositIntensityDataset,
    fit_intensity_prior,
    prosit_fragment_ladder,
)
from iona.modeling_iona import IonaForIntensityPrediction, IonaForPreTraining, IonaModel

ALPHABET = "ACDEFGHIKLMNPQRSTVWY"
ENCODER = IonaConfig(
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


def random_processor(seed: int = 0) -> IonaIntensityProcessor:
    generator = torch.Generator().manual_seed(seed)
    return IonaIntensityProcessor(torch.rand((6, 30, 174), generator=generator))


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
            loaded = IonaIntensityProcessor.from_pretrained(directory)
        self.assertTrue(np.array_equal(processor.intensity_prior, loaded.intensity_prior))
        batch = loaded(as_features(self.rows))
        for name, values in self.batch.items():
            self.assertTrue(torch.equal(values, batch[name]), name)


class IntensityPredictionModelTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.config = IonaIntensityPredictionConfig(encoder=ENCODER, head_hidden_size=16)
        self.rows = make_rows(["PEPTIDEK", "ACDEFGHIKLMNPQR", "MKWVTFISLLR"], [2, 3, 4])
        self.batch = random_processor()(as_features(self.rows))

    def test_uniform_encoder_input_is_degenerate_but_prior_input_is_not(self):
        encoder = IonaModel(ENCODER).eval()
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
        model = IonaForIntensityPrediction(
            self.config, encoder=IonaModel(ENCODER), freeze_encoder=True
        )
        model.train()
        self.assertFalse(model.iona.training)
        output = model(**self.batch)
        self.assertTrue(torch.isfinite(output.loss))
        output.loss.backward()
        self.assertTrue(all(p.grad is None for p in model.iona.parameters()))
        self.assertTrue(any(p.grad is not None for p in model.intensity_head.parameters()))

        valid = self.rows["labels"] >= 0
        self.assertTrue((output.intensities[~valid] == -1).all())
        self.assertTrue(torch.allclose(output.intensities.amax(dim=-1), torch.ones(3)))

    def test_finetuning_trains_every_used_encoder_parameter(self):
        for checkpointing in (False, True):
            with self.subTest(gradient_checkpointing=checkpointing):
                model = IonaForIntensityPrediction(self.config, encoder=IonaModel(ENCODER))
                if checkpointing:
                    model.gradient_checkpointing_enable()
                model.train()
                self.assertTrue(model.iona.training)
                model(**self.batch).loss.backward()
                mask_token = model.iona.embed.mask_token
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
        model = IonaForIntensityPrediction(self.config, encoder=IonaModel(ENCODER))
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
        encoder_ids = {id(p) for p in model.iona.parameters()}
        seen = set()
        for group in optimizer.param_groups:
            in_encoder = {id(p) in encoder_ids for p in group["params"]}
            self.assertEqual(len(in_encoder), 1, "groups must not mix encoder and head")
            self.assertEqual(group["lr"], 1e-5 if in_encoder.pop() else 1e-3)
            seen.update(id(p) for p in group["params"])
        trainable = {id(p) for p in model.parameters() if p.requires_grad}
        self.assertEqual(seen, trainable)

    def test_encoder_free_control_ignores_the_encoder(self):
        config = IonaIntensityPredictionConfig(
            encoder=ENCODER, head_hidden_size=16, use_encoder_states=False
        )
        model = IonaForIntensityPrediction(config, encoder=IonaModel(ENCODER)).eval()
        before = model(**self.batch).intensities
        with torch.no_grad():
            for parameter in model.iona.parameters():
                parameter.add_(1.0)
        self.assertTrue(torch.equal(before, model(**self.batch).intensities))


class LoadEncoderTests(unittest.TestCase):
    def test_msdelta_era_checkpoint_keys_load_as_pretrained(self):
        torch.manual_seed(0)
        model = IonaForPreTraining(ENCODER)
        with tempfile.TemporaryDirectory() as directory:
            model.save_pretrained(directory)
            weights = Path(directory) / "model.safetensors"
            state = load_file(weights)
            renamed = {
                ("msdelta." + key[len("iona.") :] if key.startswith("iona.") else key): value
                for key, value in state.items()
            }
            self.assertTrue(any(key.startswith("msdelta.") for key in renamed))
            save_file(renamed, weights, metadata={"format": "pt"})
            encoder = load_encoder(IntensityModelArguments(pretrained_path=directory))
        for name, value in model.iona.state_dict().items():
            torch.testing.assert_close(encoder.state_dict()[name], value, msg=name)


class PeakBudgetBatchSamplerTests(unittest.TestCase):
    def test_peak_budget_caps_batches_of_short_spectra(self):
        lengths = [4] * 100 + [30] * 10
        pairs_only = PeakBudgetBatchSampler(lengths, peak_pair_budget=4000, seed=0)
        capped = PeakBudgetBatchSampler(lengths, peak_pair_budget=4000, seed=0, peak_budget=64)

        def padded_peaks(batch):
            return len(batch) * max(lengths[i] for i in batch)

        self.assertGreater(max(map(padded_peaks, pairs_only)), 64)
        self.assertLessEqual(max(map(padded_peaks, capped)), 64)
        for sampler in (pairs_only, capped):
            self.assertEqual(sorted(i for batch in sampler for i in batch), list(range(110)))


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
            prior = fit_intensity_prior(dataset, IonaIntensityPredictionConfig())
        self.assertTrue(torch.isfinite(prior).all())
        # Unseen (charge, length) cells fall back to the charge-level mean. At
        # charge 2, ion numbers 8-10 exist only in the 11-mer, so there the
        # fallback must equal the 11-mer's own cell.
        only_11mer = slice(6 * 7, 6 * 10)
        self.assertTrue(torch.allclose(prior[1, 29, only_11mer], prior[1, 10, only_11mer]))
        self.assertGreater(prior[1, 7].sum().item(), 0)


REPO = Path(__file__).resolve().parents[1]
CONFIGS = REPO / "configs" / "finetune"
# What scripts/aurora-finetune.pbs appends after --args_file (MICRO_BATCH_SIZE=16 on 24 tiles).
LAUNCHER_ARGS = [
    "--run_name",
    "intensity-prosit-200m-random-123",
    "--output_dir",
    "/tmp/intensity-run",
    "--per_device_train_batch_size",
    "16",
    "--per_device_eval_batch_size",
    "16",
    "--gradient_accumulation_steps",
    "1",
    "--ddp_backend",
    "xccl",
    "--wandb_project",
    "msdelta-intensity",
]


class FinetuneConfigTests(unittest.TestCase):
    def parse(self, name: str, *extra: str):
        parser = HfArgumentParser(
            (IntensityModelArguments, IntensityDataArguments, IntensityTrainingArguments)
        )
        args = ["--args_file", str(CONFIGS / name / "training.args"), *LAUNCHER_ARGS, *extra]
        with single_process_group():
            return parser.parse_args_into_dataclasses(
                args=[*args, "--ddp_backend", "gloo"], args_file_flag="--args_file"
            )

    def test_random_config_differs_from_pretrained_only_in_init(self):
        pretrained = self.parse("intensity-prosit-200m")
        random = self.parse("intensity-prosit-200m-random")
        self.assertFalse(pretrained[0].random_init)
        self.assertTrue(random[0].random_init)
        self.assertEqual(
            asdict_without(pretrained[0], "random_init"), asdict_without(random[0], "random_init")
        )
        self.assertEqual(pretrained[1], random[1])
        model_args, data_args, training_args = random
        # The best 200m pretrained run (job 8870454): fine-tuned encoder at 3e-5, 3 epochs.
        self.assertFalse(model_args.freeze_encoder)
        self.assertEqual(training_args.encoder_learning_rate, 3e-5)
        self.assertEqual(training_args.learning_rate, 1e-3)
        self.assertEqual(training_args.num_train_epochs, 3)
        self.assertEqual((data_args.ion_pair_budget, data_args.ion_budget), (174762, 5461))
        self.assertEqual(training_args.wandb_project, "msdelta-intensity")
        self.assertEqual(training_args.run_name, "intensity-prosit-200m-random-123")


def asdict_without(value, field: str) -> dict:
    return {k: v for k, v in asdict(value).items() if k != field}


@contextmanager
def single_process_group():
    """A one-rank gloo group: with a ddp_backend set, TrainingArguments needs one, as under mpiexec."""
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    env = {
        "MASTER_ADDR": "127.0.0.1",
        "MASTER_PORT": str(port),
        "RANK": "0",
        "WORLD_SIZE": "1",
        "LOCAL_RANK": "0",
        "LOCAL_WORLD_SIZE": "1",
    }
    with mock.patch.dict(os.environ, env):
        torch.distributed.init_process_group("gloo", rank=0, world_size=1)
        try:
            yield
        finally:
            torch.distributed.destroy_process_group()
            AcceleratorState._reset_state(reset_partial_state=True)


class FinetuneSmokeTests(unittest.TestCase):
    def run_random_config(self, *extra: str) -> dict:
        """Train the random-init config for 2 steps as the launcher would; return holdout metrics."""
        peptides = ["PEPTIDEK", "ACDEFGHIKLMNPQR", "MKWVTFISLLR", "LLGNVLVCVLAR"] * 4
        charges = [2, 3, 2, 3] * 4
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for split in ("train", "val", "holdout"):
                write_split(root / "data" / split, make_rows(peptides, charges))
            np.save(
                root / "data" / "holdout" / "prosit_spectral_angle.npy",
                np.full(len(peptides), 0.9, dtype=np.float32),
            )
            IonaForPreTraining(ENCODER).save_pretrained(root / "encoder")
            argv = [
                "--args_file",
                str(CONFIGS / "intensity-prosit-200m-random" / "training.args"),
                *LAUNCHER_ARGS,
                "--output_dir",
                str(root / "run"),
                "--pretrained_path",
                str(root / "encoder"),
                "--data_dir",
                str(root / "data"),
                "--ddp_backend",
                "gloo",
                "--use_cpu",
                "true",
                "--bf16",
                "false",
                "--max_steps",
                "2",
                "--eval_steps",
                "1",
                "--save_steps",
                "1",
                "--logging_steps",
                "1",
                "--dataloader_num_workers",
                "0",
                "--prior_samples",
                "16",
                "--eval_samples",
                "8",
                "--ion_pair_budget",
                "65536",
                "--ion_budget",
                "4096",
                *extra,
            ]
            with single_process_group():
                self.assertEqual(main(argv), 0)
            metrics = json.loads((root / "run" / "holdout_metrics.json").read_text())
        self.assertEqual(metrics["holdout/n"], len(peptides))
        return metrics

    def test_launcher_style_run_writes_holdout_metrics(self):
        metrics = self.run_random_config("--wandb_project", "", "--report_to", "none")
        self.assertEqual(metrics["encoder_init"], "random")
        self.assertGreaterEqual(metrics["holdout/spectral_angle_median"], 0.0)

    def test_rank_without_a_wandb_run_still_trains(self):
        # init_wandb_run returns None on all but each node's local rank 0 (job 9095142).
        with (
            mock.patch.dict(os.environ, {"WANDB_MODE": "disabled"}),
            mock.patch("iona.finetune_intensity.init_wandb_run", return_value=None) as init,
        ):
            metrics = self.run_random_config()
        init.assert_called_once()
        self.assertEqual(metrics["encoder_init"], "random")


if __name__ == "__main__":
    unittest.main()
