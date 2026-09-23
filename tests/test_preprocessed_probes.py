"""Check finalized probe preprocessing and offline loading without downloads."""

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from datasets import Dataset, DatasetDict

from iona.posttraining import build_probe_data
from iona.preprocess import main


class PreprocessedProbeTests(unittest.TestCase):
    def test_probes_only_saves_and_loads_all_probe_datasets(self):
        splits = DatasetDict(
            {name: Dataset.from_dict({"value": [1, 2]}) for name in ("train", "validation")}
        )
        evaluation = {"retrieval": splits["validation"], "replicate_retrieval": splits["train"]}
        processor = object()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "iona-probes"
            argv = [
                "--processor_name_or_path",
                "unused",
                "--dataset_repo_id",
                "unused",
                "--preprocessed_probe_dir",
                str(root),
                "--preprocessing_num_workers",
                "0",
                "--denoise_steps",
                "1",
                "--retrieval_steps",
                "1",
                "--probes_only",
                "true",
            ]
            with (
                patch("iona.preprocess.IonaProcessor.from_pretrained", return_value=processor),
                patch("iona.preprocess.build_pretraining_datasets") as pretrain,
                patch("iona.preprocess.build_probe_data") as build,
            ):
                build.side_effect = [(splits, processor, None), (splits, processor, evaluation)]
                self.assertEqual(main(argv), 0)
                pretrain.assert_not_called()
                self.assertEqual(build.call_count, 2)
                self.assertIsNone(build.call_args.args[1].preprocessed_probe_dir)

            data_args = SimpleNamespace(
                preprocessed_probe_dir=str(root),
                preprocessing_num_workers=0,
                processor_name_or_path="unused",
            )
            args = SimpleNamespace(denoise_max_peaks=1024)
            with (
                patch("iona.posttraining.IonaProcessor.from_pretrained", return_value=processor),
                patch("iona.posttraining.build_denoising_datasets") as denoise,
                patch("iona.posttraining.build_retrieval_datasets") as retrieval,
                patch("iona.posttraining.build_retrieval_evaluation_datasets") as evaluate,
            ):
                loaded, _, galleries = build_probe_data("denoise", data_args, args, processor)
                self.assertEqual(loaded["train"]["value"], [1, 2])
                self.assertIsNone(galleries)
                loaded, _, galleries = build_probe_data("retrieval", data_args, args, processor)
                self.assertEqual(loaded["validation"]["value"], [1, 2])
                self.assertEqual(set(galleries), set(evaluation))
                denoise.assert_not_called()
                retrieval.assert_not_called()
                evaluate.assert_not_called()

    def test_off_does_not_build_probes(self):
        with (
            patch("iona.preprocess.IonaProcessor.from_pretrained"),
            patch("iona.preprocess.build_probe_data") as build,
        ):
            self.assertEqual(
                main(
                    [
                        "--processor_name_or_path",
                        "unused",
                        "--dataset_repo_id",
                        "unused",
                        "--preprocessed_probe_dir",
                        "unused",
                        "--denoise_steps",
                        "1",
                        "--probe_execution",
                        "off",
                        "--probes_only",
                        "true",
                    ]
                ),
                0,
            )
            build.assert_not_called()


if __name__ == "__main__":
    unittest.main()
