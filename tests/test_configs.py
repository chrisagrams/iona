"""Configuration and orchestration. Nothing here needs a GPU, and all of it has bitten.

These are the cheapest tests in the suite and they guard the most expensive failures: a
config mistake is not discovered until a job has been queued, scheduled, and has run far
enough to reach the broken part.
"""

from __future__ import annotations

import os
import re
import subprocess
import tempfile
from pathlib import Path

import pytest

from tests.conftest import REPO, args_files

PBS_SCRIPTS = sorted((REPO / "pbs").glob("*.pbs"))
CONFIG_DIRS = sorted(p.parent for p in args_files())


@pytest.mark.parametrize("path", args_files(), ids=lambda p: p.parent.name)
class TestArgsFiles:
    def test_no_comments(self, path):
        """HfArgumentParser reads args files with read_text().split().

        A '# ...' line does not become a comment, it becomes several stray positional
        arguments.
        """
        bad = [l for l in path.read_text().splitlines() if l.strip().startswith("#")]
        assert not bad, f"{len(bad)} comment lines become positional args"

    def test_every_line_is_a_flag_and_one_value(self, path):
        """Same cause as above: a value containing a space is silently split."""
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            parts = line.split()
            assert parts[0].startswith("--"), f"not a flag: {line!r}"
            assert len(parts) == 2, f"{len(parts) - 1} values on {line!r}"

    def test_no_duplicate_flags(self, path):
        """A repeated flag silently keeps the last value and discards the first."""
        flags = [l.split()[0] for l in path.read_text().splitlines() if l.strip()]
        assert len(flags) == len(set(flags)), \
            f"duplicated: {sorted({f for f in flags if flags.count(f) > 1})}"

    def test_has_a_description(self, path):
        """Auto-derived text says what the settings are; this says why the run exists."""
        description = path.parent / "DESCRIPTION.md"
        assert description.exists(), f"no DESCRIPTION.md beside {path.parent.name}"
        assert len(description.read_text().split()) >= 10, "too short to be a description"


def _flags(path):
    tokens = path.read_text().split()
    return dict(zip(tokens[::2], tokens[1::2]))


FINETUNE = [p for p in args_files() if p.parent.name.startswith("finetune-")]


@pytest.mark.parametrize("path", FINETUNE, ids=lambda p: p.parent.name)
class TestFinetuneConfigs:
    def test_wandb_destination(self, path):
        """Denoise and alignment metrics are not comparable and must not share a project."""
        flags = _flags(path)
        assert flags.get("--wandb_entity") == "CS_Pharm"
        # One project per objective, because the metrics are not comparable: per-peak
        # AUROC, cross-modal hit@1 and an embedding separation ratio answer different
        # questions and sharing a project invites exactly the comparison that should
        # not be made.
        # align is checked FIRST: finetune-align-contrastive is an ALIGNMENT run whose
        # metric is cross-modal hit@1 -- it merely uses a contrastively-trained teacher --
        # so it belongs with the other alignment runs it must be compared against, not
        # with the encoder runs measured by separation ratio.
        name = path.parent.name
        expected = ("msdelta-finetune-align" if "align" in name
                    else "msdelta-finetune-contrastive" if "contrastive" in name
                    else "msdelta-finetune-denoise")
        assert flags.get("--wandb_project") == expected

    def test_peak_pair_budget_can_hold_one_spectrum(self, path):
        """Below max_peaks**2 the budget sampler cannot place even a single spectrum."""
        flags = _flags(path)
        if "--peak_pair_budget" not in flags:
            pytest.skip("no peak budget")
        peaks = int(flags.get("--max_peaks", 512))
        assert int(flags["--peak_pair_budget"]) >= peaks ** 2

    def test_multi_tile_configs_name_deepspeed(self, path):
        """Twelve-way DDP takes a GPU page fault (FT7); ZeRO-2 is the only multi-tile path."""
        if not path.parent.name.endswith("-ds"):
            pytest.skip("single-tile config")
        assert "--deepspeed" in _flags(path)


@pytest.mark.parametrize("script", PBS_SCRIPTS, ids=lambda p: p.name)
class TestPBSScripts:
    def test_valid_bash(self, script):
        subprocess.run(["bash", "-n", str(script)], check=True, capture_output=True)

    def test_referenced_paths_exist(self, script):
        """Catches a rename that updated the file but not the script that launches it.

        Comment lines are skipped: the usage headers name example paths like
        sweeps/subset.txt that are meant to be supplied by the caller, not to exist. A
        trailing '/.' is stripped because `cp -r dir/. dest/` is a real reference to dir.
        """
        body = "\n".join(line for line in script.read_text().splitlines()
                          if not line.lstrip().startswith("#"))
        refs = {a or b for a, b in re.findall(
            r"\$REPO_DIR/([A-Za-z0-9_./-]+)|(?<![\w/])((?:pbs|configs|sweeps)/[A-Za-z0-9_./-]+)",
            body)}
        # A reference ending in a separator is the literal prefix of a name the script
        # builds at runtime -- "pbs/logs/bisect-$variant-$JOB.log" matches as
        # "pbs/logs/bisect-" -- so it names no file and cannot be checked.
        candidates = (x.rstrip("/.") for x in refs if x)
        missing = sorted(r for r in candidates
                         if not r.endswith(("-", "_")) and not (REPO / r).exists())
        assert not missing, f"missing: {missing}"

    def test_no_secrets_passed_through_qsub(self, script):
        """`qsub -v` variable lists are world-readable via `qstat -f`."""
        for line in script.read_text().splitlines():
            if "qsub" in line and "-v" in line:
                assert not re.search(r"(TOKEN|KEY|SECRET|PASSWORD)=", line)


class TestSweepGrid:
    def test_arms_match_their_template(self):
        """A stale grid would have run 72 arms at 8x the memory that fits on a tile."""
        out = subprocess.run(
            [str(REPO / ".venv/bin/python"), "sweeps/make_denoise_grid.py", "--check"],
            cwd=REPO, capture_output=True, text=True)
        assert out.returncode == 0, out.stdout + out.stderr

    def test_stamp_records_template_and_stage(self):
        """Without the stage, --check validates the wrong arm set and refuses the job."""
        stamp = (REPO / "configs/sweep-denoise/.template").read_text().split()
        assert len(stamp) == 2, "stamp must hold the template path and the stage"
        assert (REPO / stamp[0]).exists()

    def test_every_arm_has_a_description(self):
        arms = [d for d in (REPO / "configs/sweep-denoise").iterdir() if d.is_dir()]
        assert arms
        missing = [d.name for d in arms if not (d / "DESCRIPTION.md").exists()]
        assert not missing, f"{len(missing)} arms undescribed"

    def test_arm_names_match_their_contents(self):
        """A mislabelled arm makes the whole grid's results untrustworthy."""
        for directory in sorted((REPO / "configs/sweep-denoise").iterdir()):
            if not directory.is_dir():
                continue
            flags = _flags(directory / "training.args")
            expected = (f"lr{flags['--learning_rate'].replace('-', '')}"
                        f"_es{flags['--encoder_lr_scale'].replace('.', '')}"
                        f"_ep{flags['--num_train_epochs']}"
                        f"_h{flags['--head_hidden_size']}")
            batch = (int(flags["--per_device_train_batch_size"]) * 12
                     * int(flags["--gradient_accumulation_steps"]))
            if batch != 48:
                expected += f"_b{batch}"
            assert directory.name == expected

    def test_run_names_carry_the_version_prefix(self):
        """Runs from before the DDP and dtype fixes must not collide with these."""
        for directory in sorted((REPO / "configs/sweep-denoise").iterdir()):
            if directory.is_dir():
                assert _flags(directory / "training.args")["--run_name"].startswith("v2_")


class TestSweepPartition:
    def _dry_run(self, nodes, env=None):
        with tempfile.NamedTemporaryFile("w", suffix=".nodes", delete=False) as handle:
            handle.write("".join(f"fake-node-{i}\n" for i in range(nodes)))
            nodefile = handle.name
        # Inherit the real environment: the script uses USER to build the scratch path,
        # and a hand-built minimal env silently changes what is being tested.
        environment = dict(os.environ)
        environment.update({"DRY_RUN": "1", "PBS_NODEFILE": nodefile,
                            "PBS_JOBID": "dry"})
        environment.update(env or {})
        try:
            return subprocess.run(["bash", "pbs/aurora-finetune-sweep.pbs"], cwd=REPO,
                                  capture_output=True, text=True, env=environment)
        finally:
            Path(nodefile).unlink(missing_ok=True)

    def test_every_arm_is_assigned_exactly_once(self):
        """A dropped or duplicated arm is a silent hole in the grid."""
        out = self._dry_run(6, {"TILES_PER_ARM": "12"})
        assert out.returncode == 0, out.stdout + out.stderr
        assigned = []
        for line in out.stdout.splitlines():
            if line.startswith("  slot"):
                assigned += line.split(":", 1)[1].split()
        expected = sorted(d.name for d in (REPO / "configs/sweep-denoise").iterdir()
                          if d.is_dir())
        assert sorted(assigned) == expected
        assert len(assigned) == len(set(assigned))

    def test_slots_follow_tiles_per_arm(self):
        out = self._dry_run(2, {"TILES_PER_ARM": "1", "SKIP_GRID_CHECK": "1"})
        assert "2 hosts x 12 slots = 24 concurrent" in out.stdout

    def test_multi_tile_without_deepspeed_is_refused(self):
        """That combination is the DDP reducer, which faults (FT7)."""
        arm = REPO / "configs/sweep-denoise/nods_probe"
        arm.mkdir(exist_ok=True)
        source = next(REPO.glob("configs/sweep-denoise/lr*/training.args"))
        (arm / "training.args").write_text(
            "\n".join(l for l in source.read_text().splitlines()
                      if not l.startswith("--deepspeed")) + "\n")
        try:
            out = self._dry_run(1, {"TILES_PER_ARM": "12", "ARMS": "nods_probe",
                                    "SKIP_GRID_CHECK": "1"})
            assert out.returncode != 0
            assert "no --deepspeed" in out.stdout + out.stderr
        finally:
            (arm / "training.args").unlink(missing_ok=True)
            arm.rmdir()
