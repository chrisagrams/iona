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
    def test_sets_level_zero_environment_if_it_imports_torch(self, script):
        """An Aurora script whose module imports torch must set the ZE variables first.

        Importing torch initialises XPU even for work that never uses a GPU, and on a
        compute node without ZE_FLAT_DEVICE_HIERARCHY and an affinity mask that
        enumeration segfaults before Python can report anything: jobs 8840898 and
        8840907 died with a bare "Segmentation fault", no traceback, even under
        -X faulthandler, while every component ran fine on a login node. The launcher
        simply omitted three exports every other Aurora script already had.

        Whether a module pulls in torch is checked by importing it, not guessed from
        the filename: msdelta.preprocess does not, so aurora-preprocess.pbs is exempt
        and stays exempt only for as long as that remains true.
        """
        text = script.read_text()
        if script.name.startswith("polaris"):
            pytest.skip("CUDA, not Level Zero")
        modules = re.findall(r"-m\s+\"?(msdelta\.[a-z_.]+)", text)
        modules += re.findall(r'-m "\$([A-Z_]+)"', text) and ["msdelta.finetune_denoise"]
        if not modules:
            pytest.skip("launches no msdelta module")
        probe = subprocess.run(
            [str(REPO / ".venv/bin/python"), "-c",
             f"import {modules[0]}, sys; print('torch' in sys.modules)"],
            cwd=REPO, capture_output=True, text=True,
            env={**os.environ, "PYTHONPATH": str(REPO), "HF_HUB_OFFLINE": "1"})
        if "True" not in probe.stdout:
            pytest.skip(f"{modules[0]} does not import torch")
        # An export line may set several variables at once
        # (`export ONEAPI_DEVICE_SELECTOR=... ZE_FLAT_DEVICE_HIERARCHY=FLAT ZE_AFFINITY_MASK=0`);
        # checking for the literal "export VARIABLE" flagged four correct scripts for weeks.
        exported = set()
        for line in text.splitlines():
            line = line.split("#", 1)[0].strip()
            if line.startswith("export "):
                exported.update(re.findall(r"\b([A-Z_][A-Z0-9_]*)=", line[len("export "):]))
        for variable in ("ONEAPI_DEVICE_SELECTOR", "ZE_FLAT_DEVICE_HIERARCHY",
                         "ZE_AFFINITY_MASK"):
            assert variable in exported, f"{script.name} never exports {variable}"

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
        # ONE_TILE_OK because sweep-denoise is a DeepSpeed grid and one tile per arm is
        # now refused for it -- that changes the effective batch, not just the speed.
        # This test is about the slot ARITHMETIC, which is what the override is for.
        out = self._dry_run(2, {"TILES_PER_ARM": "1", "SKIP_GRID_CHECK": "1",
                                "ONE_TILE_OK": "1"})
        assert "2 hosts x 12 slots = 24 concurrent" in out.stdout

    def test_one_tile_with_deepspeed_is_refused(self):
        """The inverse of the FT7 guard, and the mistake that cost job 8846027.

        Omitting TILES_PER_ARM looks like omitting nothing, but it defaults to 1, and a
        DeepSpeed arm on one tile runs at effective batch 1 where every other denoise
        run uses 12 -- a different experiment, silently.
        """
        out = self._dry_run(2, {"SKIP_GRID_CHECK": "1"})
        assert out.returncode != 0
        assert "EFFECTIVE BATCH" in out.stdout + out.stderr

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


class TestEvalCheckpointArgs:
    """`expand_args_file` decides what an eval-only run inherits from a training config.

    Getting this wrong is silent, not loud: inheriting --deepspeed hangs a single-tile
    job waiting on a launcher that is not there, and inheriting --report_to publishes an
    evaluation into the sweep project as if it were a thirteenth arm.
    """

    def _expand(self, tmp_path, text, extra=()):
        from msdelta.eval_checkpoint import expand_args_file
        path = tmp_path / "training.args"
        path.write_text(text)
        return expand_args_file(["--args_file", str(path), *extra])

    def test_strips_the_flags_an_eval_must_not_inherit(self, tmp_path):
        tokens = self._expand(tmp_path, (
            "--learning_rate 2e-4\n--deepspeed configs/deepspeed-zero2.json\n"
            "--report_to wandb\n--wandb_project msdelta-finetune-denoise\n"
            "--load_best_model_at_end true\n--output_dir ./runs/original\n"
        ))
        for flag in ("--deepspeed", "--report_to", "--wandb_project",
                     "--load_best_model_at_end", "--output_dir"):
            assert flag not in tokens, f"{flag} leaked into an eval-only run"
        assert tokens == ["--learning_rate", "2e-4"]

    def test_keeps_everything_that_defines_the_model(self, tmp_path):
        """Strip too much and the checkpoint stops matching the model it is loaded into."""
        tokens = self._expand(tmp_path, (
            "--pretrained_path /flare/model\n--head_hidden_size 128\n"
            "--head_dropout 0.1\n--max_peaks 512\n--encoder_lr_scale 0.5\n"
        ))
        for flag in ("--pretrained_path", "--head_hidden_size", "--head_dropout",
                     "--max_peaks"):
            assert flag in tokens

    def test_caller_flags_survive_alongside_the_file(self, tmp_path):
        tokens = self._expand(tmp_path, "--max_peaks 512\n",
                              extra=["--output_dir", "/scratch/x", "--checkpoint", "/c"])
        assert tokens == ["--max_peaks", "512", "--output_dir", "/scratch/x",
                          "--checkpoint", "/c"]

    def test_rejects_an_unpaired_file(self, tmp_path):
        """Pairwise stripping is only safe on strict pairs, so refuse anything else."""
        with pytest.raises(SystemExit):
            self._expand(tmp_path, "--max_peaks 512 --bf16\n")

    def test_every_generated_arm_is_strict_pairs(self):
        """The property the stripper depends on, checked against the real grids."""
        roots = sorted((REPO / "configs").glob("sweep-*"))
        assert roots, "no generated grids to check"
        for args in sorted(p for root in roots for p in root.glob("*/training.args")):
            tokens = args.read_text().split()
            assert len(tokens) % 2 == 0, f"{args} has an odd token count"
            odd = [t for t in tokens[1::2] if t.startswith("--")]
            assert not odd, f"{args} has bare boolean flags: {odd[:3]}"


def _known_flags():
    """Every flag any entry point can accept, from the dataclasses themselves.

    Built by reflection rather than hardcoded, so it tracks the installed transformers
    instead of what its documentation says. That distinction is the point: transformers
    5.17 on this machine has NO `warmup_ratio` field, only `warmup_steps`, and 27 configs
    were rewritten to use it and submitted before two jobs died with "Some specified
    arguments are not used by the HfArgumentParser".
    """
    from transformers import TrainingArguments
    names = {f"--{n}" for n in TrainingArguments.__dataclass_fields__}
    for module, classes in (
        ("msdelta.finetune_denoise",
         ("DenoiseModelArguments", "DenoiseDataArguments", "DenoiseFinetuneArguments")),
        ("msdelta.finetune_contrastive",
         ("ContrastiveModelArguments", "ContrastiveDataArguments",
          "ContrastiveTrainingArguments")),
        ("msdelta.finetune_align",
         ("AlignModelArguments", "AlignDataArguments", "AlignTrainingArguments")),
        # Pretraining. configs/msdelta-base-* are for msdelta/train.py, not a
        # fine-tune, and leaving these out made the test fail on five perfectly
        # valid configs -- a test that cries wolf gets deleted, so it has to know
        # about every entry point whose configs live under configs/.
        ("msdelta.training_args",
         ("ModelArguments", "DataArguments", "MSDeltaTrainingArguments")),
    ):
        try:
            mod = __import__(module, fromlist=classes)
        except Exception:
            continue
        for cls in classes:
            obj = getattr(mod, cls, None)
            if obj is not None and hasattr(obj, "__dataclass_fields__"):
                names |= {f"--{n}" for n in obj.__dataclass_fields__}
    return names


@pytest.mark.parametrize("path", args_files(), ids=lambda p: p.parent.name)
def test_every_flag_is_one_the_parser_accepts(path):
    """(regression) A flag no dataclass declares is a hard failure at job start.

    The other args-file tests check SHAPE -- pairs, no comments, no duplicates -- and a
    well-formed flag that simply does not exist passes all of them. It then costs a
    queue slot to discover, which is exactly what jobs 8842153 and 8842154 did.
    """
    known = _known_flags()
    flags = {l.split()[0] for l in path.read_text().splitlines() if l.strip()}
    unknown = sorted(flags - known)
    assert not unknown, (
        f"{path.parent.name}: no dataclass declares {unknown}. "
        f"Check the installed transformers rather than its docs."
    )


def test_generated_sweep_arms_use_no_unknown_flags():
    """(regression) Arms can carry flags their template never had.

    args_files() deliberately excludes generated arms -- parsing 216 of them would only
    re-test the generator. But a generator adds overrides, and an override that no
    dataclass declares appears in NO template, so the per-config test above cannot see
    it: `--random_init true` in the contrastive control grid is written by
    make_contrastive_grid.py --random and exists nowhere else.

    Checking the UNION of flags over every arm costs one pass and closes that gap.
    """
    known = _known_flags()
    seen: dict[str, str] = {}
    for grid in sorted((REPO / "configs").glob("sweep-*")):
        for arm in sorted(grid.glob("*/training.args")):
            for line in arm.read_text().splitlines():
                if line.strip():
                    seen.setdefault(line.split()[0], f"{grid.name}/{arm.parent.name}")
    unknown = {f: where for f, where in seen.items() if f not in known}
    assert not unknown, f"no dataclass declares these generated flags: {unknown}"


class TestResumeIsWiredThrough:
    """RESUME_JOB in the sweep runner is useless unless train() is told about it.

    Trainer.train() signs as `resume_from_checkpoint: str | bool | None = None` and
    never falls back to `self.args.resume_from_checkpoint`. So `--resume_from_checkpoint
    <path>` parses cleanly into TrainingArguments and is then silently ignored: the run
    restarts from step 0 while every log line and the PBS output claim it resumed. That
    failure is invisible except in the step count, which is exactly the kind of thing
    nobody checks on a recovery run.
    """

    ENTRY_POINTS = ("msdelta.finetune_denoise", "msdelta.finetune_contrastive")

    @pytest.mark.parametrize("module", ENTRY_POINTS)
    def test_train_is_not_called_bare(self, module):
        import importlib
        source = Path(importlib.import_module(module).__file__).read_text()   # follows the shims
        assert "trainer.train()" not in source, (
            f"{module} calls trainer.train() with no argument, so "
            f"--resume_from_checkpoint is parsed and discarded")
        assert "trainer.train(resume_from_checkpoint=" in source, module

    def test_the_sweep_runner_can_actually_pass_one(self):
        """And the other half: the runner has to build the flag in the first place."""
        script = (REPO / "pbs" / "aurora-finetune-sweep.pbs").read_text()
        assert "RESUME_JOB" in script
        assert "--resume_from_checkpoint" in script
        # Checkpoints must be ordered numerically: lexically, checkpoint-9000 beats
        # checkpoint-29072 and a recovery run would silently rewind 20k steps.
        assert re.search(r"sort -r?n\b", script), "checkpoint selection must sort numerically"
        # And it must skip a half-written newest checkpoint: a job killed mid-save leaves
        # one without trainer_state.json, and resuming from it failed all five D2 arms
        # of 8856525 in about a minute (TODO FT25).
        block = script[script.index("RESUME_JOB"):]
        assert "trainer_state.json" in block, \
            "resume must pick the newest COMPLETE checkpoint, not just the newest"

    def test_evaluation_still_strips_it(self):
        """eval_checkpoint scores a saved checkpoint; resuming training into it is
        never what is wanted, so the flag has to keep being removed there."""
        import importlib
        source = Path(importlib.import_module("msdelta.eval_checkpoint").__file__).read_text()
        assert '"--resume_from_checkpoint",' in source
