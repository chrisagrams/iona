"""Everything about the denoise fine-tune that can be checked without a compute node.

    HF_HOME=... PYTHONPATH=. .venv/bin/python pbs/preflight_denoise.py

Submitting to find out whether a config parses is a slow way to learn it, and the debug
queue charges real wall-clock for the privilege -- job 8839122 burned a queue slot to
discover a missing shell script. Everything here runs on a login node in under a minute:
the checkpoint loads and a real spectrum goes through the real model once, but there is
no training step, no XPU, and no full-corpus map.

What deliberately is NOT covered, because it cannot be: XPU device count, the xccl
backend, bf16, distributed init, and whether the ALCF proxy reaches api.wandb.ai --
login nodes have no route through it. The job script preflights those itself.
"""

from __future__ import annotations

import os
import subprocess
import sys
import traceback
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
ARGS_FILE = os.environ.get("ARGS_FILE", "configs/finetune-denoise-50m/training.args")
PBS_FILE = os.environ.get("PBS_FILE", "pbs/aurora-finetune.pbs")
# All fine-tuning work reports here. The entity matters as much as the project: the key's
# default entity is personal, and CS_Pharm has same-named projects, so an unset entity
# produces a successful-looking run under the wrong owner (it did, for 8839683).
WANDB_ENTITY = os.environ.get("EXPECT_WANDB_ENTITY", "CS_Pharm")
WANDB_PROJECT = os.environ.get("EXPECT_WANDB_PROJECT", "msdelta-finetune")

results: list[tuple[str, bool, str]] = []


def check(name):
    """Register a check; a raised exception is a failure with its message."""
    def wrap(fn):
        try:
            detail = fn() or ""
            results.append((name, True, str(detail)))
        except Exception as error:  # noqa: BLE001
            results.append((name, False, f"{type(error).__name__}: {error}"))
            if os.environ.get("PREFLIGHT_TRACEBACK"):
                traceback.print_exc()
        return fn
    return wrap


# ---------------------------------------------------------------- environment
@check("torch resolves to the frameworks module, not a venv wheel")
def _torch():
    import torch
    if "aurora" not in torch.__file__:
        raise RuntimeError(f"torch came from {torch.__file__} -- a venv wheel shadows the module")
    return f"{torch.__version__}"


@check("modeling dependencies importable")
def _deps():
    import pytorch_metric_learning, sentence_transformers  # noqa: F401
    return "pytorch_metric_learning, sentence_transformers"


@check("msdelta.finetune_denoise imports")
def _import():
    import msdelta.finetune_denoise as m
    return m.__file__.replace(str(REPO) + "/", "")


# ---------------------------------------------------------------- files and config
@check("every path the PBS script references exists")
def _paths():
    text = (REPO / PBS_FILE).read_text()
    import re
    refs = set()
    for match in re.findall(r"\$REPO_DIR/([A-Za-z0-9_./-]+)|(?<![\w/])((?:pbs|configs)/[A-Za-z0-9_./-]+)", text):
        refs.add(match[0] or match[1])
    missing = sorted(r for r in refs if r and not (REPO / r).exists())
    if missing:
        raise FileNotFoundError(", ".join(missing))
    return f"{len(refs)} references"


@check("PBS script is valid bash")
def _bash():
    subprocess.run(["bash", "-n", str(REPO / PBS_FILE)], check=True, capture_output=True)
    return PBS_FILE


@check("args file has no comments (HfArgumentParser splits on whitespace only)")
def _comments():
    bad = [line for line in (REPO / ARGS_FILE).read_text().splitlines()
           if line.strip().startswith("#")]
    if bad:
        raise ValueError(f"{len(bad)} comment lines would become stray positional args")
    return f"{len(( REPO / ARGS_FILE).read_text().split())} tokens"


@check("args file parses into the dataclasses")
def _parse():
    from transformers import HfArgumentParser
    from msdelta.finetune_denoise import (
        DenoiseDataArguments, DenoiseFinetuneArguments, DenoiseModelArguments,
    )
    # ddp_backend/bf16 need a real accelerator, so they are overridden for the parse only.
    stripped = REPO / "pbs" / ".preflight.args"
    stripped.write_text(
        "\n".join(l for l in (REPO / ARGS_FILE).read_text().splitlines()
                  if "ddp_backend" not in l)
    )
    try:
        parser = HfArgumentParser(
            (DenoiseModelArguments, DenoiseDataArguments, DenoiseFinetuneArguments)
        )
        model_args, data_args, training_args = parser.parse_args_into_dataclasses(
            args=["--args_file", str(stripped), "--bf16", "false", "--use_cpu", "true",
                  "--report_to", "none"],
            args_file_flag="--args_file",
        )
    finally:
        stripped.unlink(missing_ok=True)
    globals()["_parsed"] = (model_args, data_args, training_args)
    return (f"project={training_args.wandb_project} peaks={data_args.max_peaks} "
            f"freeze={model_args.freeze_encoder_steps} lr_scale={model_args.encoder_lr_scale}")


@check(f"W&B destination is {WANDB_ENTITY}/{WANDB_PROJECT} and reporting is on")
def _wandb_cfg():
    _, _, training_args = globals()["_parsed"]
    if training_args.wandb_project != WANDB_PROJECT:
        raise ValueError(f"project is {training_args.wandb_project!r}, expected {WANDB_PROJECT!r}")
    # The key's default entity is the personal one, so an unset entity silently lands the
    # run beside a same-named project under the wrong owner -- which is what happened to
    # run denoise-ft-50m-8839683.
    if training_args.wandb_entity != WANDB_ENTITY:
        raise ValueError(f"entity is {training_args.wandb_entity!r}, expected {WANDB_ENTITY!r}")
    original = (REPO / ARGS_FILE).read_text()
    if "--report_to wandb" not in original:
        raise ValueError("--report_to wandb missing from the args file")
    return f"{WANDB_ENTITY}/{WANDB_PROJECT}"


@check("credentials load (length only, never the value)")
def _keys():
    out = subprocess.run(
        ["bash", "-c", f"export REPO_DIR={REPO}; source {REPO}/pbs/load_keys.sh"],
        capture_output=True, text=True,
    )
    text = out.stdout + out.stderr
    if "WANDB_API_KEY loaded" not in text:
        raise RuntimeError("WANDB_API_KEY not found by load_keys.sh")
    return " / ".join(l.strip() for l in text.splitlines() if "loaded" in l)


# ---------------------------------------------------------------- the real model
@check("pretrained checkpoint is readable and loads")
def _checkpoint():
    from msdelta.finetune_denoise import build_denoising_model
    model_args, _, _ = globals()["_parsed"]
    path = Path(model_args.pretrained_path)
    if not os.access(path / "model.safetensors", os.R_OK):
        raise PermissionError(f"cannot read {path/'model.safetensors'} -- needs chmod g+r")
    model, pretrained = build_denoising_model(str(path), model_args)
    globals()["_model"] = model
    total = sum(p.numel() for p in model.parameters())
    head = sum(p.numel() for p in model.denoising_head.parameters())
    return f"{total/1e6:.2f}M params, {head/1e6:.3f}M in the head"


@check("one real spectrum makes it through the model and produces a loss")
def _forward():
    import torch
    from datasets import load_dataset
    from transformers import DataCollatorWithPadding
    from msdelta.processing_msdelta import MSDeltaProcessor
    model_args, data_args, _ = globals()["_parsed"]
    processor = MSDeltaProcessor.from_pretrained(
        data_args.processor_name_or_path or model_args.pretrained_path,
        max_peaks=data_args.max_peaks,
    )
    rows = load_dataset(data_args.dataset_repo)["validation"].select(range(4))
    encoded = [
        processor.process_denoising_example(r["mz"], r["intensity"], r["noise"])
        for r in rows if 0 < len(r["mz"]) <= processor.max_peaks
    ]
    batch = DataCollatorWithPadding(tokenizer=processor, padding=True, return_tensors="pt")(encoded)
    globals()["_batch"] = batch
    model = globals()["_model"].eval()
    with torch.no_grad():
        out = model(**batch)
    if out.logits.shape != batch["mz"].shape:
        raise ValueError(f"logits {tuple(out.logits.shape)} != mz {tuple(batch['mz'].shape)}")
    if not torch.isfinite(out.loss):
        raise ValueError("loss is not finite")
    return f"batch {tuple(batch['mz'].shape)}, loss {float(out.loss):.4f}"


@check("collator pads labels with -100 and the model ignores them")
def _padding():
    import torch
    batch = globals()["_batch"]
    padded = batch["attention_mask"] == 0
    if not padded.any():
        return "no padding in this batch (all equal length)"
    if not (batch["labels"][padded] == -100).all():
        raise ValueError("padded positions are not -100")
    if set(batch["labels"][~padded].unique().tolist()) - {0.0, 1.0}:
        raise ValueError("real labels are not 0/1")
    noise = float(batch["labels"][~padded].mean())
    return f"{int(padded.sum())} padded positions, noise fraction {noise:.2f}"


@check("noise is the POSITIVE class (label 1), matching denoising_metrics")
def _polarity():
    import numpy as np
    from datasets import load_dataset
    from msdelta.processing_msdelta import MSDeltaProcessor
    model_args, data_args, _ = globals()["_parsed"]
    processor = MSDeltaProcessor.from_pretrained(
        data_args.processor_name_or_path or model_args.pretrained_path,
        max_peaks=data_args.max_peaks,
    )
    row = load_dataset(data_args.dataset_repo)["validation"][0]
    encoded = processor.process_denoising_example(row["mz"], row["intensity"], row["noise"])
    if not np.allclose(encoded["labels"], np.asarray(row["noise"], dtype=float)):
        raise ValueError("labels do not equal the corpus noise flags")
    return f"label==noise for all {len(encoded['labels'])} peaks"


@check("batch sampler respects the attention budget (only if budget batching is on)")
def _sampler():
    from msdelta.denoising import PeakBudgetBatchSampler
    _, _, training_args = globals()["_parsed"]
    if not training_args.use_peak_budget_batching:
        return "budget batching off; fixed batches, as pretraining uses"
    lengths = [1024] * 8 + [195] * 200 + [20] * 50
    sampler = PeakBudgetBatchSampler(
        lengths=lengths, peak_pair_budget=training_args.peak_pair_budget, seed=0
    )
    batches = list(sampler)
    worst = max(len(b) * max(lengths[i] for i in b) ** 2 for b in batches)
    if worst > training_args.peak_pair_budget:
        raise ValueError(f"a batch needs {worst} > budget {training_args.peak_pair_budget}")
    big = [len(b) for b in batches if max(lengths[i] for i in b) == 1024]
    return (f"{len(batches)} batches, worst area {worst:,} <= {training_args.peak_pair_budget:,}; "
            f"{min(big) if big else '-'} spectra per 1024-peak batch")


@check("launcher gives every rank all tiles, as set_device(LOCAL_RANK) requires")
def _affinity():
    """Static check for the mismatch that killed job 8839150.

    `main()` calls torch.xpu.set_device(LOCAL_RANK). That only works if every rank can
    see every tile, so ZE_AFFINITY_MASK must be the full list set once for the job -- a
    per-rank mask leaves one visible device at index 0 and set_device(7) raises.
    """
    import re
    text = (REPO / PBS_FILE).read_text()
    per_rank = re.search(r'export ZE_AFFINITY_MASK="?\$\{?LOCAL_RANK', text)
    if per_rank:
        raise ValueError("ZE_AFFINITY_MASK is set per rank; set_device(LOCAL_RANK) will go out of range")
    if "ZE_AFFINITY_MASK" not in text:
        raise ValueError("ZE_AFFINITY_MASK is never set")
    if "seq -s, 0" not in text:
        raise ValueError("ZE_AFFINITY_MASK is not the full tile list")
    return "job-wide full tile list"


@check("peak_pair_budget is sized for THIS checkpoint's delta_bias_n_freqs")
def _budget():
    """The repo default is tuned for n_freqs=64; these checkpoints use 256.

    The pair branch builds a Fourier encoding of every (i, j) m/z difference, so its
    tensor is batch * L^2 * 2 * n_freqs floats. The budget bounds batch * L^2, which
    means the memory it implies scales with n_freqs -- and the 50m/100m production
    checkpoints quadrupled that from 64 to 256. Job 8839166 trained 19 steps on the
    inherited budget and then took a GPU page fault.
    """
    model_args, data_args, training_args = globals()["_parsed"]
    model = globals()["_model"]
    n_freqs = model.config.encoder.delta_bias_n_freqs
    cap = data_args.max_peaks
    # The bias branch materialises (B, L, L, 2*n_freqs); with a TRAINABLE encoder every
    # intermediate is retained for backward, so the forward figure is a floor.
    area = (training_args.peak_pair_budget if training_args.use_peak_budget_batching
            else training_args.per_device_train_batch_size * cap * cap)
    gigabytes = area * 2 * n_freqs * 4 / 1e9
    if gigabytes > 4.0:
        raise ValueError(
            f"n_freqs={n_freqs}, cap={cap}, area={area:,} imply a {gigabytes:.2f} GB "
            "pair tensor per train batch; lower the batch size or the cap"
        )
    # Eval is NOT covered by the budget: get_eval_dataloader takes the standard fixed-batch
    # path, so per_device_eval_batch_size is the only thing bounding it. Job 8839203 trained
    # all 60 steps and then faulted two batches into evaluation at batch size 8 (17.18 GB).
    eval_gb = training_args.per_device_eval_batch_size * cap * cap * 2 * n_freqs * 4 / 1e9
    if eval_gb > 4.0:
        raise ValueError(
            f"per_device_eval_batch_size={training_args.per_device_eval_batch_size} implies "
            f"{eval_gb:.2f} GB at the {cap}-peak cap; the budget does not apply to eval"
        )
    return (f"n_freqs={n_freqs}, cap={cap}, train {gigabytes:.2f} GB/batch, "
            f"eval {eval_gb:.2f} GB/batch")


@check("optimizer builds two groups at the right learning rates")
def _optimizer():
    import tempfile
    from transformers import DataCollatorWithPadding
    from msdelta.finetune_denoise import DenoiseFinetuneTrainer
    from msdelta.processing_msdelta import MSDeltaProcessor
    from datasets import Dataset
    model_args, data_args, training_args = globals()["_parsed"]
    processor = MSDeltaProcessor.from_pretrained(
        data_args.processor_name_or_path or model_args.pretrained_path,
        max_peaks=data_args.max_peaks,
    )
    batch = globals()["_batch"]
    tiny = Dataset.from_list([
        {"mz": batch["mz"][0].tolist(), "log_intensity": batch["log_intensity"][0].tolist(),
         "labels": batch["labels"][0].tolist()}
    ])
    with tempfile.TemporaryDirectory() as out:
        args = type(training_args)(**{**training_args.to_dict(), "output_dir": out,
                                      "report_to": [], "use_cpu": True, "bf16": False})
        trainer = DenoiseFinetuneTrainer(
            model=globals()["_model"], args=args, train_dataset=tiny, eval_dataset=tiny,
            data_collator=DataCollatorWithPadding(tokenizer=processor, padding=True,
                                                  return_tensors="pt"),
            processing_class=processor,
            peak_pair_budget=training_args.peak_pair_budget,
            encoder_lr_scale=model_args.encoder_lr_scale,
            freeze_encoder_steps=model_args.freeze_encoder_steps,
        )
        groups = trainer.create_optimizer().param_groups
        rates = [g["lr"] for g in groups]
        expected = [training_args.learning_rate * model_args.encoder_lr_scale,
                    training_args.learning_rate]
        if len(groups) != 2 or not all(abs(a - b) < 1e-12 for a, b in zip(rates, expected)):
            raise ValueError(f"groups {rates} != expected {expected}")

        # The encoder must be TRAINABLE at construction even when freeze_encoder_steps>0.
        # DDP builds its reducer from the parameters that require grad at wrap time, so a
        # parameter frozen then is never all-reduced if it is later unfrozen -- each rank
        # would train a different encoder and only rank 0's would be saved. The freeze is
        # implemented by discarding the encoder's gradient instead.
        encoder = [p for n, p in trainer.model.named_parameters()
                   if n.startswith("msdelta.") and "mask_token" not in n]
        if not all(p.requires_grad for p in encoder):
            raise ValueError("encoder must stay trainable so DDP tracks it; freeze via grads")

        # mask_token is never read when mask_positions is absent, which denoising never
        # supplies, so it must be frozen before the wrap or DDP aborts on an unused param.
        if trainer.model.msdelta.embed.mask_token.requires_grad:
            raise ValueError("mask_token must be frozen: it can never receive a gradient")
        return (f"encoder lr {rates[0]:g}, head lr {rates[1]:g}, "
                f"encoder trainable={all(p.requires_grad for p in encoder)}, mask_token frozen")


@check("the freeze is an LR gate, and the encoder/head rates follow it")
def _freeze_mechanism():
    """Trace both groups' learning rates across a full run.

    The freeze is a per-group LambdaLR multiplier now, not a gradient edit. That matters
    because the two mechanisms it replaces both fight DDP: toggling requires_grad breaks
    the reducer (built once, at wrap time) and zeroing .grad writes into the flat
    all-reduce bucket DDP owns -- GPU page faults on a write, jobs 8840007/8/10/11.
    An LR multiplier touches only param_group["lr"], which DDP never reads.
    """
    import tempfile
    from datasets import Dataset
    from transformers import DataCollatorWithPadding
    from msdelta.finetune_denoise import DenoiseFinetuneTrainer
    from msdelta.processing_msdelta import MSDeltaProcessor

    model_args, data_args, training_args = globals()["_parsed"]
    if model_args.freeze_encoder_steps <= 0:
        return "freeze disabled in this config; gate not exercised"
    processor = MSDeltaProcessor.from_pretrained(
        data_args.processor_name_or_path or model_args.pretrained_path,
        max_peaks=data_args.max_peaks,
    )
    tiny = Dataset.from_list([{"mz": [100.0, 200.0], "log_intensity": [1.0, 0.5],
                               "labels": [1.0, 0.0]}])
    steps = 2018
    with tempfile.TemporaryDirectory() as out:
        args = type(training_args)(**{**training_args.to_dict(), "output_dir": out,
                                      "report_to": [], "use_cpu": True, "bf16": False,
                                      "eval_strategy": "no", "save_strategy": "no"})
        trainer = DenoiseFinetuneTrainer(
            model=globals()["_model"], args=args, train_dataset=tiny,
            data_collator=DataCollatorWithPadding(tokenizer=processor, padding=True,
                                                  return_tensors="pt"),
            processing_class=processor,
            peak_pair_budget=training_args.peak_pair_budget,
            encoder_lr_scale=model_args.encoder_lr_scale,
            freeze_encoder_steps=model_args.freeze_encoder_steps,
        )
        optimizer = trainer.create_optimizer()
        scheduler = trainer.create_scheduler(steps, optimizer)
        gate = model_args.freeze_encoder_steps
        trace = {}
        for step in range(steps):
            if step in (0, gate - 1, gate, gate + 200):
                trace[step] = [g["lr"] for g in optimizer.param_groups]
            scheduler.step()

    if trace[gate - 1][0] != 0.0:
        raise ValueError(f"encoder rate nonzero inside the gate: {trace[gate-1][0]}")
    if trace[gate][0] <= 0.0:
        raise ValueError("encoder rate did not resume when the gate opened")
    if trace[gate - 1][1] <= 0.0:
        raise ValueError("head rate should be warming up during the encoder's gate")
    for step in (gate, gate + 200):
        encoder, head = trace[step]
        if abs(encoder / head - model_args.encoder_lr_scale) > 1e-9:
            raise ValueError(f"step {step}: ratio {encoder/head:g} != {model_args.encoder_lr_scale}")
    return (f"encoder 0 until step {gate}, then {model_args.encoder_lr_scale:g}x the head "
            f"on the same curve")


@check("metrics survive a poisoned gather instead of killing the run")
def _metrics():
    """The three inputs that have to not crash: clean, a stray label value, one class.

    Job 8839579 lost a four-hour allocation at the first evaluation because upstream's
    denoising_metrics passed an unexpected label value straight to sklearn.
    """
    import numpy as np
    from transformers import EvalPrediction
    from msdelta.finetune_denoise import denoise_metrics

    clean = denoise_metrics(EvalPrediction(
        predictions=np.array([[3.0, -3.0, 2.0, 99.0]]),
        label_ids=np.array([[1.0, 0.0, 1.0, -100.0]])))
    if clean["accuracy"] != 1.0 or clean["auroc"] != 1.0:
        raise ValueError(f"perfect input did not score perfectly: {clean}")

    poisoned = denoise_metrics(EvalPrediction(
        predictions=np.array([[3.0, -3.0, 2.0, 1.0]]),
        label_ids=np.array([[1.0, 0.0, 1.0, 7.0]])))
    if poisoned["label_dropped"] != 1.0 or poisoned["auroc"] != 1.0:
        raise ValueError(f"a stray label was not dropped and reported: {poisoned}")

    one_class = denoise_metrics(EvalPrediction(
        predictions=np.array([[3.0, 2.0]]), label_ids=np.array([[1.0, 1.0]])))
    if one_class["auroc"] != 0.5:
        raise ValueError(f"single-class slice should be 0.5, got {one_class['auroc']}")
    return "clean / stray label reported / single class survives"


# ---------------------------------------------------------------- report
width = max(len(n) for n, _, _ in results)
print("\n=== denoise fine-tune preflight ===\n")
for name, ok, detail in results:
    print(f"  [{'PASS' if ok else 'FAIL'}] {name:<{width}}  {detail}")
failed = [n for n, ok, _ in results if not ok]
print(f"\n{len(results) - len(failed)}/{len(results)} passed")
if failed:
    print("failed: " + ", ".join(failed))
print("\nNot covered here (needs a compute node): XPU count, xccl, bf16, distributed init,")
print("and the ALCF proxy route to api.wandb.ai -- login nodes have no route through it.")
sys.exit(1 if failed else 0)
