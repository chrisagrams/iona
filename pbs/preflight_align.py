"""Everything about the alignment fine-tune that can be checked without a compute node.

    HF_HOME=... PYTHONPATH=. .venv/bin/python pbs/preflight_align.py

Same contract as pbs/preflight_denoise.py: the real checkpoint loads and real spectra go
through the real model once, but there is no training loop, no XPU and no full-corpus
map. What it cannot reach -- XPU count, xccl, bf16, distributed init, the proxy route --
the job script preflights itself.
"""

from __future__ import annotations

import os
import subprocess
import sys
import traceback
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
ARGS_FILE = os.environ.get("ARGS_FILE", "configs/finetune-align-50m/training.args")
PBS_FILE = os.environ.get("PBS_FILE", "pbs/aurora-finetune.pbs")
WANDB_ENTITY = os.environ.get("EXPECT_WANDB_ENTITY", "CS_Pharm")
WANDB_PROJECT = os.environ.get("EXPECT_WANDB_PROJECT", "msdelta-finetune")

results: list[tuple[str, bool, str]] = []


def check(name):
    def wrap(fn):
        try:
            results.append((name, True, str(fn() or "")))
        except Exception as error:  # noqa: BLE001
            results.append((name, False, f"{type(error).__name__}: {error}"))
            if os.environ.get("PREFLIGHT_TRACEBACK"):
                traceback.print_exc()
        return fn
    return wrap


@check("torch resolves to the frameworks module, not a venv wheel")
def _torch():
    import torch
    if "aurora" not in torch.__file__:
        raise RuntimeError(f"torch from {torch.__file__}: a venv wheel shadows the module")
    return torch.__version__


@check("msdelta.finetune_align imports")
def _import():
    import msdelta.finetune_align  # noqa: F401
    return "ok"


@check("every path the PBS script references exists")
def _paths():
    import re
    text = (REPO / PBS_FILE).read_text()
    refs = {a or b for a, b in re.findall(
        r"\$REPO_DIR/([A-Za-z0-9_./-]+)|(?<![\w/])((?:pbs|configs)/[A-Za-z0-9_./-]+)", text)}
    missing = sorted(r for r in refs if r and not (REPO / r).exists())
    if missing:
        raise FileNotFoundError(", ".join(missing))
    return f"{len(refs)} references"


@check("PBS script is valid bash and can launch this module")
def _bash():
    subprocess.run(["bash", "-n", str(REPO / PBS_FILE)], check=True, capture_output=True)
    if "MODULE=${MODULE:-" not in (REPO / PBS_FILE).read_text():
        raise ValueError("launcher has no MODULE override; it would run the denoise entry point")
    return PBS_FILE


@check("args file has no comments (HfArgumentParser splits on whitespace only)")
def _comments():
    bad = [l for l in (REPO / ARGS_FILE).read_text().splitlines() if l.strip().startswith("#")]
    if bad:
        raise ValueError(f"{len(bad)} comment lines become stray positional args")
    return f"{len((REPO / ARGS_FILE).read_text().split())} tokens"


@check("args file parses into the dataclasses")
def _parse():
    from transformers import HfArgumentParser
    from msdelta.finetune_align import (AlignDataArguments, AlignModelArguments,
                                        AlignTrainingArguments)
    stripped = REPO / "pbs" / ".preflight_align.args"
    stripped.write_text("\n".join(
        l for l in (REPO / ARGS_FILE).read_text().splitlines() if "ddp_backend" not in l))
    try:
        parser = HfArgumentParser((AlignModelArguments, AlignDataArguments, AlignTrainingArguments))
        parsed = parser.parse_args_into_dataclasses(
            args=["--args_file", str(stripped), "--bf16", "false", "--use_cpu", "true",
                  "--report_to", "none"], args_file_flag="--args_file")
    finally:
        stripped.unlink(missing_ok=True)
    globals()["_parsed"] = parsed
    model_args, data_args, training_args = parsed
    return (f"pooling={model_args.pooling} peaks={data_args.max_peaks} "
            f"lr={training_args.learning_rate:g} epochs={training_args.num_train_epochs:g}")


@check(f"W&B destination is {WANDB_ENTITY}/{WANDB_PROJECT}")
def _wandb():
    _, _, training_args = globals()["_parsed"]
    if (training_args.wandb_project, training_args.wandb_entity) != (WANDB_PROJECT, WANDB_ENTITY):
        raise ValueError(f"{training_args.wandb_entity}/{training_args.wandb_project}")
    return f"{WANDB_ENTITY}/{WANDB_PROJECT}"


@check("credentials load (length only, never the value)")
def _keys():
    out = subprocess.run(["bash", "-c", f"export REPO_DIR={REPO}; source {REPO}/pbs/load_keys.sh"],
                         capture_output=True, text=True)
    text = out.stdout + out.stderr
    if "WANDB_API_KEY loaded" not in text:
        raise RuntimeError("WANDB_API_KEY not found")
    return " / ".join(l.strip() for l in text.splitlines() if "loaded" in l)


@check("teacher loads, and is frozen in eval mode")
def _teacher():
    from msdelta.modeling_msdelta import MSDeltaForPreTraining
    from msdelta.reranking import build_alignment_model
    model_args, _, _ = globals()["_parsed"]
    if not os.access(Path(model_args.pretrained_path) / "model.safetensors", os.R_OK):
        raise PermissionError(f"cannot read weights under {model_args.pretrained_path}")
    teacher = MSDeltaForPreTraining.from_pretrained(model_args.pretrained_path)
    model = build_alignment_model(teacher, pooling=model_args.pooling,
                                  hidden_size=model_args.sequence_hidden_size,
                                  num_layers=model_args.sequence_num_layers,
                                  num_heads=model_args.sequence_num_heads)
    globals()["_model"] = model
    if any(p.requires_grad for p in model.spectrum_model.parameters()):
        raise ValueError("teacher is not frozen")
    if model.spectrum_model.training:
        raise ValueError("teacher must be in eval mode or dropout moves the target every epoch")
    student = sum(p.numel() for p in model.sequence_encoder.parameters())
    return f"student {student/1e6:.2f}M, teacher frozen"


@check("student output width equals the teacher's, measured not assumed")
def _width():
    from msdelta.reranking import teacher_embedding_size
    model_args, _, _ = globals()["_parsed"]
    model = globals()["_model"]
    target = teacher_embedding_size(model.spectrum_model, model_args.pooling)
    produced = model.sequence_encoder.projection[-1].out_features
    if target != produced:
        raise ValueError(f"teacher emits {target}, student emits {produced}")
    return f"{target} ({model_args.pooling})"


@check("teacher and student pooling agree")
def _pooling():
    model = globals()["_model"]
    if model.pooling != model.sequence_encoder.pooling:
        raise ValueError("pooling mismatch between towers")
    from msdelta.reranking import POOLING_MODES
    return f"{model.pooling} (available: {', '.join(POOLING_MODES)})"


@check("real spectra and peptides make a batch, and the loss is finite")
def _batch():
    import torch
    from datasets import load_dataset
    from msdelta.processing_msdelta import MSDeltaProcessor
    from msdelta.reranking import AlignmentCollator
    model_args, data_args, _ = globals()["_parsed"]
    processor = MSDeltaProcessor.from_pretrained(
        data_args.processor_name_or_path or model_args.pretrained_path,
        max_peaks=data_args.max_peaks)
    raw = load_dataset(data_args.dataset_repo)["test"].select(range(8))
    rows = []
    for row in raw:
        values = processor(torch.tensor(row["mz"][: data_args.max_peaks]),
                           torch.tensor(row["intensity"][: data_args.max_peaks]), padding=False)
        rows.append({"mz": values["mz"], "log_intensity": values["log_intensity"],
                     "peptide": row["peptide"], "charge": int(row["charge"])})
    batch = AlignmentCollator(max_peptide_length=model_args.max_peptide_length)(rows)
    globals()["_rows"] = rows
    model = globals()["_model"].eval()
    with torch.no_grad():
        out = model(**batch)
    if not torch.isfinite(out["loss"]):
        raise ValueError("loss is not finite")
    if out["embeddings"].shape != out["target"].shape:
        raise ValueError(f"student {tuple(out['embeddings'].shape)} != teacher {tuple(out['target'].shape)}")
    return f"batch {tuple(batch['mz'].shape)}, loss {float(out['loss']):.4f}"


@check("L2 falls and the teacher does not move")
def _learns():
    import torch
    model = globals()["_model"]
    from msdelta.reranking import AlignmentCollator
    model_args, _, _ = globals()["_parsed"]
    batch = AlignmentCollator(max_peptide_length=model_args.max_peptide_length)(globals()["_rows"])
    before = torch.cat([p.detach().flatten()[:8] for p in
                        list(model.spectrum_model.parameters())[:4]]).clone()
    model.train()
    optimizer = torch.optim.AdamW(model.sequence_encoder.parameters(), lr=1e-3)
    first = last = None
    for step in range(30):
        optimizer.zero_grad()
        loss = model(**batch)["loss"]
        loss.backward()
        optimizer.step()
        first = loss.item() if step == 0 else first
        last = loss.item()
    after = torch.cat([p.detach().flatten()[:8] for p in
                       list(model.spectrum_model.parameters())[:4]])
    if not torch.equal(before, after):
        raise ValueError("teacher weights changed; it must stay frozen")
    if last >= first * 0.8:
        raise ValueError(f"L2 did not fall: {first:.4f} -> {last:.4f}")
    return f"L2 {first:.4f} -> {last:.4f}, teacher bit-identical"


@check("eval mode under autocast, where the fused fast path lives")
def _eval_autocast():
    """The gap that let job 8840257 through.

    Everything above runs the model in TRAIN mode in fp32. torch's
    TransformerEncoderLayer keeps a fused kernel it takes only when grad is disabled, so
    train-mode checks cannot reach it, and its autocast guard (transformer.py:869) calls
    the no-argument torch.is_autocast_enabled(), which reports CUDA state and so is blind
    to torch.autocast("xpu"). The result was 200 clean training steps and a crash on the
    first evaluation. This runs the combination that actually failed.

    It is a weaker check on a login node than on a tile -- CPU and XPU do not take the
    fast path under identical conditions -- so a debug run that performs at least one
    evaluation stays mandatory.
    """
    import torch
    from msdelta.reranking import AlignmentCollator
    model_args, _, _ = globals()["_parsed"]
    model = globals()["_model"].eval()
    batch = AlignmentCollator(max_peptide_length=model_args.max_peptide_length)(
        globals()["_rows"])
    with torch.no_grad(), torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        out = model(**batch)
    if not torch.isfinite(out["loss"]):
        raise ValueError("loss is not finite under autocast")
    if out["embeddings"].dtype != torch.float32:
        raise ValueError(f"student emits {out['embeddings'].dtype}, expected float32")
    return f"loss {float(out['loss']):.4f}, student output {out['embeddings'].dtype}"


@check("the student survives bf16 weights, as DeepSpeed gives it")
def _bf16_weights():
    """Job 8840336 died at step 0 under ZeRO-2 with the earlier error inverted.

    The fix for 8840257 forced the student's transformer stack to fp32, which is right
    only while the weights are fp32. DeepSpeed holds them in bf16, so fp32 activations
    then met bf16 weights: "expected scalar type Float but found BFloat16". What the
    fused kernel cannot tolerate is a mismatch, not a particular dtype, so the stack now
    runs in whatever dtype its parameters are. Both cases have to be checked, because
    fixing one broke the other.
    """
    import copy
    import torch
    from msdelta.reranking import AlignmentCollator
    model_args, _, _ = globals()["_parsed"]
    batch = AlignmentCollator(max_peptide_length=model_args.max_peptide_length)(
        globals()["_rows"])
    student = copy.deepcopy(globals()["_model"].sequence_encoder).eval()
    widths = {}
    for dtype in (torch.float32, torch.bfloat16):
        cast = copy.deepcopy(student).to(dtype)
        with torch.no_grad(), torch.autocast(device_type="cpu", dtype=torch.bfloat16):
            out = cast(batch["residues"], batch["modifications"],
                       batch["sequence_mask"], batch["charge"])
        if out.dtype != torch.float32:
            raise ValueError(f"{dtype} weights produced {out.dtype}, expected float32 out")
        widths[str(dtype)] = tuple(out.shape)
    if len(set(widths.values())) != 1:
        raise ValueError(f"shape depends on weight dtype: {widths}")
    return f"fp32 and bf16 weights both give {next(iter(widths.values()))} float32"


@check("the Trainer can actually produce eval_loss")
def _eval_loss_reachable():
    """Job 8840277 evaluated cleanly and then died on metric_for_best_model.

    Trainer only computes a loss during evaluation when the model either takes a `labels`
    argument (find_labels) or a `return_loss=True` one (can_return_loss). This task is
    self-supervised against a frozen teacher and has neither by nature, so evaluation
    returned only eval_runtime and friends, and load_best_model_at_end then raised on a
    missing 'eval_loss'. A stray --label_names made it worse by sending Trainer down the
    has_labels path to look for a key the collator never emits.
    """
    from transformers.utils.generic import can_return_loss, find_labels
    from msdelta.reranking import SequenceAlignmentModel
    _, _, training_args = globals()["_parsed"]
    labels = find_labels(SequenceAlignmentModel)
    if not can_return_loss(SequenceAlignmentModel) and not labels:
        raise ValueError("neither return_loss nor labels in forward(); eval_loss will "
                         "never exist")
    if training_args.label_names:
        raise ValueError(f"label_names={training_args.label_names} but the collator emits "
                         "none; leave it unset so Trainer takes the no-labels path")
    metric = (training_args.metric_for_best_model or "").removeprefix("eval_")
    if training_args.load_best_model_at_end and metric not in ("loss",):
        raise ValueError(f"metric_for_best_model={training_args.metric_for_best_model!r} "
                         "is not produced by this evaluation")
    return f"can_return_loss=True, label_names={training_args.label_names}, best on eval_loss"


@check("cross-modal metrics deduplicate candidates")
def _metrics():
    import numpy as np
    import torch
    from msdelta.reranking import cross_modal_metrics
    torch.manual_seed(0)
    candidates = torch.nn.functional.normalize(torch.randn(4, 16), dim=-1)
    truth = np.array([0, 0, 1, 1, 2, 3])
    out = cross_modal_metrics(candidates, candidates[truth], truth, np.arange(4))
    if out["crossmodal/hit@1"] != 1.0:
        raise ValueError(f"aligned embeddings should rank perfectly: {out}")
    if out["crossmodal/n_candidates"] != 4.0 or out["crossmodal/n_spectra"] != 6.0:
        raise ValueError("candidate deduplication not reflected in the counts")
    return "6 spectra over 4 candidates, hit@1 1.0"


@check("the peptide split holds out whole peptides")
def _split():
    import numpy as np
    rows = globals()["_rows"]
    peptides = sorted({r["peptide"] for r in rows})
    generator = np.random.default_rng(0)
    held = set(generator.choice(peptides, size=max(1, int(len(peptides) * 0.5)),
                                replace=False).tolist())
    train = {r["peptide"] for r in rows if r["peptide"] not in held}
    validation = {r["peptide"] for r in rows if r["peptide"] in held}
    if train & validation:
        raise ValueError("a peptide appears on both sides")
    return f"{len(train)} train / {len(validation)} validation peptides, disjoint"


width = max(len(n) for n, _, _ in results)
print("\n=== alignment fine-tune preflight ===\n")
for name, ok, detail in results:
    print(f"  [{'PASS' if ok else 'FAIL'}] {name:<{width}}  {detail}")
failed = [n for n, ok, _ in results if not ok]
print(f"\n{len(results) - len(failed)}/{len(results)} passed")
if failed:
    print("failed: " + ", ".join(failed))
sys.exit(1 if failed else 0)
