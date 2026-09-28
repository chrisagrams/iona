"""Opt-in per-block wall-time breakdown for a few training steps (K121-P).

OFF unless ``--block_timing_steps N`` (N > 0) is given. Then, for optimizer steps
``block_timing_start_step .. block_timing_start_step + N - 1``, forward hooks and full
backward hooks on the encoder's sub-blocks synchronise the device before and after each
block and record the wall time; after the last timed step the hooks are removed and
``<output_dir>/block_timing.json`` is written (world rank 0). Nothing is registered outside
that window, so the other steps run exactly as without the flag.

Groups (module-name patterns relative to ``MSDeltaForPreTraining``; letters as in
``notes/PAIRFORMER.md`` D2; a group whose modules never run is left out):

* both: ``embed``, ``block_total`` (one encoder layer's single-stream block), ``final_norm``,
  ``head``
* Pairformer: ``pair_features`` (Δm/z Fourier bank + chemistry features), ``z_init_proj``
  (W_a, W_b, W_c), ``pair_layer_total`` (one pair layer, a..g), ``a_writeback``,
  ``b_tri_mul_out``, ``c_tri_mul_in``, ``d_tri_attn_start``, ``e_tri_attn_end``,
  ``f_pair_transition``, ``g_bias_readout``, ``i_single_transition``; derived
  ``h_single_attention = block_total - i_single_transition``
* transformer: ``delta_bias`` (DeltaMZBias), ``attention``, ``ffn``

Each group has three phases, in ms per optimizer step (all micro-steps summed, averaged over
the timed steps): ``fwd`` (the forward pass), ``recompute`` (the same blocks re-run inside
backward by gradient checkpointing) and ``bwd`` (full backward hooks: from the gradient
w.r.t. the block's output to the gradient w.r.t. its input). Caveats, also in the JSON:

* The synchronisations serialise the device, so the timed steps are slower than untimed
  ones. Use the breakdown for proportions; take s/step from the untimed steps.
* Under checkpointing, the ``bwd`` of the block that first needs a saved tensor of a
  checkpointed segment includes that segment's recompute (reported separately as
  ``recompute``); ``*_total`` groups nest their sub-blocks.
* Under DDP the backward includes gradient all-reduce traffic that overlaps the blocks.
"""

from __future__ import annotations

import json
import re
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable

import torch
from torch import nn
from transformers import TrainerCallback

# (group, regex over names from model.named_modules()); first match wins.
BLOCK_GROUPS: tuple[tuple[str, str], ...] = (
    ("embed", r"msdelta\.embed"),
    ("pair_features", r"msdelta\.bias_module\.pair_feats"),
    ("z_init_proj", r"msdelta\.bias_module\.w_[abc]"),
    ("delta_bias", r"msdelta\.bias_module"),
    ("pair_layer_total", r"msdelta\.bias_module\.layers\.(\d+)"),
    ("a_writeback", r"msdelta\.bias_module\.layers\.(\d+)\.opm"),
    ("b_tri_mul_out", r"msdelta\.bias_module\.layers\.(\d+)\.tri_out"),
    ("c_tri_mul_in", r"msdelta\.bias_module\.layers\.(\d+)\.tri_in"),
    ("d_tri_attn_start", r"msdelta\.bias_module\.layers\.(\d+)\.tri_attn_start"),
    ("e_tri_attn_end", r"msdelta\.bias_module\.layers\.(\d+)\.tri_attn_end"),
    ("f_pair_transition", r"msdelta\.bias_module\.layers\.(\d+)\.transition"),
    ("g_bias_readout", r"msdelta\.bias_module\.layers\.(\d+)\.(?:bias_norm|to_bias)"),
    ("block_total", r"msdelta\.blocks\.(\d+)"),
    ("i_single_transition", r"msdelta\.blocks\.(\d+)\.transition"),
    ("attention", r"msdelta\.blocks\.(\d+)\.attn"),
    ("ffn", r"msdelta\.blocks\.(\d+)\.ffn"),
    ("final_norm", r"msdelta\.norm"),
    ("head", r"intensity_head"),
)
PHASES = ("fwd", "recompute", "bwd")


def _synchronizer(module: nn.Module) -> Callable[[], None]:
    try:
        device = next(module.parameters()).device
    except StopIteration:
        return lambda: None
    if device.type == "cuda":
        return lambda: torch.cuda.synchronize(device)
    if device.type == "xpu":
        return lambda: torch.xpu.synchronize(device)
    return lambda: None


def _in_backward() -> bool:
    graph_task_id = getattr(torch._C, "_current_graph_task_id", None)
    return graph_task_id is not None and graph_task_id() != -1


class BlockTimer:
    """Attach/detach the timing hooks and accumulate milliseconds per (group, layer, phase)."""

    def __init__(self, model: nn.Module):
        self.model = model
        self.sync = _synchronizer(model)
        self.handles: list[Any] = []
        self.ms: dict[tuple[str, int, str], float] = defaultdict(float)
        self.calls: dict[tuple[str, int, str], int] = defaultdict(int)
        self.batch_shapes: list[list[int]] = []
        self.targets = self._match(model)

    @staticmethod
    def _match(model: nn.Module) -> list[tuple[str, int, nn.Module]]:
        targets = []
        for name, module in model.named_modules():
            for group, pattern in BLOCK_GROUPS:
                match = re.fullmatch(pattern, name)
                if match:
                    layer = int(match.group(1)) if match.groups() else -1
                    targets.append((group, layer, module))
                    break
        return targets

    @property
    def attached(self) -> bool:
        return bool(self.handles)

    def attach(self) -> None:
        if self.attached:
            return
        for group, layer, module in self.targets:
            self._attach_one(group, layer, module)
        self.handles.append(
            self.model.register_forward_pre_hook(self._record_shape, with_kwargs=True)
        )

    def detach(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles = []

    def _record_shape(self, module, args, kwargs):
        mz = args[0] if args else kwargs.get("mz")
        if isinstance(mz, torch.Tensor) and module.training and not _in_backward():
            self.batch_shapes.append(list(mz.shape))

    def _attach_one(self, group: str, layer: int, module: nn.Module) -> None:
        fwd_starts: list[float] = []
        bwd_starts: list[float] = []

        def fwd_pre(mod, args):
            if mod.training:
                self.sync()
                fwd_starts.append(time.perf_counter())

        def fwd_post(mod, args, output):
            if mod.training and fwd_starts:
                self.sync()
                phase = "recompute" if _in_backward() else "fwd"
                self._add(group, layer, phase, time.perf_counter() - fwd_starts.pop())

        def bwd_pre(mod, grad_output):
            self.sync()
            bwd_starts.append(time.perf_counter())

        def bwd_post(mod, grad_input, grad_output):
            if bwd_starts:
                self.sync()
                self._add(group, layer, "bwd", time.perf_counter() - bwd_starts.pop())

        self.handles += [
            module.register_forward_pre_hook(fwd_pre),
            module.register_forward_hook(fwd_post),
            module.register_full_backward_pre_hook(bwd_pre),
            module.register_full_backward_hook(bwd_post),
        ]

    def _add(self, group: str, layer: int, phase: str, seconds: float) -> None:
        self.ms[(group, layer, phase)] += 1e3 * seconds
        self.calls[(group, layer, phase)] += 1

    def summary(self, n_steps: int, step_ms: list[float], extra: dict | None = None) -> dict:
        n = max(n_steps, 1)
        groups: dict[str, dict[str, float]] = {}
        per_layer: dict[str, dict[str, list[float]]] = {}
        for (group, layer, phase), ms in sorted(self.ms.items()):
            entry = groups.setdefault(group, {p: 0.0 for p in PHASES})
            entry[phase] += ms / n
            if layer >= 0:
                rows = per_layer.setdefault(group, {p: [] for p in PHASES})[phase]
                while len(rows) <= layer:
                    rows.append(0.0)
                rows[layer] += ms / n
        for group, entry in groups.items():
            entry["total"] = sum(entry[p] for p in PHASES)
            entry["calls_per_step"] = sum(
                c for (g, _, _), c in self.calls.items() if g == group
            ) / n
        derived = {}
        if "block_total" in groups and "i_single_transition" in groups:
            derived["h_single_attention"] = {
                p: groups["block_total"][p] - groups["i_single_transition"][p]
                for p in (*PHASES, "total")
            }
        mean_step = sum(step_ms) / len(step_ms) if step_ms else None
        return {
            "kind": "msdelta-block-timing",
            "version": 1,
            "timed_steps": n_steps,
            "step_ms": step_ms,
            "step_ms_mean": mean_step,
            "micro_batch_shapes": self.batch_shapes,
            "groups_ms_per_step": groups,
            "derived_ms_per_step": derived,
            "per_layer_ms_per_step": per_layer,
            "notes": [
                "device synchronised around every timed block: timed steps are slower than "
                "untimed ones; use proportions, take s/step from untimed steps",
                "under gradient checkpointing a segment's recompute also lands in the bwd of "
                "the block that first unpacks a saved tensor; it is reported as 'recompute'",
                "*_total groups contain their sub-blocks; under DDP bwd overlaps all-reduce",
            ],
            **(extra or {}),
        }


class BlockTimingCallback(TrainerCallback):
    """Time ``steps`` optimizer steps starting at ``start_step`` and write a JSON summary."""

    def __init__(self, model: nn.Module, steps: int, start_step: int, out_path: str | Path):
        if steps < 1 or start_step < 1:
            raise ValueError("block timing needs steps >= 1 and start_step >= 1")
        self.timer = BlockTimer(model)
        self.model = model
        self.first = start_step
        self.last = start_step + steps - 1
        self.out_path = Path(out_path)
        self.step_ms: list[float] = []
        self._t0: float | None = None
        self.done = False

    def on_step_begin(self, args, state, control, **kwargs):
        step = state.global_step + 1
        if not self.done and self.first <= step <= self.last:
            self.timer.attach()
            self.timer.sync()
            self._t0 = time.perf_counter()
        return control

    def on_step_end(self, args, state, control, **kwargs):
        if self._t0 is None:
            return control
        self.timer.sync()
        self.step_ms.append(1e3 * (time.perf_counter() - self._t0))
        self._t0 = None
        if state.global_step >= self.last:
            self._finish(args, state)
        return control

    def on_train_end(self, args, state, control, **kwargs):
        if not self.done and self.timer.attached:
            self._finish(args, state)
        return control

    def _finish(self, args, state) -> None:
        self.timer.detach()
        self.done = True
        config = getattr(self.model, "config", None)
        extra = {
            "first_step": self.first,
            "last_step": self.last,
            "architecture": getattr(config, "architecture", "transformer"),
            "gradient_checkpointing": bool(getattr(args, "gradient_checkpointing", False)),
            "per_device_train_batch_size": args.per_device_train_batch_size,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "world_size": args.world_size,
            "device": str(next(self.model.parameters()).device),
        }
        result = self.timer.summary(len(self.step_ms), self.step_ms, extra)
        if state.is_world_process_zero:
            self.out_path.parent.mkdir(parents=True, exist_ok=True)
            self.out_path.write_text(json.dumps(result, indent=1) + "\n")
            top = sorted(
                result["groups_ms_per_step"].items(), key=lambda kv: -kv[1]["total"]
            )[:6]
            print(
                f"[block-timing] steps {self.first}-{self.last}: "
                f"{result['step_ms_mean']:.0f} ms/step (synchronised); "
                + ", ".join(f"{g} {v['total']:.0f}" for g, v in top)
                + f" ms -> {self.out_path}",
                flush=True,
            )
