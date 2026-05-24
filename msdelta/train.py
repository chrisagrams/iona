"""Pretraining entrypoint: m/z denoising autoencoder with Δm/z-biased transformer."""
from __future__ import annotations

import argparse
import math
import os
import sys
import time
from contextlib import nullcontext
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch.optim import AdamW
from torch.utils.data import DataLoader

import wandb

from .data import (
    ConsensusParquet,
    DenoiseConfig,
    PreprocessConfig,
    denoise_collate,
    split_paths,
)
from .model import (
    DeltaBiasConfig,
    DenoiseHead,
    FourierConfig,
    MSEncoder,
    ModelConfig,
)
from .probe import run_all_probes
from .viz import attention_entropy_per_head, render_bias_panels


# ---------- config ----------

def build_model_config(d: dict[str, Any]) -> ModelConfig:
    return ModelConfig(
        d_model=d["d_model"],
        n_heads=d["n_heads"],
        n_layers=d["n_layers"],
        ffn_mult=d["ffn_mult"],
        dropout=d["dropout"],
        max_peaks=d["max_peaks"],
        fourier_mz=FourierConfig(**d["fourier_mz"]),
        fourier_int=FourierConfig(**d["fourier_int"]),
        delta_bias=DeltaBiasConfig(**d["delta_bias"]),
    )


def load_config(path: str | Path) -> dict[str, Any]:
    with open(path) as f:
        return yaml.safe_load(f)


# ---------- scheduler ----------

def lr_lambda(step: int, warmup: int, total: int) -> float:
    if step < warmup:
        return step / max(1, warmup)
    if step >= total:
        return 0.0
    progress = (step - warmup) / max(1, total - warmup)
    return 0.5 * (1.0 + math.cos(math.pi * progress))


# ---------- training ----------

def grad_norm(parameters) -> float:
    total = 0.0
    for p in parameters:
        if p.grad is not None:
            total += p.grad.detach().float().pow(2).sum().item()
    return math.sqrt(total)


def make_loaders(cfg: dict[str, Any]) -> tuple[DataLoader, DataLoader]:
    dcfg = cfg["data"]
    pp = PreprocessConfig(
        intensity_threshold_frac=dcfg["intensity_threshold_frac"],
        top_n=dcfg["top_n"],
    )
    train_paths, val_paths = split_paths(dcfg["root"], dcfg["n_val_files"])
    print(f"[data] {len(train_paths)} train shards, {len(val_paths)} val shards", flush=True)

    train_ds = ConsensusParquet(train_paths, preprocess=pp, seed=cfg["train"].get("seed", 0))
    val_ds = ConsensusParquet(val_paths, preprocess=pp, seed=cfg["train"].get("seed", 0) + 1)
    print(
        f"[data] train row-groups: {len(train_ds._units)}  val row-groups: {len(val_ds._units)}",
        flush=True,
    )

    den_cfg = DenoiseConfig(
        gauss_sigma=dcfg["denoise"]["gauss_sigma"],
    )

    def collate(batch):
        batch = [b for b in batch if b[0].numel() > 0]
        if not batch:
            return None
        return denoise_collate(batch, den_cfg)

    tcfg = cfg["train"]
    train_loader = DataLoader(
        train_ds,
        batch_size=tcfg["batch_size"],
        num_workers=tcfg["num_workers"],
        collate_fn=collate,
        pin_memory=True,
        persistent_workers=tcfg["num_workers"] > 0,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=tcfg["batch_size"],
        num_workers=max(1, tcfg["num_workers"] // 2) if tcfg["num_workers"] > 0 else 0,
        collate_fn=collate,
        pin_memory=True,
        persistent_workers=tcfg["num_workers"] > 0,
    )
    return train_loader, val_loader


def to_device(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {k: v.to(device, non_blocking=True) for k, v in batch.items()}


def run_validation(
    encoder: MSEncoder,
    heads: DenoiseHead,
    val_loader: DataLoader,
    device: torch.device,
    autocast_ctx,
    max_batches: int,
) -> dict[str, float]:
    encoder.eval()
    heads.eval()
    sums = {"nll_mz": 0.0, "rmse_mz": 0.0}
    n = 0
    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            if batch is None:
                continue
            if i >= max_batches:
                break
            batch = to_device(batch, device)
            with autocast_ctx:
                tokens = encoder(batch["mz_noisy"], batch["log_int"], batch["key_padding_mask"])
            loss, parts = heads.loss(
                tokens, batch["mz_noisy"], batch["mz_clean"], batch["key_padding_mask"],
            )
            sums["nll_mz"] += float(parts["nll_mz"])
            sums["rmse_mz"] += float(parts["rmse_mz"])
            n += 1
    encoder.train()
    heads.train()
    return {f"val/{k}": v / max(1, n) for k, v in sums.items()}


def save_ckpt(path: Path, encoder, heads, optimizer, scheduler, step, cfg) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "step": step,
        "encoder": encoder.state_dict(),
        "heads": heads.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "cfg": cfg,
    }, path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--run-name", type=str, default=None,
                        help="override log.wandb_run_name and the output dir name")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    if args.run_name:
        cfg["log"]["wandb_run_name"] = args.run_name

    tcfg = cfg["train"]
    lcfg = cfg["log"]

    device = torch.device(tcfg["device"])
    run_name = lcfg["wandb_run_name"] or time.strftime("%Y%m%d-%H%M%S")
    out_dir = Path(lcfg["out_dir"]) / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "figs").mkdir(exist_ok=True)
    print(f"[run] {run_name} → {out_dir}", flush=True)

    # wandb
    wandb.init(
        project=lcfg["wandb_project"],
        name=run_name,
        config=cfg,
        dir=str(out_dir),
    )

    # Model
    model_cfg = build_model_config(cfg["model"])
    encoder = MSEncoder(model_cfg).to(device)
    heads = DenoiseHead(model_cfg.d_model).to(device)
    n_params = sum(p.numel() for p in encoder.parameters()) + sum(p.numel() for p in heads.parameters())
    print(f"[model] {n_params/1e6:.2f}M params", flush=True)

    # Optimizer & scheduler
    params = list(encoder.parameters()) + list(heads.parameters())
    optimizer = AdamW(params, lr=tcfg["lr"], weight_decay=tcfg["weight_decay"], betas=(0.9, 0.95))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=lambda s: lr_lambda(s, tcfg["warmup_steps"], tcfg["total_steps"])
    )

    # Loaders
    train_loader, val_loader = make_loaders(cfg)

    # Probe inputs (frozen-encoder linear probes, logged every log.probe_every steps)
    _, probe_val_paths = split_paths(cfg["data"]["root"], cfg["data"]["n_val_files"])
    probe_pp = PreprocessConfig(
        intensity_threshold_frac=cfg["data"]["intensity_threshold_frac"],
        top_n=cfg["data"]["top_n"],
    )

    # Precision
    if tcfg["precision"] == "bf16":
        autocast_ctx = torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16)
    elif tcfg["precision"] == "fp16":
        autocast_ctx = torch.amp.autocast(device_type=device.type, dtype=torch.float16)
    else:
        autocast_ctx = nullcontext()

    # Resume
    start_step = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        encoder.load_state_dict(ckpt["encoder"])
        heads.load_state_dict(ckpt["heads"])
        optimizer.load_state_dict(ckpt["optimizer"])
        if ckpt.get("scheduler") is not None:
            scheduler.load_state_dict(ckpt["scheduler"])
        start_step = int(ckpt["step"]) + 1
        print(f"[resume] step {start_step} from {args.resume}", flush=True)

    encoder.train()
    heads.train()
    delta_bias_params = list(encoder.bias_module.parameters())

    step = start_step
    t0 = time.time()
    running_loss = 0.0
    running_n = 0

    train_iter = iter(train_loader)
    while step < tcfg["total_steps"]:
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)
        if batch is None:
            continue
        batch = to_device(batch, device)

        # Save attention once just before bias-curve renders so we can also
        # log per-head attention entropy at the same cadence.
        save_attn_this_step = (step % lcfg["bias_curve_every"] == 0)
        encoder.set_save_attn(save_attn_this_step)

        with autocast_ctx:
            tokens = encoder(batch["mz_noisy"], batch["log_int"], batch["key_padding_mask"])
        loss, parts = heads.loss(
            tokens, batch["mz_noisy"], batch["mz_clean"], batch["key_padding_mask"],
        )
        # L1 sparsity penalty on the bias curve (λ=0 → no-op, reproduces denoise baseline).
        l1_lambda = tcfg.get("l1_lambda", 0.0)
        if l1_lambda > 0:
            bias_l1 = encoder.bias_module.l1_penalty()
            loss = loss + l1_lambda * bias_l1
        else:
            bias_l1 = torch.zeros((), device=device)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gn_total = grad_norm(params)
        gn_bias = grad_norm(delta_bias_params)
        torch.nn.utils.clip_grad_norm_(params, tcfg["grad_clip"])
        optimizer.step()
        scheduler.step()

        running_loss += float(loss)
        running_n += 1

        if save_attn_this_step:
            ent = attention_entropy_per_head(encoder.blocks)
            encoder.set_save_attn(False)
        else:
            ent = None

        # Logging
        if step % lcfg["log_every"] == 0:
            lr = optimizer.param_groups[0]["lr"]
            wandb_log = {
                "train/loss": running_loss / max(1, running_n),
                "train/nll_mz": float(parts["nll_mz"]),
                "train/rmse_mz": float(parts["rmse_mz"]),
                "train/mean_log_var": float(parts["mean_log_var"]),
                "train/bias_l1": float(bias_l1),
                "train/grad_norm_total": gn_total,
                "train/grad_norm_delta_bias": gn_bias,
                "train/lr": lr,
                "train/step_per_sec": running_n / max(1e-6, time.time() - t0),
                "step": step,
            }
            if ent is not None:
                for layer_i, row in enumerate(ent):
                    for h_i, e in enumerate(row):
                        wandb_log[f"attn_entropy/L{layer_i}_H{h_i}"] = float(e)
            wandb.log(wandb_log, step=step)
            print(
                f"step {step:>6} loss {wandb_log['train/loss']:.4f} "
                f"nll {wandb_log['train/nll_mz']:.4f} rmse {wandb_log['train/rmse_mz']:.4f} "
                f"|g| {gn_total:.3f} |g_bias| {gn_bias:.3f} lr {lr:.2e}",
                flush=True,
            )
            running_loss = 0.0
            running_n = 0
            t0 = time.time()

        if step > 0 and step % lcfg["val_every"] == 0:
            val_metrics = run_validation(
                encoder, heads, val_loader, device, autocast_ctx,
                max_batches=tcfg["val_batches"],
            )
            wandb.log({**val_metrics, "step": step}, step=step)
            print(f"  val: " + "  ".join(f"{k}={v:.4f}" for k, v in val_metrics.items()), flush=True)

        probe_every = lcfg.get("probe_every", 0)
        if probe_every and step > 0 and step % probe_every == 0:
            probe_metrics = run_all_probes(
                encoder, probe_val_paths, device, probe_pp,
                n_spectra=lcfg.get("probe_n_spectra", 3000),
            )
            wandb.log({**probe_metrics, "step": step}, step=step)
            key = lambda k: probe_metrics.get(k, float("nan"))
            print(f"  probe: precursor_r2={key('probe/precursor_mz_r2'):.3f} "
                  f"charge_acc={key('probe/charge_acc'):.3f} "
                  f"nloss_auc={key('probe/neutral_loss_auc'):.3f} "
                  f"iso_f1={key('probe/isotope_f1'):.3f}", flush=True)

        if step % lcfg["bias_curve_every"] == 0:
            panels = render_bias_panels(encoder.bias_module, step)
            wandb_imgs = {}
            for name, fig in panels.items():
                fig_path = out_dir / "figs" / f"{name.replace('/', '_')}_step{step:06d}.png"
                fig.savefig(fig_path, dpi=110)
                wandb_imgs[name] = wandb.Image(str(fig_path))
                import matplotlib.pyplot as plt
                plt.close(fig)
            wandb.log({**wandb_imgs, "step": step}, step=step)

        if step > 0 and step % lcfg["ckpt_every"] == 0:
            ckpt_path = out_dir / f"step{step:06d}.pt"
            save_ckpt(ckpt_path, encoder, heads, optimizer, scheduler, step, cfg)
            save_ckpt(out_dir / "last.pt", encoder, heads, optimizer, scheduler, step, cfg)
            print(f"  ckpt → {ckpt_path}", flush=True)

        step += 1

    # Final
    save_ckpt(out_dir / "final.pt", encoder, heads, optimizer, scheduler, step, cfg)
    panels = render_bias_panels(encoder.bias_module, step)
    import matplotlib.pyplot as plt
    for name, fig in panels.items():
        fig_path = out_dir / "figs" / f"{name.replace('/', '_')}_final.png"
        fig.savefig(fig_path, dpi=110)
        plt.close(fig)
    wandb.finish()
    return 0


if __name__ == "__main__":
    sys.exit(main())
