"""Contrastive post-training entrypoint.

Loads a masked-intensity-pretrained `MSEncoder` and fine-tunes it (full encoder,
low LR) with a supervised-contrastive objective on two augmented views per
spectrum, so the *raw* pooled-embedding cosine geometry becomes retrieval-good
without eval-time whitening. Two matched runs (MassIVE-KB vs consensus) answer
"which corpus yields more powerful embeddings", scored by the inline
replicate-retrieval benchmark (Hit@1 / MAP / PairF1).

Reuses config/DDP/schedule/eval scaffolding from `train.py`; only the collate
(two-view), the forward (pool→project→SupCon, optional aux-KL), the checkpoint
(adds the projection head), and the resume semantics (cold-init the encoder from
a pretrained ckpt) differ.
"""
from __future__ import annotations

import argparse
import math
import sys
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.utils.data import DataLoader

import wandb

from .data import (
    ConsensusParquet,
    MaskConfig,
    PreprocessConfig,
    contrastive_collate,
    resolve_dataset_paths,
)
from .model import IntensityHead, MSEncoder
from .contrastive import AugmentConfig, ProjectionHead, SupConLoss
from .probe import _pool, run_all_probes
from .analyze import alignment_metrics
from .retrieval import retrieval_inline_metrics
from .replicate_retrieval import replicate_retrieval_inline_metrics
from .viz import attention_entropy_per_head, render_bias_panels

# Reuse the pretraining scaffolding verbatim.
from .train import (
    build_model_config,
    load_config,
    setup_distributed,
    lr_lambda,
    grad_norm,
    to_device,
)


class ContrastiveModule(nn.Module):
    """Encoder + projection head (+ optional intensity head) under one DDP wrapper.

    Forward builds the whole loss graph — contrastive on the two-view batch plus
    (optionally) the masked-intensity KL on view A — so a single backward
    all-reduces every trained parameter.
    """

    def __init__(
        self,
        encoder: MSEncoder,
        proj_head: ProjectionHead,
        supcon: SupConLoss,
        intensity_head: IntensityHead | None = None,
        lambda_kl: float = 0.0,
    ):
        super().__init__()
        self.encoder = encoder
        self.proj_head = proj_head
        self.supcon = supcon
        self.intensity_head = intensity_head
        self.lambda_kl = lambda_kl

    def _embed(self, batch, sl=slice(None), mask_positions=None):
        tokens = self.encoder(
            batch["mz"][sl], batch["log_int"][sl],
            batch["key_padding_mask"][sl], mask_positions,
            charge=batch["charge"][sl], precursor_mz=batch["precursor_mz"][sl],
        )
        return tokens

    def forward(self, batch: dict[str, torch.Tensor]):
        # Clean (unmasked) encode of all 2B rows → pooled → projected → SupCon.
        tokens = self._embed(batch)                               # (2B, K, D)
        pooled = _pool(tokens, ~batch["key_padding_mask"])        # (2B, 2D)
        z = self.proj_head(pooled)                                # (2B, P), L2-normed
        contrastive, parts = self.supcon(z, batch["label"])

        loss = contrastive
        kl = z.new_zeros(())
        if self.lambda_kl > 0 and self.intensity_head is not None and "mask_positions" in batch:
            # One extra masked forward on view A only (rows [0:B]) — do NOT pool a
            # masked forward for the embedding (mask tokens would dominate).
            B = batch["mz"].size(0) // 2
            a = slice(0, B)
            mtokens = self._embed(batch, sl=a, mask_positions=batch["mask_positions"][a])
            kl, _ = self.intensity_head.loss(
                mtokens, batch["intensity_prob"][a], batch["mask_positions"][a])
            loss = contrastive + self.lambda_kl * kl

        parts = {"contrastive": contrastive.detach(), "kl": kl.detach(), **parts}
        return loss, parts


def save_ckpt(path: Path, encoder, proj_head, intensity_head, optimizer, scheduler, step, cfg) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "step": step,
        "encoder": encoder.state_dict(),
        # Keep the `heads` key so analyze.load_encoder / the retrieval CLI load
        # the encoder unchanged; None when the aux-KL head isn't used.
        "heads": intensity_head.state_dict() if intensity_head is not None else None,
        "proj_head": proj_head.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "cfg": cfg,
    }, path)


def make_loaders(
    cfg: dict[str, Any],
    train_paths: list[Path],
    val_paths: list[Path],
    aug_cfg: AugmentConfig,
    mode: str,
    mask_cfg: MaskConfig | None,
    rank: int = 0,
    world_size: int = 1,
) -> tuple[DataLoader, DataLoader]:
    dcfg = cfg["data"]
    tcfg = cfg["train"]
    pp = PreprocessConfig(
        intensity_threshold_frac=dcfg["intensity_threshold_frac"],
        top_n=dcfg["top_n"],
    )
    print(f"[data] {len(train_paths)} train shards, {len(val_paths)} val shards", flush=True)

    train_ds = ConsensusParquet(train_paths, preprocess=pp, seed=tcfg.get("seed", 0),
                                rank=rank, world_size=world_size)
    val_ds = ConsensusParquet(val_paths, preprocess=pp, seed=tcfg.get("seed", 0) + 1)
    print(f"[data] train row-groups: {len(train_ds._units)}  val row-groups: {len(val_ds._units)}",
          flush=True)

    def collate(batch):
        batch = [b for b in batch if b[0].numel() > 0]
        if not batch:
            return None
        return contrastive_collate(batch, aug_cfg, mode=mode, mask_cfg=mask_cfg)

    train_loader = DataLoader(
        train_ds, batch_size=tcfg["batch_size"], num_workers=tcfg["num_workers"],
        collate_fn=collate, pin_memory=True,
        persistent_workers=tcfg["num_workers"] > 0,
    )
    val_loader = DataLoader(
        val_ds, batch_size=tcfg["batch_size"],
        num_workers=max(1, tcfg["num_workers"] // 2) if tcfg["num_workers"] > 0 else 0,
        collate_fn=collate, pin_memory=True,
        persistent_workers=tcfg["num_workers"] > 0,
    )
    return train_loader, val_loader


def run_validation(module, val_loader, device, autocast_ctx, max_batches: int) -> dict[str, float]:
    module.eval()
    sums = {"loss": 0.0, "contrastive": 0.0, "kl": 0.0, "pos_frac": 0.0}
    n = 0
    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            if batch is None:
                continue
            if i >= max_batches:
                break
            batch = to_device(batch, device)
            with autocast_ctx:
                loss, parts = module(batch)
            sums["loss"] += float(loss)
            sums["contrastive"] += float(parts["contrastive"])
            sums["kl"] += float(parts["kl"])
            sums["pos_frac"] += float(parts["pos_frac"])
            n += 1
    module.train()
    return {f"val/{k}": v / max(1, n) for k, v in sums.items()}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--resume", type=Path, default=None,
                        help="resume a contrastive run (encoder+proj+optim+step)")
    parser.add_argument("--pretrained-ckpt", type=Path, default=None,
                        help="override train.pretrained_ckpt (cold-init encoder only)")
    parser.add_argument("--run-name", type=str, default=None)
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    if args.run_name:
        cfg["log"]["wandb_run_name"] = args.run_name

    tcfg = cfg["train"]
    lcfg = cfg["log"]
    ccfg = cfg["contrastive"]

    rank, world_size, local_rank, is_dist = setup_distributed()
    is_main = rank == 0
    device = torch.device(f"cuda:{local_rank}") if is_dist else torch.device(tcfg["device"])

    def log0(*a, **k):
        if is_main:
            print(*a, **k)

    seed = int(tcfg.get("seed", 0))
    import random as _random
    _random.seed(seed + rank)
    np.random.seed(seed + rank)
    torch.manual_seed(seed + rank)
    torch.cuda.manual_seed_all(seed + rank)
    log0(f"[seed] {seed} (+rank) world_size={world_size}", flush=True)

    run_name = lcfg["wandb_run_name"] or time.strftime("%Y%m%d-%H%M%S")
    out_dir = Path(lcfg["out_dir"]) / run_name
    if is_main:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "figs").mkdir(exist_ok=True)
    log0(f"[run] {run_name} → {out_dir}", flush=True)

    if is_main:
        wandb.init(project=lcfg["wandb_project"], name=run_name, config=cfg, dir=str(out_dir))

    # ---- model: encoder + projection head (+ optional aux-KL intensity head) ----
    lambda_kl = float(ccfg.get("lambda_kl", 0.0))
    model_cfg = build_model_config(cfg["model"])
    encoder = MSEncoder(model_cfg).to(device)
    proj_head = ProjectionHead(
        in_dim=2 * model_cfg.d_model,
        hidden=int(ccfg.get("proj_hidden", 2 * model_cfg.d_model)),
        out_dim=int(ccfg.get("proj_dim", 256)),
    ).to(device)
    intensity_head = IntensityHead(model_cfg.d_model).to(device) if lambda_kl > 0 else None
    supcon = SupConLoss(temperature=float(ccfg.get("temperature", 0.07)))

    # Cold-init the encoder from the masked-pretrain checkpoint (encoder only —
    # NOT optimizer/scheduler/step). --resume (below) takes precedence for
    # continuing a contrastive run.
    pretrained = args.pretrained_ckpt or tcfg.get("pretrained_ckpt")
    if pretrained and not args.resume:
        pck = torch.load(pretrained, map_location=device)
        encoder.load_state_dict(pck["encoder"])
        log0(f"[init] encoder ← pretrained {pretrained}", flush=True)

    n_params = sum(p.numel() for p in encoder.parameters()) + sum(p.numel() for p in proj_head.parameters())
    log0(f"[model] {n_params/1e6:.2f}M trainable params (encoder+proj)"
         f"{' +KL head' if intensity_head is not None else ''}", flush=True)

    params = list(encoder.parameters()) + list(proj_head.parameters())
    if intensity_head is not None:
        params += list(intensity_head.parameters())
    optimizer = AdamW(params, lr=tcfg["lr"], weight_decay=tcfg["weight_decay"], betas=(0.9, 0.95))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=lambda s: lr_lambda(s, tcfg["warmup_steps"], tcfg["total_steps"]))

    # ---- data ----
    aug_cfg = AugmentConfig(**ccfg.get("augmentation", {}))
    mode = ccfg.get("mode", "supcon")
    mask_cfg = MaskConfig(mask_ratio=cfg["data"]["mask"]["mask_ratio"]) if lambda_kl > 0 else None
    train_paths, val_paths = resolve_dataset_paths(cfg["data"])
    train_loader, val_loader = make_loaders(
        cfg, train_paths, val_paths, aug_cfg, mode, mask_cfg, rank=rank, world_size=world_size)

    probe_val_paths = val_paths
    probe_pp = PreprocessConfig(
        intensity_threshold_frac=cfg["data"]["intensity_threshold_frac"],
        top_n=cfg["data"]["top_n"],
    )

    if tcfg["precision"] == "bf16":
        autocast_ctx = torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16)
    elif tcfg["precision"] == "fp16":
        autocast_ctx = torch.amp.autocast(device_type=device.type, dtype=torch.float16)
    else:
        autocast_ctx = nullcontext()

    start_step = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        encoder.load_state_dict(ckpt["encoder"])
        proj_head.load_state_dict(ckpt["proj_head"])
        if intensity_head is not None and ckpt.get("heads") is not None:
            intensity_head.load_state_dict(ckpt["heads"])
        optimizer.load_state_dict(ckpt["optimizer"])
        if ckpt.get("scheduler") is not None:
            scheduler.load_state_dict(ckpt["scheduler"])
        start_step = int(ckpt["step"]) + 1
        log0(f"[resume] step {start_step} from {args.resume}", flush=True)

    module: nn.Module = ContrastiveModule(encoder, proj_head, supcon, intensity_head, lambda_kl)
    if is_dist:
        module = DDP(module, device_ids=[local_rank], output_device=local_rank)
    if tcfg.get("compile", False):
        module = torch.compile(module)
        log0("[compile] torch.compile enabled", flush=True)

    module.train()
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

        save_attn_this_step = (step % lcfg["bias_curve_every"] == 0)
        encoder.set_save_attn(save_attn_this_step)

        with autocast_ctx:
            loss, parts = module(batch)

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

        if is_main and step % lcfg["log_every"] == 0:
            lr = optimizer.param_groups[0]["lr"]
            wandb_log = {
                "train/loss": running_loss / max(1, running_n),
                "train/contrastive": float(parts["contrastive"]),
                "train/kl": float(parts["kl"]),
                "train/pos_frac": float(parts["pos_frac"]),
                "train/avg_pos": float(parts["avg_pos"]),
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
            print(f"step {step:>6} loss {wandb_log['train/loss']:.4f} "
                  f"con {wandb_log['train/contrastive']:.4f} kl {wandb_log['train/kl']:.4f} "
                  f"pos_frac {wandb_log['train/pos_frac']:.3f} avg_pos {wandb_log['train/avg_pos']:.2f} "
                  f"|g| {gn_total:.3f} lr {lr:.2e}", flush=True)
            running_loss = 0.0
            running_n = 0
            t0 = time.time()

        if is_main and step > 0 and step % lcfg["val_every"] == 0:
            val_metrics = run_validation(module, val_loader, device, autocast_ctx,
                                         max_batches=tcfg["val_batches"])
            wandb.log({**val_metrics, "step": step}, step=step)
            print("  val: " + "  ".join(f"{k}={v:.4f}" for k, v in val_metrics.items()), flush=True)

        probe_every = lcfg.get("probe_every", 0)
        if is_main and probe_every and step > 0 and step % probe_every == 0:
            if device.type == "cuda":
                torch.cuda.empty_cache()
            probe_metrics = run_all_probes(encoder, probe_val_paths, device, probe_pp,
                                           n_spectra=lcfg.get("probe_n_spectra", 3000))
            align = alignment_metrics(encoder)
            retr = retrieval_inline_metrics(encoder, probe_val_paths, device, probe_pp)
            rr = replicate_retrieval_inline_metrics(
                encoder, lcfg.get("replicate_retrieval_repo"), device, probe_pp)
            wandb.log({**probe_metrics, **align, **retr, **rr, "step": step}, step=step)
            if rr:
                print(f"  replicate-retrieval: "
                      f"Hit@1={rr.get('replicate_retrieval/Hit@1', float('nan')):.3f} "
                      f"MAP={rr.get('replicate_retrieval/MAP', float('nan')):.3f} "
                      f"PairF1={rr.get('replicate_retrieval/PairF1', float('nan')):.3f} | "
                      f"whiten Hit@1={rr.get('replicate_retrieval/Hit@1_w', float('nan')):.3f} "
                      f"MAP={rr.get('replicate_retrieval/MAP_w', float('nan')):.3f} "
                      f"PairF1={rr.get('replicate_retrieval/PairF1_w', float('nan')):.3f}", flush=True)

        if is_main and step % lcfg["bias_curve_every"] == 0:
            panels = render_bias_panels(encoder.bias_module, step)
            wandb_imgs = {}
            import matplotlib.pyplot as plt
            for name, fig in panels.items():
                fig_path = out_dir / "figs" / f"{name.replace('/', '_')}_step{step:06d}.png"
                fig.savefig(fig_path, dpi=110)
                wandb_imgs[name] = wandb.Image(str(fig_path))
                plt.close(fig)
            wandb.log({**wandb_imgs, "step": step}, step=step)

        if is_main and step > 0 and step % lcfg["ckpt_every"] == 0:
            ckpt_path = out_dir / f"step{step:06d}.pt"
            save_ckpt(ckpt_path, encoder, proj_head, intensity_head, optimizer, scheduler, step, cfg)
            save_ckpt(out_dir / "last.pt", encoder, proj_head, intensity_head, optimizer, scheduler, step, cfg)
            print(f"  ckpt → {ckpt_path}", flush=True)

        step += 1

    if is_main:
        save_ckpt(out_dir / "final.pt", encoder, proj_head, intensity_head, optimizer, scheduler, step, cfg)
        wandb.finish()
    if is_dist:
        dist.barrier()
        dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    sys.exit(main())
