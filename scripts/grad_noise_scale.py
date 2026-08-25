"""Gradient noise scale — a principled way to choose the effective batch size.

Estimates the *simple noise scale* ``B_simple = tr(Σ) / |G|²`` from McCandlish,
Kaplan & McCandlish, *An Empirical Model of Large-Batch Training* (2018), where
``G`` is the true gradient and ``Σ`` the per-example gradient covariance.

Interpretation:
  * effective batch ≪ B_simple  → doubling batch ≈ halves steps-to-target
                                   (you're noise-limited; bigger batch pays off)
  * effective batch ≫ B_simple  → diminishing returns (compute wasted per step)
  * the compute-efficient batch sits around B_simple.

B_simple *grows* as the loss falls, so read it where you actually care — near the
plateau you're stuck on (fresh init already sits there for the masked-intensity
task; use --ckpt to measure from a checkpoint further in).

Estimator (no covariance matrix needed): split a big batch into ``n-micro``
micro-batches of size ``micro-batch``. From the mean per-micro-batch squared
gradient norm (|G|² at the small batch) and the squared norm of the *averaged*
gradient (|G|² at the big batch), solve the two-point system for tr(Σ) and |G|².
Numerator and denominator are averaged over ``iters`` outer draws *separately*
before the ratio (the ratio-of-noisy-estimates is biased otherwise).

Reusable across configs — everything (model, preprocessing, mask ratio, data
source) is read from the training YAML:

    python scripts/grad_noise_scale.py --config configs/massivekb_xl_debug.yaml
    python scripts/grad_noise_scale.py --config configs/massivekb_xl.yaml \
        --micro-batch 16 --n-micro 32 --iters 8 --precision bf16 --ckpt runs/foo/final

Tunables: raise --n-micro (bigger probe batch = less extrapolation) and --iters
(more averaging = tighter estimate) when you have the GPU to yourself.
"""
from __future__ import annotations

import argparse
from contextlib import nullcontext
from pathlib import Path

import torch

from msdelta.config import parse_config
from msdelta.data import (
    MaskIntensityCollator,
    build_preprocessed_dataset,
    resolve_dataset_paths,
)
from msdelta.model import MSDeltaForPretraining


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True, type=Path, help="training YAML")
    ap.add_argument("--micro-batch", type=int, default=16, help="B_small")
    ap.add_argument("--n-micro", type=int, default=32,
                    help="micro-batches per draw; B_big = micro_batch * n_micro")
    ap.add_argument("--iters", type=int, default=8, help="outer draws to average over")
    ap.add_argument("--split", choices=["val", "train"], default="val")
    ap.add_argument("--n-shards", type=int, default=2, help="parquet shards to sample from")
    ap.add_argument("--ckpt", type=Path, default=None,
                    help="optional encoder+heads checkpoint to measure at (else fresh init)")
    ap.add_argument("--precision", choices=["fp32", "bf16"], default="fp32",
                    help="fp32 = clean intrinsic estimate; bf16 = match a bf16 run")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    _, margs, dargs, targs, _ = parse_config(["--config", str(args.config)])

    # ---- data: a bounded slice through the real preprocessing ----
    train_paths, val_paths = resolve_dataset_paths(dargs.to_source_dict())
    paths = (val_paths if args.split == "val" else train_paths)[: args.n_shards]
    ds = build_preprocessed_dataset(paths, dargs.preprocess(), num_proc=4)
    collate = MaskIntensityCollator(mask_ratio=dargs.mask_ratio)
    print(f"[data] {len(ds)} spectra from {len(paths)} {args.split} shard(s); "
          f"mask_ratio={dargs.mask_ratio}", flush=True)

    # ---- model: exactly what this config trains ----
    dev = args.device
    model = MSDeltaForPretraining(margs.to_model_config()).to(dev)
    if args.ckpt:
        ckpt = torch.load(args.ckpt, map_location=dev)
        model.encoder.load_state_dict(ckpt["encoder"])
        model.heads.load_state_dict(ckpt["heads"])
        print(f"[model] loaded {args.ckpt}", flush=True)
    model.train()  # match training conditions (dropout is part of the noise)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[model] {n_params / 1e6:.1f}M params  precision={args.precision}", flush=True)

    params = [p for p in model.parameters() if p.requires_grad]
    autocast = (torch.autocast(dev.split(":")[0], dtype=torch.bfloat16)
                if args.precision == "bf16" else nullcontext())

    g = torch.Generator().manual_seed(args.seed)

    def micro_grad_sq_and_add(accum: list[torch.Tensor]) -> float:
        """One micro-batch backward; add grads into `accum`, return |g|²."""
        idx = torch.randint(0, len(ds), (args.micro_batch,), generator=g).tolist()
        batch = {k: v.to(dev) for k, v in collate([ds[i] for i in idx]).items()}
        model.zero_grad(set_to_none=True)
        with autocast:
            loss = model(**batch)["loss"]
        loss.backward()
        sq = 0.0
        with torch.no_grad():
            for acc, p in zip(accum, params):
                if p.grad is not None:
                    acc += p.grad
                    sq += float(p.grad.pow(2).sum())
        return sq

    b, n = args.micro_batch, args.n_micro
    B_small, B_big = b, b * n
    G2_sum, S_sum = 0.0, 0.0
    print(f"[probe] B_small={B_small}  B_big={B_big}  iters={args.iters}\n", flush=True)
    for it in range(args.iters):
        accum = [torch.zeros_like(p) for p in params]
        sum_sq = sum(micro_grad_sq_and_add(accum) for _ in range(n))
        g_small_sq = sum_sq / n                                    # E‖g_b‖²
        g_big_sq = sum(float((a / n).pow(2).sum()) for a in accum)  # ‖mean g‖²
        # Two-point solve for the noise-free quantities.
        G2 = (B_big * g_big_sq - B_small * g_small_sq) / (B_big - B_small)  # ‖G‖²
        S = (g_small_sq - g_big_sq) / (1.0 / B_small - 1.0 / B_big)          # tr(Σ)
        G2_sum += G2
        S_sum += S
        running = S_sum / G2_sum if G2_sum > 0 else float("nan")
        print(f"  iter {it:2d}: |G|²={G2:.3e}  tr(Σ)={S:.3e}  "
              f"running B_simple={running:.0f}", flush=True)

    B_simple = S_sum / G2_sum if G2_sum > 0 else float("nan")
    print(f"\n=== B_simple ≈ {B_simple:.0f} (examples) ===")
    print("Guidance (this is a plateau-region estimate; it grows later in training):")
    for cand in (targs.batch_size, 320, 512, 1024):
        verdict = ("noise-limited — larger batch would still help"
                   if cand < 0.5 * B_simple else
                   "diminishing returns — mostly wasted compute"
                   if cand > 2 * B_simple else "near the efficient sweet spot")
        print(f"  eff_batch {cand:5d}: {verdict}")
    print("Remember: batch and LR are coupled — scale LR (≈√ for Adam) if you change batch.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
