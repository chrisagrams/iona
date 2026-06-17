"""Throughput A/B for the GB10: compile x batch_size, on the real XL model.

Reports steps/s AND spectra/s (= bs * sps). For a data-exposure-matched run
(fixed ~12.8M spectra), spectra/s is the figure of merit and total_steps must
scale as 12.8M / bs. Uses synthetic batches so we measure pure GPU step time,
isolated from the dataloader.
"""
import time, argparse, torch
from msdelta.train import build_model_config, load_config
from msdelta.model import MSEncoder, IntensityHead


def make_step(enc, heads, opt, B, P, dev, ac, compile_enc):
    if compile_enc:
        enc_fwd = torch.compile(enc)
    else:
        enc_fwd = enc

    def batch():
        mz = torch.rand(B, P, device=dev) * 1500
        li = torch.randn(B, P, device=dev)
        kpm = torch.zeros(B, P, dtype=torch.bool, device=dev)
        mp = torch.rand(B, P, device=dev) < 0.5
        tgt = torch.rand(B, P, device=dev)
        tgt = tgt / tgt.sum(-1, keepdim=True)
        return mz, li, kpm, mp, tgt

    def step():
        mz, li, kpm, mp, tgt = batch()
        with ac:
            tok = enc_fwd(mz, li, kpm, mp)
        loss, _ = heads.loss(tok, tgt, mp)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    return step


def bench(cfg, B, compile_enc, dev, warmup, iters):
    P = cfg["model"]["max_peaks"]
    mc = build_model_config(cfg["model"])
    enc = MSEncoder(mc).to(dev)
    heads = IntensityHead(mc.d_model).to(dev)
    enc.train(); heads.train()
    opt = torch.optim.AdamW(list(enc.parameters()) + list(heads.parameters()), lr=1e-4)
    ac = torch.amp.autocast("cuda", dtype=torch.bfloat16)
    step = make_step(enc, heads, opt, B, P, dev, ac, compile_enc)
    for _ in range(warmup):  # compile warmup happens here
        step()
    torch.cuda.synchronize()
    t = time.time()
    for _ in range(iters):
        step()
    torch.cuda.synchronize()
    dt = time.time() - t
    peak = torch.cuda.max_memory_allocated() / 1e9
    sps = iters / dt
    del enc, heads, opt
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    return sps, sps * B, peak


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/v14_cap_XL.yaml")
    ap.add_argument("--batches", type=int, nargs="+", default=[80, 160, 256])
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--iters", type=int, default=20)
    args = ap.parse_args()

    cfg = load_config(args.config)
    dev = torch.device("cuda")
    A100_XL_SPS = 3.45  # PBS baseline, for reference

    print(f"model={args.config}  P={cfg['model']['max_peaks']}  d={cfg['model']['d_model']}")
    print(f"{'compile':>8} {'bs':>5} {'steps/s':>9} {'spectra/s':>11} {'peak_GB':>9} {'160k-equiv_h':>13}")
    base_spectra_s = None
    for compile_enc in (False, True):
        for B in args.batches:
            try:
                sps, spec_s, peak = bench(cfg, B, compile_enc, dev, args.warmup, args.iters)
            except RuntimeError as e:
                print(f"{str(compile_enc):>8} {B:>5}   OOM/err: {str(e)[:50]}")
                continue
            # walltime for the SAME 12.8M-spectra exposure: 12.8e6 / spectra_s
            equiv_h = 12.8e6 / spec_s / 3600
            if base_spectra_s is None:
                base_spectra_s = spec_s
            print(f"{str(compile_enc):>8} {B:>5} {sps:>9.3f} {spec_s:>11.1f} {peak:>9.2f} {equiv_h:>13.1f}")
    print(f"\nreference: A100 XL @ bs=80 = {A100_XL_SPS} sps = {A100_XL_SPS*80:.0f} spectra/s "
          f"= {12.8e6/(A100_XL_SPS*80)/3600:.1f}h for 12.8M")


if __name__ == "__main__":
    main()
