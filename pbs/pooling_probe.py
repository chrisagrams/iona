"""Which pooling extracts the most peptide structure from a FROZEN encoder?

    python pbs/pooling_probe.py --pretrained PATH [--denoiser PATH]

No training. The encoder is whatever it already is; only the reduction from per-peak
token embeddings to one vector changes. That makes this the cheapest experiment
available -- and it asks a question the whole contrastive effort assumed away.

The current default takes an unweighted mean over every peak. A mass spectrum is mostly
noise (53% by count in the denoise corpus), so that mean is dominated by peaks carrying
no identity, and the concatenated max is taken over those same dimensions. Two weightings
are worth a look: raw intensity, which is free and where the identifying fragments
usually are, and a denoiser's P(signal), which is the same idea learned rather than
assumed -- and we have one at 0.932 test AUROC.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch import Tensor

from msdelta.reranking import (AlignmentCollator, group_separation_metrics, peptide_key,
                               pool_sequence)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pretrained", required=True)
    parser.add_argument("--denoiser", default="", help="MSDeltaForDenoising checkpoint")
    parser.add_argument("--cache", required=True, help="precompute_align cache")
    parser.add_argument("--split", default="validation")
    parser.add_argument("--max_rows", type=int, default=1200)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--random_init", action="store_true",
                        help="same architecture, NO pretrained weights. The control the "
                             "trajectory sweep needs: replicate spectra of one peptide "
                             "have similar peaks, so an untrained encoder that merely "
                             "passes its input through already separates them somewhat. "
                             "Without this number, a high ratio early in pretraining "
                             "cannot be told apart from not having learned anything yet.")
    parser.add_argument("--layers", action="store_true",
                        help="also sweep every encoder layer, not just the output")
    cli = parser.parse_args()

    device = "xpu" if torch.xpu.is_available() else "cpu"
    from datasets import load_from_disk
    from msdelta.modeling_msdelta import MSDeltaForPreTraining
    rows = list(load_from_disk(str(Path(cli.cache) / cli.split)))[: cli.max_rows]
    if cli.random_init:
        # from_config, not from_pretrained-then-reinitialise: loading first would leave
        # any buffer the init does not touch still carrying pretrained values.
        from msdelta.configuration_msdelta import MSDeltaConfig
        model = MSDeltaForPreTraining(
            MSDeltaConfig.from_pretrained(cli.pretrained)).to(device).eval()
        print(f"[pool] RANDOM INIT control: architecture of {cli.pretrained}, no weights",
              flush=True)
    else:
        model = MSDeltaForPreTraining.from_pretrained(cli.pretrained).to(device).eval()
    encoder = getattr(model, "msdelta", model)

    denoiser = None
    if cli.denoiser:
        from msdelta.modeling_msdelta import MSDeltaForDenoising
        denoiser = MSDeltaForDenoising.from_pretrained(cli.denoiser).to(device).eval()
        print(f"[pool] denoiser from {cli.denoiser}", flush=True)

    # Capture every block's output with forward hooks rather than modifying the model.
    # The last layer of a masked-prediction encoder is specialised for predicting peak
    # intensities, which is not the same thing as representing peptide identity, and in
    # BERT-like models the most transferable representation usually sits in the middle.
    # Nothing here has ever looked anywhere but the final normalised output.
    layer_outputs: dict[int, Tensor] = {}
    handles = []
    if cli.layers:
        def capture(index):
            def hook(_module, _inputs, output):
                layer_outputs[index] = output
            return hook
        for index, block in enumerate(encoder.blocks):
            handles.append(block.register_forward_hook(capture(index)))
        print(f"[pool] hooked {len(handles)} encoder layers", flush=True)

    # Separately captured, and NOT added to layer_outputs, so the per-layer table keeps
    # its existing numbering (layerNN == block NN) and stays comparable with job
    # 8841973. It exists only so the uniform-mixture baseline can average the same
    # eleven things LayerMixPooler does -- ten blocks plus the pre-block embedding.
    embed_output: dict[int, Tensor] = {}
    if cli.layers:
        handles.append(encoder.embed.register_forward_hook(
            lambda _m, _i, out: embed_output.__setitem__(0, out)))

    collator = AlignmentCollator()
    modes = ["mean", "mean+max", "weighted_mean", "weighted_mean+max"]
    pooled: dict[str, list] = {m: [] for m in modes}
    pooled |= {f"{m}/denoised": [] for m in ("weighted_mean", "weighted_mean+max")
               if denoiser is not None}

    with torch.no_grad():
        for start in range(0, len(rows), cli.batch_size):
            chunk = rows[start : start + cli.batch_size]
            # Strip `target` before collating. AlignmentCollator drops the spectrum
            # columns when a cached target is present -- correct for training, where the
            # model reads the target and never the spectrum, and exactly wrong here,
            # where the spectrum is the only thing we want.
            spectra = [{k: v for k, v in row.items() if k != "target"} for row in chunk]
            batch = {k: v.to(device) for k, v in collator(spectra).items()
                     if k in ("mz", "log_intensity", "attention_mask")}
            if not batch:
                raise RuntimeError("collator produced no spectrum columns")
            hidden = encoder(**batch).last_hidden_state
            # Intensity as stored is log1p; undo it so a peak ten times taller counts
            # ten times, not log(10) times.
            intensity = torch.expm1(batch["log_intensity"]).clamp_min(0)
            signal = None
            if denoiser is not None:
                logits = denoiser(**batch).logits
                # The head predicts NOISE as the positive class, so P(signal) is the
                # complement. Getting this backwards would weight by noise and look
                # like a failed idea rather than an inverted one.
                signal = torch.sigmoid(-logits.squeeze(-1).float())
            for mode in modes:
                weights = intensity if mode.startswith("weighted") else None
                pooled[mode].append(
                    pool_sequence(hidden, batch["attention_mask"], mode,
                                  weights=weights).float().cpu())
            if signal is not None:
                for mode in ("weighted_mean", "weighted_mean+max"):
                    pooled[f"{mode}/denoised"].append(
                        pool_sequence(hidden, batch["attention_mask"], mode,
                                      weights=signal).float().cpu())
            # The UNTRAINED uniform mixture: exactly what LayerMixPooler computes at
            # initialisation, before any of its weights have moved. Without it the
            # trained frozen arm's 1.49 cannot be read -- a trained readout that lands
            # where the untrained one already was has achieved nothing, and there was
            # no way to tell those apart.
            if cli.layers and layer_outputs:
                depths = ([embed_output[0]] if embed_output else []) \
                    + [layer_outputs[i] for i in sorted(layer_outputs)]
                # LayerNorm each depth with no affine parameters, matching the pooler:
                # raw block outputs differ by an order of magnitude across depth, so an
                # unnormalised average is just the largest layer.
                normed = torch.stack([
                    torch.nn.functional.layer_norm(h, (h.shape[-1],)) for h in depths
                ], dim=0)
                uniform = normed.mean(0)      # softmax(zeros) is exactly uniform
                pooled.setdefault("layer_mix/uniform", []).append(
                    pool_sequence(uniform, batch["attention_mask"], "mean").float().cpu())

            for index, states in layer_outputs.items():
                # `mean` only for the layer sweep: the +max variants were uniformly
                # worse on the output layer and doubling the table would hide the axis
                # being measured.
                key = f"layer{index:02d}/mean"
                pooled.setdefault(key, []).append(
                    pool_sequence(states, batch["attention_mask"], "mean").float().cpu())
            if start % (cli.batch_size * 40) == 0:
                print(f"[pool] {start}/{len(rows)}", flush=True)

    for handle in handles:
        handle.remove()
    groups = np.unique(np.array([peptide_key(r["peptide"], int(r.get("charge") or 0))
                                 for r in rows]), return_inverse=True)[1]
    print(f"\n[pool] {len(rows)} spectra, {len(set(groups.tolist()))} groups, "
          f"frozen encoder, no training\n", flush=True)
    print(f"  {'pooling':<24} {'RATIO':>7} {'in_mean':>8} {'out_mean':>9} {'clean':>7}")
    for name, chunks in pooled.items():
        if not chunks:
            continue
        m = group_separation_metrics(torch.cat(chunks), groups, "s")
        print(f"  {name:<24} {m['s/ratio']:>7.2f} {m['s/in_mean']:>8.4f} "
              f"{m['s/out_mean']:>9.4f} {m['s/clean']:>7.3f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
