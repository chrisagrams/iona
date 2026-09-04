# `msdelta/eval/` — diagnostics

Roughly two thirds of this repo's non-model code. These run **inline during pretraining** (driven
by `train/callbacks.py`) rather than as a separate offline pass, so you watch chemistry emerge as
the loss drops.

| file | question it answers |
| --- | --- |
| `alignment.py` | Do the learned bias peaks land on **real chemistry** more often than chance? |
| `probe.py` | What is linearly decodable from the frozen encoder? |
| `retrieval.py` | Do replicate spectra of the same peptide embed near each other — better than a binned baseline? |
| `denoising.py` | Does the frozen encoder support a downstream signal/noise task? |
| `embedding.py` | Shared plumbing: `pool_tokens`, `encode_batch`, `embed_spectra` |
| `viz.py` | Per-head bias-curve panels with reference masses overlaid |

Every module here reads the encoder; **none of them mutate it.** Each restores `.training` mode on
the way out, and the denoising probe goes considerably further (below).

## `alignment.py` — the load-bearing scientific claim

The model is supposed to learn chemistry as bumps in `B_h(Δ)`. This tests that claim properly
instead of eyeballing plots:

1. Evaluate each head's curve on a dense Δ grid (`bias_module.evaluate`).
2. `scipy.signal.find_peaks` with a prominence threshold.
3. Match each peak to the nearest reference mass (isotope / neutral loss / residue) within `tol`.
4. **Compare against a real null**: `_chance_rate` is the fraction of the Δ axis lying within
   `tol` of *any* reference mass — so a head that spikes everywhere gets no credit.
5. `binomtest(n_hit, n_peaks, chance, alternative="greater")` per head, then Bonferroni across
   heads × ranges.

Two ranges: **fine** `[-5, 5]` Da at 1 mDa against isotopes, **coarse** `[2, 200]` Da at 10 mDa
against neutral losses and residues. `align/n_sig01_bonf` is the headline number.

## `probe.py` — linear probes

One batched pass (`extract_representations`) collects three representation levels and their
targets:

| level | representation | targets |
| --- | --- | --- |
| spectrum | mean ‖ max pooled tokens | precursor m/z, charge, peak count, log TIC |
| peak | a single token | isotope rank (0–3), fragment m/z |
| peak pair | two concatenated tokens | neutral-loss membership |

Ridge / logistic regression on a seeded 70/30 split. **Every metric ships with a baseline** —
majority class for classification, `max(m/z)` for precursor m/z — because "R² = 0.9 on precursor
m/z" means nothing if the largest peak already gets you there. That is precisely the "is the
model cheating?" worry in `TODO.txt`.

## `retrieval.py`

Leave-one-out cosine retrieval over replicate groups, chunked through `torchmetrics`. The number
that matters is `retrieval/gap_vs_binned`: the same metric computed on a **1-Da binned intensity
vector** built in the same pass. A learned embedding that cannot beat binning has not earned its
parameters. `all_but_top(X, k)` applies the standard whitening trick (the external benchmark uses
`k=16`).

## `denoising.py` — a probe that cannot corrupt your run

`run_denoising_probe` wraps the **live** encoder in a frozen `MSDeltaForDenoising`, trains a fresh
head in a nested `Trainer`, and evaluates. Since it runs mid-pretraining, it is careful:

* `torch.random.fork_rng` around everything;
* Python and NumPy RNG state saved and restored in a `finally`;
* every `requires_grad` flag on the encoder recorded and restored;
* `module.train(was_training)` on the way out;
* the encoder is pinned to `.eval()` via an overridden `train()` so a frozen encoder never
  re-enables dropout.

`PeakBudgetBatchSampler` batches by `(batch + 1) × longest²` against `denoise_peak_pair_budget`
rather than a fixed batch size, because at 1024 peaks the padded attention bias — not the token
count — is what OOMs.

## If you change the architecture

Four call sites reach into encoder internals and will break if you move things:

```
encoder.bias_module        → viz.render_bias_panels, alignment._eval_curves,
                             train/callbacks.FourierProbeCallback, train/cli.main
encoder.bias_module.ff     → train/callbacks.FourierProbeCallback
encoder.embed.ff_int       → train/callbacks.FourierProbeCallback
module.msdelta             → callbacks._InlineCallback.encoder, eval/denoising.run_denoising_probe
```

Grep for those four before merging any change to `model/modeling.py`. Making the bias per-layer
(the highest-leverage experiment) touches all of them.
