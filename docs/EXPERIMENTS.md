# msdelta — experiment log

A running record of what we built, what we tried, what broke, and how we know.
Read top-to-bottom — the iteration history is chronological.

---

## 0. Project goal

Train a transformer encoder over peak-tokens with a **learned per-head Δm/z
attention bias**, and verify that heads specialize on chemically meaningful
Δm patterns (isotopes, amino-acid residues, neutral losses) when pretrained
on consensus MS/MS spectra. Everything in this log is in service of getting
to a point where we can read off chemistry from the bias-curve plots.

The plot we want to look at, per head, is `bias_h(Δm)` evaluated on a dense
Δm grid:

- **Fine range** `Δm ∈ [-5, 5] Da` step 1 mDa — for isotopes and small mods.
- **Coarse range** `Δm ∈ [-200, 200] Da` step 10 mDa — for residues and losses.

Reference Δm values overlaid: ±1.003 (¹³C), 17.027 (NH₃), 18.011 (H₂O),
27.995 (CO), 43.990 (CO₂), 57.021 (G) … 186.079 (W), 162.053 (hexose), …

---

## 1. Architecture (current)

Source of truth: `msdelta/`. Key components:

- **`fourier.py`** — log-spaced sin/cos features, clamp-guarded.
- **`data.py`** — parquet streaming dataset, per-spectrum preprocessing
  (intensity-floor + top-N + log1p+norm), **denoising collate** (Gaussian
  m/z noise).
- **`model.py`** — `PeakEmbed` (Fourier(m/z) ⊕ Fourier(log_int) → MLP →
  d_model), `DeltaMZBias` (per-head independent MLPs from `Fourier(Δm)`),
  `BiasedMHA` (custom multi-head attention that adds per-head bias to
  logits), `EncoderBlock` (pre-norm), `MSEncoder` (shared bias across
  layers), `DenoiseHead` (per-peak residual + log_var, Gaussian NLL on
  the cleaning residual).
- **`train.py`** — AdamW + warmup→cosine, bf16 autocast, wandb,
  periodic bias-curve PNGs.
- **`viz.py`** — bias-curve plotting with chemistry reference lines.

Current pretraining task: **m/z denoising autoencoder** with Gaussian σ =
0.3 Da. Predict the clean m/z given the noisy m/z + clean intensities.

---

## 2. Data infrastructure

### What broke: MSZX dataloader memory explosion

The original plan used the mscompress MSZX dataloader pointed at
`/mnt/vault-1/k8/concensus_spec_30M/consensus_100M/mszx/` (~100M consensus
spectra in 90 .mszx shards).

**Symptom:** memory ballooned to ~500 GB during a real run (8 workers ×
~60 GB/worker). One bug we fixed cheaply, two we didn't:

1. *Fixed:* `worker_init_fn` was re-opening all 88 shards in every worker
   instead of inheriting via fork. Removed.
2. *Fixed:* `__init__` was eagerly opening all 88 shards. Now reads only
   the manifest.json from each tar for length info.
3. *Couldn't fix from msdelta-side:* mscompress's C decoders allocate
   per-process state that grows with the number of distinct shards
   touched. At RandomSampler scale, every worker eventually touches
   every shard.

**Decision:** switched to the parquet files in the parent directory.

### Switch to Parquet `IterableDataset`

`ConsensusParquet(IterableDataset)` streams `(file, row_group)` units —
one ~263 MB row-group held in arrow per worker, shuffled within and
across. No random per-spectrum lookups across all shards.

**Memory after switch (50 batches, batch_size=64):**

| num_workers | system Δ |
|---|---|
| 0 | 1.14 GB |
| 2 | 2.14 GB |
| 4 | 4.41 GB |
| 8 | **9.58 GB** |

~12× reduction vs MSZX, bounded by row-group footprint. Investigation
of the MSZX leak is parked.

---

## 3. Task iterations — what didn't work

This is the heart of the log. Each version solved the previous failure
and exposed a new one.

### v1 — MPM with m/z zeroed at masked positions

**The task.** Mask 15 % of peaks. For masked peaks, set `m/z = 0` and
`log_int = 0` in the input; swap the embedding for a learned `[MASK]`
token; predict the original `m/z` (Gaussian NLL) and `log_int` (MSE).

**Observation at step 10k.** A large *positive* spike at Δm = 0 on every
head, and nothing else.

![v1: spike at Δm=0 from masked m/z=0](figures/v1_mpm_zeroed_mz_fine_10k.png)
![v1: coarse view, same artifact](figures/v1_mpm_zeroed_mz_coarse_10k.png)

**Root cause.** Zeroing the m/z at masked positions doesn't just hide it
from the embedding — the Δm-bias module sees the same zeros and computes
fake `Δm = 0 − 0 = 0` pairs for every pair of masked positions, every
pair of padded positions, and every masked-padded pair:

| pair type | Δm computed | count per spectrum (K=150, ~22 masked) |
|---|---|---|
| diagonal `(i, i)` | `m_i − m_i = 0` | 150 |
| masked × masked (off-diag) | `0 − 0 = 0` | 22 × 21 ≈ 462 |
| padded × padded, padded × masked | `0 − 0 = 0` | depends on batch |
| real × real | actual Δm | the rest |

The bias module learned `bias(0) ≫ 0` because it's a free, perfectly
reliable signal for "these two are both mask tokens, lump them
together" — pure bookkeeping, no chemistry.

### v2 — keep real m/z at masked positions; zero the bias for masked-touching pairs

**The fix.** Stop zeroing `m/z` in the collate (keep real values, the
`[MASK]` embedding swap still prevents direct leakage via the token).
In `MSEncoder.forward`, multiply the bias by a mask that's zero on any
pair where either endpoint is masked or padded. Together these:

- Remove the fake `Δm=0` cluster from the bias path.
- Prevent the masked m/z from leaking back through `m_masked = m_key +
  Δm` (a real concern: if a head learns spike at Δm = +113, the model
  could recover masked m/z from key m/z).

**Observation at step 22k.** The positive spike is gone — replaced by
a *negative* dip centered at Δm=0, ~3 Da wide, on every head, growing
deeper as training proceeds.

![v2: smooth negative trough at Δm=0](figures/v2_suppress_masked_pairs_fine_22k.png)
![v2: coarse view, dip is the only feature](figures/v2_suppress_masked_pairs_coarse_22k.png)

**Diagnosis.** The bias module discovered that **self-attention is
near-useless for MPM** (the `[MASK]` token has no info on its own; it
must look at neighbors). The bias path is a much cheaper lever for
expressing "don't attend to yourself" than tuning Q/K to make `q_i · k_i`
small. So the model poured most of its bias gradient into a single
broad bump at Δm=0 — global self-attention suppression, no chemistry.

The dip was *growing*, not shrinking, over training: −0.4 at step 4k →
−4 at step 22k. At −4 a softmax logit reduces attention mass by exp(−4)
≈ 50×.

### v3 — also zero the diagonal of the bias

**The fix.** In addition to v2, set `bias[:, :, i, i] = 0`. Now the
bias module *cannot* modulate self-attention at all. The model must
suppress self-attention through Q/K content — slower to learn, but the
right place for that decision to live.

**Observation at step 50k.** The dip got smoother and wider, but didn't
go away. 7 of 8 heads still showed the same shape.

![v3: dip persisted despite diagonal-zero](figures/v3_diagonal_zero_fine_50k.png)
![v3: coarse view](figures/v3_diagonal_zero_coarse_50k.png)

**Diagnosis.** Sampling the actual data revealed that ~14 % of all
consecutive-peak Δm pairs in real spectra are < 0.5 Da. The dataset
itself contains a *flood* of near-zero non-diagonal pairs (near-duplicate
peaks, charge variants, ions with tiny m/z differences). The model
correctly learns that these pairs are uninformative for MPM and
suppresses them. Removing the diagonal entry alone doesn't help —
the surrounding off-diagonal small-Δm pairs carry the same signal.

The bias module's capacity was being soaked up by this high-volume
"deduplicate small Δm" task. Of 8 heads, one (head 5) happened to
specialize on the ¹³C isotope spike — random init lottery.

### v4 — independent per-head bias MLPs

**The fix.** Replaced the shared `Linear → GELU → Linear` MLP with **8
independent per-head MLPs** (stacked as `(H, in_dim, hidden)` Parameter
tensors, applied with einsum for batched compute). Decoupling the
gradient pools so each head can drift into its own basin instead of all
8 competing for the same hidden representations.

`DeltaBiasConfig.hidden` → `per_head_hidden`, default 32 (to keep
activation memory bounded; the activation cost is `(B, K, K, H *
per_head_hidden)` which is ~6 GB at production scale).

**Observation at step 48k.** Bias curves DID become more diverse than
v3 — heads no longer all converged to the same shape. But still no
clean chemistry, and the ±1.003 spike that head 5 had in v3 was *gone*.

![v4: more diverse but still mostly dips](figures/v4_per_head_mlp_fine_48k.png)
![v4: coarse view, no residue peaks](figures/v4_per_head_mlp_coarse_48k.png)

**Then I checked the actual prediction quality.** This is where the
real problem surfaced:

```
val/nll_mz       = 7.06         ← "looks like it's learning" (was 13.3 at step 0)
mean_log_var     = +10.00       ← clamped at the upper bound, for 100% of predictions
val/mse_int      = 0.006        ← intensity head, basically untrained

Inferred over 45,804 val targets:
   model RMS m/z error    = 307.0 Da
   predict-spectrum-mean  = 307.2 Da   ← identical to the no-model baseline
   prediction range       = 260 — 1020 Da (target range 56 — 2180 Da)
   prediction std         = 129 Da     (target std 332 Da)

grad_norm_delta_bias / grad_norm_total = 0.007 / 0.486 = 1.4 %
```

**Root cause: the MPM task is fundamentally ambiguous in our setup.**
All 22 masked positions in a spectrum get the same `[MASK]` embedding,
the bias is suppressed for masked queries (so no Δm-derived position
hint), and there's no positional encoding (peaks are a set). From the
model's perspective, every masked query is *literally identical*: "I'm
some peak in this spectrum, predict my m/z." The loss-minimizing answer
is: predict the spectrum's mean, crank the variance up to absorb the
error. That's what it found.

The bias module gets ~no useful gradient (1.4 % of total) because the
task doesn't reward chemistry. The "more diverse heads" in v4 were just
heads drifting under noise pressure, not specializing.

**Conclusion: the v3 fixes were all correct in isolation, but they were
treating symptoms of a deeper problem — the wrong task.**

### v5 — m/z denoising autoencoder

**The fix.** Switch the objective:

- **Don't mask.** Every peak's identity (its m/z) is preserved.
- **Add Gaussian noise** (σ = 0.3 Da) to every peak's m/z.
- **Predict the clean m/z** per peak — head outputs `(residual, log_var)`,
  predicted clean = `noisy + residual`, loss = Gaussian NLL on the
  residual against the true delta.

Why this works in principle:

- Every peak has a unique identifier (its noisy m/z). Task is
  well-defined per position.
- Knowing neighbors' m/z values and the typical Δm structure (isotopes,
  residues) *directly helps* denoising — that's the model's lever.
- The bias path is no longer suppressed for predicted positions, so it
  can carry chemistry signal.

The collate is `DenoiseConfig(gauss_sigma=0.3)` and the head is
`DenoiseHead`. Initial smoke test on a tiny model:

```
init RMSE (= noise σ)   = 0.31 Da
after 100 toy steps     = 0.38 → 0.41  (toy is too small to show real denoising)
```

**Deliberate non-choice.** I almost added a 15% chance of ±1.003 Da
"isotope swap" to the noise model. We discussed and dropped it: it would
have *pre-loaded* the answer for ¹³C and made any ±1.003 spike in the
bias plots uninterpretable as discovery. The cleanest experiment is
Gaussian-only — *anything* the bias module learns is then attributable
to real chemistry in the spectra, not to an injected signal.

**Observation at step 6k (first real run).** The denoising *works* and
the bias module is *finally getting gradient* — but a new failure mode
appeared:

```
step    rmse    nll      |g_total|   |g_bias|
0       0.299   -0.67     2.3         0.000
1500    0.275   -1.00     1.5         0.007
3000    0.267   -1.11     5.1         0.059
4500    0.260   -1.17     5.6         0.047
6000    0.247   -1.34    16.7         0.069
6750    0.246   -1.29    61.7         0.234   ← total grad norm exploding
```

- RMSE 0.30 → 0.245 — real denoising, but modest.
- `|g_bias|` 0.004 → 0.234 — **the bias module is being actively shaped now**
  (vs 1.4 % vestigial in MPM). The per-head MLPs + well-defined task worked.
- `|g_total|` spiking to 60+ — training landscape gone pathological
  (saved only by `grad_clip=1.0`).

The bias curves show a **giant positive bump at small Δm**, +6 logits on
heads 0/1/2 (`exp(6) ≈ 400×` attention):

![v5: bias-dominates-content bump at step 6k](figures/v5_denoise_bias_bump_6k.png)

**Diagnosis: "bias dominates content" — item 4 in the high-level plan's
failure-mode list (§6.3),** which says verbatim: *"all attention goes to
nearest-mass peaks regardless of intensity. Add a scale factor on the
bias output or layer-norm it."*

Why the sign flipped vs MPM (dip → bump): denoising rewards *using*
nearby peaks, so the bias gets positive gradient at small Δm. With no
ceiling there's a positive feedback loop (stronger local bias → more
peaked attention → lower loss → stronger bias) → runaway.

The subtle part: peaks are noised **independently**, so a generic
"attend to whatever's nearby" strategy *can't* actually denoise well —
which is why RMSE stalled at 0.245. The strategy that *would* work is
chemistry-specific (anchor to your ¹³C partner at +1.003, your residue
neighbors at exact masses). The broad bump is a suboptimal local minimum
the unbounded bias let the model over-commit to.

### v6 — bound the bias magnitude (tanh)

**The fix.** Cap the per-head bias to ±`scale` logits in `DeltaMZBias`:

```python
bias = scale * tanh(raw / scale)        # scale = 3.0, fixed (config: delta_bias.scale)
```

±3 logits is comparable to the content term's natural scale
(`q·k/√d ~ O(1–2)`), so neither can steamroll the other. Rationale:
(1) stop the runaway / gradient explosion, and (2) by capping the cheap
broad-bump strategy, create pressure to find the sharper,
chemistry-specific features that actually lower the loss further.

**Observation (full 50k run, σ still 0.3).** The bound worked
mechanically and the run is the best so far — but the chemistry result
is *blurry, not clean*.

```
val/rmse_mz : 0.30 → 0.22   (v5 stalled at 0.245 — bound let it push further)
bias range  : capped at ±3  (no more +6 runaway)
|g_total|   : spiked to 25–32 mid-training (v5 was 60; halved, not fixed)
|g_bias|    : 0.05–0.2, stable, non-vestigial — bias module is being shaped
```

Coarse curves show sharp peaks above a bounded baseline; fine curves
show a broad bump centered near 0:

![v6 coarse: sharp peaks above baseline](figures/v6_bounded_coarse_48k.png)
![v6 fine: broad bump, no isotope spike](figures/v6_bounded_fine_48k.png)

**Quantitative alignment check (plan §6.2b), NOT eyeballing.** With 27
reference Δm values in [2, 200] Da and tolerance 0.15 Da (chance hit-rate
3.9%):

| | strong peaks | aligned <0.15 Da |
|---|---|---|
| all 8 heads | 80 | **2 (0.6× chance — below random)** |

The strong peaks are *real* (heights 1–1.6, well above noise) and
**clustered in the chemically active 17–160 Da band**, but they sit
**0.2–0.8 Da off** the actual masses:

| head | top peak | nearest ref | offset |
|---|---|---|---|
| 1 | 28.14 | CO 27.995 | 0.15 ✓ |
| 1 | 113.81 | N 114.04 | 0.23 ~ |
| 7 | 129.58 | E 129.04 | 0.54 ✗ |
| 7 | 111.35 | L/I 113.08 | 1.7 ✗ |
| 3 | 56.23 | G 57.02 | 0.79 ✗ |

So: *chemistry-adjacent structure in roughly the right region, but not
locked to specific masses.* Not clean discovery.

**Root cause — the noise scale, tying fine and coarse together.**
`σ = 0.3 Da` on m/z → `σ√2 ≈ 0.42 Da` on Δm. Every Δm relationship is
pre-blurred by ±0.42 Da:

- **Residues (57–186 Da):** 0.42 blur is small vs the spacing → peak
  survives but broadens and its center drifts a few tenths of a Da
  (the shifted coarse peaks).
- **Isotopes (1.003 Da):** 0.42 blur is *huge* vs 1.003 → a true ¹³C
  pair smears across `1.003 ± 0.85`, i.e. into the broad ±2 Da bump
  that dominates the fine plot. The isotope signal is present but
  unresolvable.

The fine bump and the coarse drift are the **same blur at two scales**.

**Grad spikes are a separate issue: Gaussian-NLL overconfidence.** Not
the bias (`|g_bias|` is small). The head predicts heteroscedastic
variance (val NLL −1.67 < the −1.0 a fixed-variance Gaussian gives at
RMSE 0.22), so a tiny predicted `log_var` on a hard peak makes
`err²/exp(log_var)` and its gradient blow up. `grad_clip=1.0` caught it.

### v7 — drop noise to σ = 0.1

**The change.** `data.denoise.gauss_sigma: 0.3 → 0.1` (Δm blur drops
from 0.42 to ~0.14 Da). Single highest-value lever: sharpen coarse peaks
onto true masses and shrink the isotope blur so the ±1.003 spike can
resolve. Changed *only* σ to isolate the effect.

**Result (full 50k run).** Denoising got much better; chemistry got
*closer but not convincing*.

```
val/rmse_mz : 0.10 floor → 0.035   (v6 was 0.22 off a 0.30 floor — 59% vs 27% of noise removed)
probe       : Spearman(bias, attention) 0.69–0.98 across heads/ranges → bias LOAD-BEARING
alignment   : FINE  best head6 1.7×, p=0.26  (isotopes still at chance)
              COARSE best head3 2.0×, p=0.064 (closest yet, but NOT significant)
```

![v7 coarse alignment: head 3 (sparse) is the only near-hit](figures/v7_align_coarse_50k.png)
![v7 probe: attention (blue) still tracks bias (red)](figures/v7_probe_coarse_50k.png)

Trend of best coarse p-value across runs: **0.15 (v6) → 0.08 (v7@15k) →
0.031 (v7@35k) → 0.064 (v7 final).** It crossed 0.05 mid-run then drifted
back as the cosine LR decayed to ~0 — i.e. it's hovering at the threshold,
not locking in. With 16 comparisons (8 heads × 2 ranges) the null expects
~0.8 false positives, so one head at p≈0.05 is **not** robust evidence.
Honest verdict: **no convincing chemistry discovery.**

**The structural signal that matters most.** Head 3 — the only head
approaching alignment — has **138 peaks vs 300–400 for the others**. It's
the one head that went *sparse* instead of staying a noisy thicket, and
that sparsity is exactly what let its peaks (L/I 113, V 99, K 128, H₂O 18)
land on real masses. The noisy heads can't align because they have a peak
near everything. This is the empirical motivation for v8.

The fine/isotope range still shows the broad locality bump — even at
σ=0.1's 0.14 Da resolution, the model isn't making a sharp ±1.003 spike,
because the broad "attend within ±1–2 Da" bump already captures the
isotope partner. No pressure to be precise.

### v8 — L1 sparsity penalty on the bias *(implemented; not yet run)*

**The hypothesis, from head 3.** Chemistry emerges when a head goes
sparse. Head 3 did it by accident; force *all* heads sparse and more
should lock onto chemistry. Add `λ · mean|bias_h(Δm)|` (evaluated on a
uniform Δm grid) to the loss. L1 (not L2) because its constant gradient
drives small values to *exactly zero* → a few tall spikes survive, the
broad bump and noise floor get pruned.

This is a **fork test, not a tuning exercise** (coarse 3-point λ probe,
everything else frozen):
- bias sharpens **and** RMSE holds ≈0.035 → chemistry-denoising viable,
  lock in denoise+L1, *then* tune.
- bias sharpens **but** RMSE craters → locality was load-bearing, the
  task is wrong → pivot to a discriminative task (ELECTRA-style).

`l1_lambda: 0.0` reproduces v7 exactly.

**Result (3 runs, 20k steps each: λ = 0.01, 0.1, 1.0).** Neither fork
branch — a *third* outcome we hadn't written down.

| run | mean\|bias\| | RMSE | head outcome | best align p |
|---|---|---|---|---|
| v7 (λ=0) | 1.56 | ~0.039 | 8 noisy heads (~40 pk ea) | 0.064 |
| λ=0.01 | 0.55 | 0.043 | shrunk uniformly, still noisy | n.s. |
| λ=0.1 | 0.035 | 0.047 | **7/8 heads zeroed; 1 survivor still a thicket** | n.s. |
| λ=1.0 | 0.00 | 0.048 | all heads dead (content-only) | n.s. |

The L1 worked mechanically (bias magnitude collapsed with λ) but produced
**head-death, not head-sharpening** — the optimizer zeroed whole heads
rather than concentrating each onto a few chemistry spikes. No setting
produced significant chemistry.

**The decisive insight — an ablation hiding in the sweep.** λ=1.0 is a
clean ablation: bias ≡ 0, pure content attention. Its RMSE (0.048) is only
~0.005 Da (~11%) worse than the near-full-bias λ=0.01 run (0.043). So:

> **The Δm bias is *used* (probe: attention follows it) but nearly
> *dispensable* (ablation: removing it costs ~11% RMSE). Content attention
> reproduces ~90% of what it does.**

That reconciles "load-bearing" (probe) with "head-death under L1": since
the bias barely helps denoising, the optimizer happily sacrifices 7 heads
to satisfy the penalty. There is no gradient pressure to encode chemistry
because **the denoising task can be solved by content attention alone** —
it never *needs* Δm-relational reasoning.

**Verdict — the real fork was a third branch:** *the bias is optional for
this task; no regularizer can force chemistry into a parameter the
objective doesn't require.* Four interventions (per-head MLPs v4, bounded
bias v6, smaller noise v7, sparsity v8) all leave the bias chemistry-free.
v7's head 3 (p=0.064) was a lucky fluctuation, not an amplifiable signal.
→ **Pivot to a task where Δm-relational reasoning is irreducibly necessary
(can't be done peak-by-peak). See v9.**

### Interlude — the probe suite (plan §6.1) flips the narrative

Before pivoting we built `msdelta-probe` (frozen-encoder linear probes) and
ran it on v7's checkpoint. The result reframes everything:

| tier | probe | result | baseline |
|---|---|---|---|
| 1 | precursor m/z | MAE 7.9 Da, R²=0.997 | 520 Da |
| 1 | peak count | R²=0.994 | — |
| 1 | log TIC | R²=0.989 | — |
| 2 | **charge** | **acc 100%** | 59% |
| 2 | **neutral loss** | **AUC 0.96** | 0.49 |
| 2 | **isotope M+k** | **F1 0.86** | — |

**The encoder learned the chemistry — extremely well.** Charge is read
straight from isotope spacing (1/z Da); the encoder gets it 100%. So the
chemistry is *present* — it just lives in the **content path** (Q/K/V over
m/z-bearing tokens), not in the Δm bias. The bias-curve analyses weren't
wrong; the chemistry simply isn't where we were looking.

Root cause, made precise: **tokens carry m/z** (`PeakEmbed` =
`MLP([Fourier(m/z) ⊕ Fourier(int)])`), so `q_i·k_j` can compute any
function of `(m/z_i, m/z_j)`. Content attention is a complete substitute
for a Δm bias — so the bias is never forced to carry chemistry. This is
the "absolute position baked into tokens" regime; relative-position
biases (ALiBi/T5) only become load-bearing when absolute position is
*stripped from the tokens*.

### v9 — m/z-free tokens + masked-intensity prediction *(this branch)*

The T5 mapping for a spectrum: **m/z = position**, **intensity = content**.
T5 strips position from tokens (relative bias carries it) and predicts
*content*. Our v5–v7 denoising predicted m/z = *position* — the one thing
you can't predict once you strip it. The faithful analog predicts
intensity instead:

- **`PeakEmbed` becomes m/z-free:** token = `MLP(Fourier(log_int))` + a
  learned `[MASK]`. Tokens no longer know their own m/z.
- **m/z flows only through `DeltaMZBias`** → the bias is now the *sole*
  carrier of all m/z structure. Forced load-bearing, ALiBi-style.
- **Task = masked-intensity prediction:** mask ~15% of peaks' intensity
  (token → `[MASK]`), predict it. Loss = **MSE on masked positions only**:
  `L = mean_{(b,i)∈mask} (pred_i − logint_i)²`.
- No leak: a masked peak is handed its *position* (m/z, via the bias) and
  asked for *content* (intensity) — never the reverse. Its identity is
  purely relational, exactly the T5 sentinel.

**Why this forces chemistry into the bias.** To predict a masked peak's
intensity the cleanest move is to find its M+0 partner at −1.003 Da and
scale by the isotope ratio — which the model can *only* do by reading a
¹³C feature off the bias (tokens have no m/z). Minimizing this MSE
directly rewards a +1.003 bias spike. The incentive is in the objective,
not hoped for.

**Why MSE not Gaussian-NLL:** intensity is bounded in (0,1]; the NLL
variance head caused the v5–v7 grad-norm explosions. Point estimate is
safer for the fork test.

**The gate (don't repeat the v8 mistake):** before a full run, train this
**with the bias ablated** (content-only). With m/z-free tokens, content
attention has no m/z at all — if content-only *still* solves masked-
intensity, the task doesn't need the bias and we rethink. If content-only
fails and the full model succeeds, the bias is doing the work.

**Risk:** a single scalar/peak is thin signal; intensity may lean on
absolute m/z (now unavailable) more than relational structure, making the
task too hard. The ablation gate + probe suite tell us before we commit.

**Scoreboard:** rerun `msdelta-probe` on the v9 checkpoint. The question
is whether charge/isotope/neutral-loss *still* decode well now that the
bias is forced to carry m/z — and whether the bias curves finally show
significant alignment (`msdelta-analyze`).

**RESULT (full 50k run, `runs/20260524-140946`) — the pivot worked.**
First positive result of the project.

*Transfer (probe suite) — chemistry fully decodable, now necessarily via
the bias:*

| probe | v9 (m/z-free) | note |
|---|---|---|
| precursor m/z | R² 0.996 | reconstructed with **no m/z in tokens** |
| charge | **100%** | charge = isotope spacing → only reachable via the bias |
| neutral loss | AUC 0.95 | |
| isotope M+k | F1 0.87 | |

charge=100% + precursor R²=0.996 with m/z-free tokens proves the Δm bias
is carrying the m/z chemistry — the load-bearing property v4–v8 never had.

*Bias-curve alignment (`msdelta-analyze`) — first significant chemistry head:*

| | enrichment | p | top hits |
|---|---|---|---|
| **coarse head 4** | **2.2×** | **0.002** | M·131, V·99, E·129, L/I·113 (<0.1 Da) |
| **fine head 4** | **2.5×** | **0.011** | ¹³C/z3·0.334, ¹³C/z2·0.501, ¹³C·1.003 |

Head 4 is significant in **both** ranges. Coarse **survives
multiple-comparison correction** (16 tests × 0.002 ≈ 0.032 < 0.05) — vs
v7's best non-surviving p=0.064.

**Measurement fix (charge-aware isotopes).** Initially fine-isotope
alignment looked merely near-significant (p=0.06) — because we scored
only the z=1 spacings {1.003, 2.005}. But ¹³C spacing in *m/z* is
`1.003/z`, and the data is mostly z=2/3, so the real isotope peaks sit at
**0.502 (z=2)** and **0.334 (z=3)** — which the model learned and we were
scoring as misses. Adding the `1.003/z` references (z=1,2,3) to
`viz.ISOTOPES` flipped head 4 fine to p=0.011, with hits landing exactly
at 0.334 / 0.501 / 1.003. The model learned **charge-resolved isotope
spacing**. (Heads 2/3/5/7 also show precise 0.33/0.50/0.67 hits, p≈0.05–0.10.)
Lesson, again: score the right targets — most of the isotope signal was
at Δm we weren't looking at.

![v9 coarse alignment — head 4 residue peaks](figures/v9_align_coarse_50k.png)
![v9 functional probe — attention follows bias](figures/v9_probe_fine_50k.png)

*Functional probe:* Spearman(bias, attention) 0.56–0.91 (fine) — attention
concentrates where the bias peaks. Load-bearing confirmed directly.

**Honest calibration.** It's primarily *one* head (head 4) that clearly
specialized; others are at/near chance in coarse. Fine-isotope alignment
is near- (not past-) significant, though hit precision (±1.003 to the mDa)
is more convincing than the p-value. Coarse functional-probe correlations
are modest (0.13–0.48) — residue-scale bias structure is real but doesn't
dominate attention. So: "a head learned residues + isotopes," not "all 8
did" — but a correction-surviving chemical head is a categorical step up
from eight versions of "vestigial."

**Conclusion.** The original thesis — heads specialize on chemically
meaningful Δm, surfaced in a learned per-head bias — is **demonstrated**
for head 4, and the m/z-free architecture is *why*: stripping m/z from
tokens made the bias the only path for relational chemistry, exactly as
the T5/ALiBi analogy predicted.

**Next steps (decided — NOT ready to scale yet; n=1 head, n=1 seed).**
Before a big expensive run, confirm the effect is robust and get more
heads to specialize, all at current (small) scale:
1. **Reproducibility:** 2 more seeds of the v9 config. A correction-
   surviving chemical head in all 3 → green light to scale. (Gate.)
2. **Charge-conditioned bias `bias_h(Δm, z)`** — now the *best-motivated*
   lever: head 4 is cramming isotope spacing at 0.33 *and* 0.50 *and*
   1.003 into one curve (it learned all three!). Let each charge index
   its own spacing and the isotope head should sharpen sharply, and more
   heads may free up for residues.
3. **L1 sparsity penalty** — meaningful now that the bias is load-bearing
   (v8's head-death was on a *dispensable* bias).

Scaling (d=512 / 12 layers / 16 heads) comes *after* these confirm a
robust, multi-head effect — scaling amplifies what's there, and "1 of 8
on 1 seed" is too fragile to bet a 4–8× run on. "Only 1 head" looks like
an optimization/incentive problem, not a capacity one.

---

## 4. Targets to watch on the v7 run (σ = 0.1)

| metric | v6 result (σ=0.3) | v7 target (σ=0.1) |
|---|---|---|
| `val/rmse_mz` | 0.22 (off a 0.30 floor) | lower *absolute* Da, and meaningfully below the new 0.1 floor |
| `train/grad_norm_delta_bias` | 0.05–0.2, non-vestigial | stays meaningful (watch for collapse → would mean task too easy) |
| `train/grad_norm_total` | spiked 25–32 (NLL overconfidence) | likely still spiky — *not* addressed this run, separate lever |
| **Fine curves: ±1.003 (¹³C)** | broad bump, unresolvable | **sharp spike on ≥1 head** — the falsifiable discovery signal |
| **Coarse peaks: alignment to masses** | 0.6× chance (0.2–0.8 Da off) | **strong peaks land <0.15 Da of residue/loss masses, > chance** |
| Per-head differentiation | present | present/stronger |

Run the same quantitative alignment check (`find_peaks` + nearest-ref vs
3.9% chance baseline) on v7's `final.pt` — that's the objective test, not
the eyeball-the-reference-lines impression.

**The single most falsifiable claim:** if a head develops a sharp peak
at ±1.003 Da in the bias curve under Gaussian-only denoising, that's
the model learning ¹³C isotope spacing from real spectra without any
chemistry-specific supervision. If no head does this within ~20k steps,
something else is wrong (capacity, frequency range, learning rate on
the bias module).

---

## 4b. How we evaluate (`msdelta-analyze`)

`msdelta/analyze.py` has two modes (`--mode {align, probe, both}`) that
answer two different questions. Both implement deliverables from plan §6.2.

### Mode `align` (§6.2b) — *what did the bias curve learn?*

Eyeballing reference lines is unreliable — 27 reference Δm values in
[2, 200] Da means some peak is always near some line.

```
msdelta-analyze --ckpt runs/<run>/final.pt --mode align [--plot]
```

Per head: evaluate the bounded bias curve on a dense Δm grid, detect
peaks (`scipy.signal.find_peaks`), count how many land within `tol` Da
of a chemistry reference (isotopes / residues+losses), matched on `|Δm|`.
**Binomial null:** `binomtest(n_aligned, n_peaks, chance_rate)` — a head
only "found chemistry" if significantly enriched over chance (p<0.05).
Writes `alignment_scores.json` (+ annotated PNGs with `--plot`).

**Baselines:**

```
v6 (σ=0.3) final  — FINE  enrich ~1.0–1.6×, all p>0.30 ; COARSE ~1.0×, all p>0.15
v7 (σ=0.1) @15k   — FINE  best p=0.34       ; COARSE best head0 p=0.081, 1.5×
→ Neither run has a head significantly above chance. (v7 trending up but unconverged.)
```

"Coverage" (refs hit by ≥1 head) reads 25/26 but is meaningless without
the binomial test — at chance, scattered peaks cover most refs anyway.

### Mode `probe` (§6.2c) — *does the head actually USE its bias?*

The alignment mode reads the static bias function; the encoder never sees
a spectrum. The probe is the load-bearing test: on real val spectra,
histogram each head's attention weights `α^(h)_ij` as a function of
`Δm_ij`, density-normalize to **mean attention-per-pair** (removes the
"nearby peaks are just more numerous" confound), and Spearman-correlate
against the bias curve. Attention averaged over layers (bias is shared),
diagonal + padded pairs excluded.

```
msdelta-analyze --ckpt runs/<run>/last.pt --mode probe --plot
```

Writes `probe_scores.json` + `probe_{fine,coarse}.png` (twin-axis:
attention/pair in blue, bias logit in red).

**Finding — v7 (σ=0.1) @15k: the bias is LOAD-BEARING, not vestigial.**

| range | per-head Spearman(bias, attention/pair) |
|---|---|
| fine [−5, 5] | 0.65 – **0.98** |
| coarse [−200, 200] | 0.72 – 0.91 |

![v7 probe: attention (blue) tracks bias (red)](figures/v7_probe_coarse_15k.png)

Every head's attention concentrates exactly where its bias is high. This
**rewrites the diagnosis.** The earlier worry — "the model denoises via
the content path and the bias is vestigial" — is *false*. The bias is the
dominant mechanism routing attention. The architecture works as intended.

The problem is therefore **not** that the bias is ignored; it's that the
bias learned the *wrong shape* — a broad locality bump (attend to
everything within ±1–2 Da) plus high-frequency noise, rather than sharp
peaks at chemistry offsets. The model faithfully uses that bias for
generic local smoothing, which is exactly what the denoising task
rewards.

**Implication for next steps:** the lever is the *incentive*, not the
plumbing. Reshaping the bias toward chemistry means changing what the
task / regularization rewards (penalize the broad-locality solution,
or reward using specific Δm offsets) — not worrying about whether the
bias matters. (b) + (c) together: the wiring is sound, the objective
isn't pointing it at chemistry.

---

## 5. Things we changed along the way that aren't task-related

A few infrastructure / small fixes worth recording so we don't re-litigate:

- **Package layout:** `src/msdelta/` → top-level `msdelta/`. Hatch's
  editable install was silently building empty wheels for the src
  layout; top-level fixed it.
- **uv pin for torch:** explicit `torch>=2.10,<2.12` + `[tool.uv.sources]
  pytorch-cu128` index. The default PyPI torch is cu130 which doesn't
  match the system driver.
- **YAML float parsing:** `1.0e3` parses as a *string* in PyYAML; use
  `1.0e+3`. Bit me once.
- **`MPMHeads.init_log_var = 10`:** initial Gaussian NLL was ~10⁵ otherwise
  (squared error on raw m/z is huge). Init the log_var head's bias near
  the upper clamp so the initial Gaussian is wide. The denoise variant
  inits to −2 instead (small residual targets).
- **Diagonal-zero:** kept it in v5 even though we changed the task. The
  argument is the same: self-attention suppression isn't a Δm-driven
  decision, so the bias path shouldn't carry it.

---

## 6. What I'd consider next

**Reframe from the probe finding (§4b mode c):** the bias is load-bearing,
so the goal is no longer "make the model use the bias" — it does. The goal
is "make the bias's *shape* be chemistry, not a broad locality bump." That
means changing the incentive so the broad-locality solution stops being
optimal.

Direct levers on the incentive (most-aligned with the finding first):

1. **Penalize the broad-locality bias shape.** Add a regularizer that
   discourages a smooth low-frequency bump (e.g. L1 on the bias curve, or
   penalize bias mass at small |Δm|), forcing the limited bias budget onto
   sparse, specific Δm offsets. Cheapest test of the hypothesis.
2. **Make locality unhelpful for the task.** The denoising-by-local-average
   shortcut works because nearby peaks exist. A task where the *useful*
   reference is at a specific chemical Δm — not just "nearby" — would force
   sharp peaks. E.g. ELECTRA-style replaced-peak detection (decide if a
   peak is real or swapped from another spectrum); a swapped peak breaks
   the residue/isotope ladder, which a locality bump can't detect.
3. **Use the `peptide_charge` label** (already in the parquet) for
   contrastive / classification. Biggest scope; the supervision is
   structural so the bias would have reason to encode real spacings.

Task-noise tweaks (lower priority now that we know the bias is used):
- Wider/narrower σ — affects resolution but not the locality-bump
  incentive itself.

Things we should *not* go back to without a new idea:
- MPM as originally formulated (positional ambiguity is fundamental).
- Bigger shared bias MLP (gradient-pool competition, not raw capacity).
- Chemistry-specific noise (engineers the answers).

---

## Appendix: file inventory (current)

```
msdelta/
├── pyproject.toml         # uv-managed; pinned torch cu128
├── configs/
│   ├── pretrain_small.yaml   # production: d=256, 6 layers, 50k steps
│   └── toy.yaml              # smoke: d=64, 2 layers, 100 steps
├── msdelta/
│   ├── __init__.py
│   ├── fourier.py            # Fourier feature module
│   ├── data.py               # ConsensusParquet + DenoiseConfig + denoise_collate
│   ├── model.py              # MSEncoder, DeltaMZBias (per-head), DenoiseHead
│   ├── viz.py                # bias-curve plots with chemistry references
│   ├── train.py              # CLI: msdelta-train --config ...
│   └── analyze.py            # CLI: msdelta-analyze --mode {align,probe,both}
├── docs/
│   ├── EXPERIMENTS.md        # this file
│   └── figures/              # PNGs referenced above
└── runs/                     # gitignored: checkpoints + bias-curve PNGs + wandb
```
