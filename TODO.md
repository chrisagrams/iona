# TODO — observed problems

Issues found while reading the repo. Each is tagged **Verified** (reproduced with a script in this
repo) or **Flagged** (read from the code, not yet reproduced). Ordered by severity within each
section.

`TODO.txt` holds the *research* agenda (precursor leakage, ablations, scaling laws, SAE
interpretability). This file is about defects and infrastructure gaps.

---

## P0 — Broken code

### 1. `MSDeltaModel_2`'s residual loop makes the model one layer deep — **Verified**

[`model/experimental.py:262-269`](msdelta/model/experimental.py#L262-L269)

```python
for block in self.blocks:
    hidden_states_residual = block(hidden_states, bias, padding_mask)   # overwritten each pass
hidden_states = self.norm(hidden_states + hidden_states_residual)
```

`hidden_states` is never reassigned inside the loop, so every block receives the embedding output
and all but the last result is discarded. The model computes exactly
`norm(embed + blocks[-1](embed))` (`allclose == True` against a direct construction).

Measured on a 4-layer model: blocks 0–2 have `grad=None`, **48% of parameters can never train**
(~89% at the 10-layer 50m tier). Depth becomes a no-op, so any depth sweep would produce a flat
line that looks like a finding. Under DDP this most likely raises the unused-parameter error
unless `ddp_find_unused_parameters=True` — which is probably how it would first surface.

```python
# fix, if a global embedding→output skip was the intent
residual = hidden_states
for block in self.blocks:
    hidden_states = block(hidden_states, bias, padding_mask)
hidden_states = self.norm(hidden_states + residual)
```

- [ ] Fix the loop, or delete the class if the experiment is abandoned.
- [ ] Decide whether the skip is even wanted: each `EncoderBlock` is already internally residual,
      so the embedding already reaches the output. An unweighted skip also double-counts the
      embedding against an L-deep stream and shifts the pre-`norm` scale; a learned or scaled skip
      would be the controlled version.

### 2. `experimental.py` reads a config field that no longer exists — **Verified**

[`model/experimental.py:65`](msdelta/model/experimental.py#L65) does `self.scale =
config.delta_bias_scale`, but that field was deleted from `MSDeltaConfig` in commit `7396e38`
("Remove delta bias output scaling"). Constructing any model from this file raises
`AttributeError`.

- [ ] Add `delta_bias_scale: float = 8.0` to `MSDeltaConfig.__init__` **and** `_validate()`, then
      apply it inside `DeltaMZBias._curve` so `forward` and `evaluate` stay consistent.
- [ ] Add the key to whichever `configs/*/config.json` tiers will use it.

---

## P1 — Silently wrong

These produce no error. That is what makes them expensive.

### 3. The 400M config sets an option that no longer exists — **Verified**

[`configs/msdelta-base-400m/config.json:18`](configs/msdelta-base-400m/config.json#L18) carries
`"zero_bias_diagonal": true`. The option was removed in commit `22c0a14`; `PretrainedConfig`
absorbs unknown keys through `**kwargs`, so it becomes a live attribute
(`getattr(cfg, "zero_bias_diagonal") is True`) that **nothing reads**. The 400M tier is not doing
what its config says, and it is the only tier that differs.

- [ ] Delete the key, or re-implement the option.
- [ ] Consider a `_validate()` warning on unrecognized keys — this failure mode is generic and
      will recur for any future removed option.

### 4. The processor pads `labels` two different ways — **Verified**

| method | pad value | used by |
| --- | --- | --- |
| [`__call__` (`processing.py:197`)](msdelta/data/processing.py#L197) | `0.0` | pretraining |
| [`pad` (`processing.py:135`)](msdelta/data/processing.py#L135) | `-100.0` | denoising |

```
__call__ : [[0.167, 0.333, 0.5], [0.333, 0.667,  0.0 ]]
pad()    : [[0.0,   1.0,   0.0], [1.0,   0.0,  -100.0]]
```

Both are correct *in their own context* — pretraining labels are an intensity distribution where
`0.0` is harmless, denoising labels are binary where `-100` is the ignore index. But they are the
same attribute name on the same class. Anyone who reaches for `__call__(return_labels=True)` to
build denoising data gets padding silently labelled as the **signal** class, with no error.

- [ ] Rename one of them, or make the pad value an explicit argument, or at minimum document the
      split at both call sites.

### 5. Padded query rows carry garbage into `last_hidden_state` — **Verified**

[`modeling.py:124`](msdelta/model/modeling.py#L124) masks padded **keys** only. Padded query rows
still produce output vectors: a padded position came back with hidden-state norm **5.54**, not 0.

Every current consumer (`pool_tokens`, both losses) re-applies `attention_mask`, so nothing is
wrong today — but it is an unwritten invariant. New code that reads `last_hidden_state` and
forgets will silently average garbage into its result.

- [ ] Either zero padded rows before returning, or document the invariant in
      `MSDeltaModel.forward`'s docstring. A test would be better than either.

### 6. Malformed spectra vanish without a trace — **Flagged**

[`data/loading.py:107`](msdelta/data/loading.py#L107) catches `ValueError` from the processor and
emits an empty spectrum, which `build_preprocessed_dataset` then filters out. There is no counter
and no log line. A preprocessing regression that drops 30% of the corpus looks identical to a
working run.

- [ ] Count drops and log the total (and the first few reasons) after preprocessing.

### 7. The eval set is the head of the validation split, not a sample — **Verified**

[`train/cli.py:107`](msdelta/train/cli.py#L107):

```python
eval_ds = val_ds.select(range(min(len(val_ds), eval_size)))
```

This takes the **first** `validation_batches × per_device_eval_batch_size` rows. It is only
representative because the upstream dataset happens to be pre-shuffled
(`massive_kb_v1_shuffled`). Point it at any ordered corpus — or a locally built
`--dataset_root` — and the eval metric silently describes a biased slice.

- [ ] Use `val_ds.shuffle(seed=training_args.seed).select(...)`.

### 8. Importing `experimental.py` rebinds the HF auto-classes — **Flagged**

[`model/experimental.py:341-342`](msdelta/model/experimental.py#L341-L342) re-declares
`MSDeltaModel` / `MSDeltaForPreTraining` and calls `register_for_auto_class` at **import time**, so
importing it after `msdelta/__init__.py` re-registers the auto-classes against different class
objects. It is deliberately excluded from `msdelta/model/__init__.py` for this reason, but nothing
stops a direct import.

- [ ] Give variant classes distinct names and drop the registration calls.

---

## P2 — Design constraints worth an experiment

### 9. The dense bias tensor is the scaling ceiling — **Verified (by construction)**

The bias is passed to `scaled_dot_product_attention` as a float `attn_mask`, which rules out the
FlashAttention kernel and materializes `(B, A, K, K)`. The `DeltaMZBias` intermediate is worse:
`(B, K, K, 2·n_freqs)` ≈ 184M floats at `B=64, K=150, n_freqs=64`, before the head MLPs reduce it.

This — not parameter count — is why `max_peaks` is 150 and why the denoising probe needs a
peak-pair budget sampler. Any meaningful increase in `K` requires a low-rank, bucketed, or sparse
bias. See [ARCHITECTURE.md §6.6](docs/ARCHITECTURE.md#66-breaking-the-k-ceiling).

- [ ] Profile how much of step time is the bias path before optimizing anything else.

### 10. One bias module is shared by every layer — **Flagged**

Layer 0 and layer 19 see identical mass structure. The bias module is **0.08% of parameters**, so
per-layer curves are nearly free in weights. Highest-leverage architecture experiment available;
recipe and the four call sites it breaks are in
[ARCHITECTURE.md §6.1](docs/ARCHITECTURE.md#61-per-layer-bias-curves-highest-leverage).

### 11. Seven of sixteen intensity Fourier frequencies are dead at init — **Verified**

`fourier_int_f_min=0.01` with `log_intensity ∈ (0,1]` (span ≈ 1.0). By the repo's own
`dead_freqs` metric — frequencies completing less than half a cycle across the data span —
**7/16 are dead at initialization** in every shipped tier:

```
0.01, 0.0185, 0.0341, 0.0631, 0.1166, 0.2154, 0.3981 | 0.7356, 1.36, 2.51, ... 100.0
└──────────────── dead (7) ────────────────────────┘
```

Nearly half the intensity featurizer's capacity is wasted unless training moves those frequencies
(which `fourier/int_drift_log10` would show). The Δ featurizer is fine: 0/64 dead over a 2000 Da
span.

- [ ] Check `fourier/int_dead` on a real run — if it stays at 7, raise `fourier_int_f_min` to
      ~0.5 and re-tune. Cheap ablation, and it is already instrumented.

### 11b. float32 m/z undercuts the top of the Δ-bias frequency range — **Verified**

`delta_bias_f_max = 1000` implies a finest period of **1e-3 Da**, which is what justifies the claim
that the bias can resolve isotope fine structure. But m/z arrives as float32 from the processor and
`Δ = mz_i − mz_j` is computed in that dtype, so:

| m/z | float32 ULP | periods-per-ULP at f=1000 |
| --- | --- | --- |
| 200 | 1.53e-5 Da | ~65 samples per period — fine |
| 1000 | 6.10e-5 Da | ~16 |
| 2000 | 1.22e-4 Da | **~8** |

Phase error at `f_max` for an isotope-spaced pair near m/z 1900 is **0.364 rad (≈21°)**. The
practical symptom: the bias is mathematically translation-invariant in mass, but numerically it is
not — shifting a whole spectrum by +1000 Da changes the bias by up to **0.0167**, about 1.6% of the
curve's dynamic range at init.

So the highest-frequency Δ features are progressively noisier as m/z grows, exactly in the regime
(large peptides) where fine mass differences matter most.

- [ ] Check whether the learned frequencies actually populate the top of the range, or whether
      training abandons it — `fourier/dm_f_max` and the `fourier/dm_log10_freqs` histogram already
      log this.
- [ ] Cheap fix to test: compute `Δ` in float64 (or subtract a per-spectrum offset such as the
      precursor or min m/z before differencing) and cast to float32 only afterwards.
      `FourierFeatures.forward` already casts to float32 internally, so only the subtraction in
      `DeltaMZBias.forward` needs to change.
- [ ] If the top of the range is unusable, lower `delta_bias_f_max` and reclaim the frequencies.

### 12. Pretraining loss is not comparable across `--mask_ratio` — **Flagged**

[`modeling.py:303`](msdelta/model/modeling.py#L303) uses `reduction="batchmean"`, which divides by
batch size, not by the number of masked peaks. The shipped args use `mask_ratio=0.50` against a
collator default of `0.15`, so the loss magnitude moves with the masking setting.

- [ ] Log a normalized variant (per masked peak) alongside it, so sweeps stay readable.

### 13. `clamp_abs=2000.0` silently saturates large Δ — **Flagged**

`FourierFeatures` clamps `|x| ≤ 2000`. Correct for peptide MS/MS; wrong for intact-protein or
lipid data, where it would silently flatten the tail rather than error.

- [ ] Make it a config field rather than a constructor default if the data ever broadens.

---

## P3 — Missing infrastructure

### 14. There are no tests — **Verified**

No `tests/`, no CI, no fixtures. Nothing checks the processor invariants, the KL loss, the
collator's masking, or that the model runs. Every issue in the P1 section above is the kind a
handful of tests would have caught, and the restructure just proved how much rests on unwritten
invariants.

Highest-ROI first tests:

- [ ] `_process_one`: thresholding, top-K keeps m/z order, `labels` sums to 1, `selected` indices
      align with the input.
- [ ] Collator: mask count matches `mask_ratio`, `min_masked` respected, padding is consistent.
- [ ] Loss: KL is zero for a perfect prediction; batch with nothing masked returns exact 0; a row
      with nothing masked does not produce NaN (all three currently hold — lock them in).
- [ ] Round-trip: `save_pretrained` → `AutoModelForPreTraining.from_pretrained` preserves outputs.
- [ ] Shapes: `bias_module(mz)` is `(B, A, K, K)`; `evaluate(grid)` is `(G, A)`.
- [ ] A CI job running `ruff check`, `ruff format --check`, and the tests.

### 15. No extension point for architecture experiments — **Verified**

[`train/cli.py:65`](msdelta/train/cli.py#L65) hardcodes `MSDeltaForPreTraining(model_config)`, so
every new architecture needs an edit to shared training code.

- [ ] Add a `--model_class` dotted-path loader (~15 lines). After that, a new architecture is one
      new file plus one args-file line, with no edits to existing code.

### 16. `--config_overrides` cannot introduce new config keys — **Verified**

`update_from_string` raises `ValueError: key my_new_knob isn't in the original config dict` for any
key not already on the config. New keys must go in `config.json`, where the `**kwargs`
pass-through accepts them. Not a bug, but it is the first thing that will trip up a sweep over a
new knob.

- [ ] Documented in [configs/README.md](configs/README.md); no code change needed.

### 17. Two different `precursor_mz` implementations — **Verified**

[`data/loading.py:31`](msdelta/data/loading.py#L31) returns `0.0` for unknown residues and handles
`[+15.99]` modification brackets. [`eval/probe.py:36`](msdelta/eval/probe.py#L36) returns `None`
and rejects non-standard peptides. The stricter one is right for probing; the divergence is
undocumented and they will drift.

- [ ] Consolidate into `data/chemistry.py` with an explicit `strict=` flag.

### 18. No FLOP accounting — **Flagged** (also in `TODO.txt`)

The scaling-law comparison needs FLOPs per step. Non-standard here because the bias path is
`O(K²·n_freqs·per_head_hidden)` and does not follow the usual `6ND` rule.

- [ ] Derive a closed form including the bias path; validate against wall-clock on one tier.

---

## Working tree — uncommitted, from the in-progress edits to `modeling.py`

Not committed yet, so these are cheap to fix now and annoying later.

### 0. **The cross-block residual edit in `modeling.py` collapses the model to one layer** — **Verified**

[`model/modeling.py`](msdelta/model/modeling.py) `MSDeltaModel.forward` now reads:

```python
for block in self.blocks:
    residual_hidden_states = block(hidden_states, bias, padding_mask)   # overwritten each pass
hidden_states = self.norm(hidden_states + residual_hidden_states)
```

This is issue 1 (`MSDeltaModel_2`) reproduced in the **production** model class. `hidden_states` is
never reassigned, so every block receives the embedding output and only the last block's result
survives. Measured on a 6-layer model: blocks 0–4 have `grad=None`, **75% of parameters cannot
train**; the model computes `norm(embed + blocks[-1](embed))`.

An instrumented forward with all residual branches neutralised shows the pre-final-norm state is
**exactly 2× the embedding** — the embedding counted once by the (already existing) implicit
residual highway, and once again by the new explicit skip.

- [ ] Fix by assigning back into `hidden_states` and hoisting the residual out of the loop, or
      revert. See the gated alternative discussed below.

### 19. `modeling_classic.py` is an unmarked byte-identical fork — **Verified**

[`model/modeling_classic.py`](msdelta/model/modeling_classic.py) is a 394-line copy of
`modeling.py` as it stood before the section-comment edits. Three consequences:

* **3 `ruff` TID252 errors** — it uses the same relative imports as `modeling.py`, but the
  `pyproject.toml` per-file ignore lists only `model/modeling.py` and `model/experimental.py`.
  `ruff check` currently fails.
* **It re-registers the HF auto-classes at import time** (same hazard as issue 8): it declares
  `MSDeltaModel` / `MSDeltaForPreTraining` / `MSDeltaForDenoising` and calls
  `register_for_auto_class` at module scope.
* **Nothing marks which file is canonical.** A fix applied to `modeling.py` will not reach the
  copy, and the two will diverge silently.

- [ ] If it is a safety snapshot, delete it — git already has that history.
- [ ] If it is a real baseline variant, give the classes distinct names, drop the
      `register_for_auto_class` calls, add it to the per-file ignore list, and put a one-line
      docstring at the top saying what it is a baseline *for*.

### 20. The new section comments break isort — **Verified**

`ruff check` reports `msdelta/model/modeling.py:3:1: I001 Import block is un-sorted or
un-formatted`. The `# Outputs` comment sits one blank line after the import block, so ruff's isort
treats it as a trailing part of that block. Two blank lines before the comment fixes it.

The comments themselves are a genuine improvement to the file's navigability — worth keeping.
Two typos to fix while you are in there, since they are now the file's headings:
`## Custom ttention implementation` → `attention`, and `## Stack if MHA` → `of`.

- [ ] Add the blank line, fix the two typos.

### 21. Two files fail `ruff format --check` — **Verified**

`model/modeling.py` (the comment spacing above) and `data/processing.py` (one line that fits in
100 chars is split across three). Both are cosmetic and the `ruff format` pre-commit hook will fix
them automatically on commit — noted only so the current red `ruff check` is not mistaken for
something deeper.

---

## Fixed during the documentation and restructure pass

- [x] **`.gitignore` swallowed the new `msdelta/data/` package.** The benchmark-data rule was an
      unanchored `data/`, which matched any directory at any depth — `msdelta/data/__init__.py`
      and its README would never have been committed. Anchored to `/data/`.
- [x] **Root `README.md` was empty** (0 bytes). Now the project overview, with
      [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md), [docs/API.md](docs/API.md),
      [model/MODULES.md](msdelta/model/MODULES.md), and per-package READMEs.
- [x] **Two `ruff` TID252 errors** in the then-untracked scratch file. The per-file ignore now
      covers `model/modeling.py` and `model/experimental.py`, with a comment explaining that the
      relative imports are load-bearing for HF's `custom_object_save`.
- [x] **Flat package restructured** into `model/`, `data/`, `train/`, `eval/` with one-way
      dependencies; `python -m msdelta.train` preserved for the Polaris launcher.
