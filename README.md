# MSDelta

**A mass-spectrum transformer whose only notion of "position" is the continuous mass
difference (Δm/z) between peaks.**

MSDelta encodes a centroided MS/MS spectrum as an unordered set of peak tokens. Each token
carries *only* its log-intensity — m/z is deliberately **not** in the token. All m/z
information enters the network through a **learned, per-attention-head additive bias that is
a function of the signed pairwise mass difference `Δ = m/z_i − m/z_j`**.

This is a continuous-valued analogue of relative position encoding (T5 bias / ALiBi), where the
"distance" between tokens is a physical mass gap in daltons. Because a neutral loss (−18.011 Da
for H₂O), an isotope spacing (+1.00336 Da for ¹³C), or an amino-acid residue gap (+57.021 Da for
glycine) are all *fixed mass offsets*, the model can in principle learn chemistry directly as
bumps in these bias curves — and the repo ships diagnostics that measure exactly that.

---

## Table of contents

| Document | What it covers |
| --- | --- |
| **This file** | Project overview, install, quickstart, repo layout, data flow |
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | **Illustrated architecture + what to change to experiment** |
| [docs/API.md](docs/API.md) | Every module, class, and function that is implemented |
| [configs/README.md](configs/README.md) | Model-size tiers and the full training-argument reference |
| [msdelta/README.md](msdelta/README.md) | Package layout, import conventions, old→new file map |
| [msdelta/model/README.md](msdelta/model/README.md) | The architecture package |
| [msdelta/data/README.md](msdelta/data/README.md) | Preprocessing contract and masking |
| [msdelta/train/README.md](msdelta/train/README.md) | Running a job, callbacks, distributed setup |
| [msdelta/eval/README.md](msdelta/eval/README.md) | The diagnostics and what each one proves |

---

## The core idea in one picture

```
 classic transformer                        MSDelta
 ────────────────────                       ───────
 token = content + position embedding       token   = f(log intensity)      ← no m/z!
 attention = QKᵀ/√d                         attention = QKᵀ/√d + B_h(Δm/z)  ← all m/z lives here
                                                                  ▲
                                            per-head learned curve over signed mass difference
```

A single `DeltaMZBias` module produces a `(batch, heads, peaks, peaks)` bias tensor once per
forward pass; **every encoder layer reuses the same tensor**. Its per-head curves are the
model's interpretable surface: plot `B_h(Δ)` against a table of known isotope / neutral-loss /
residue masses and you can see what chemistry each head has latched onto.

---

## What is in the box

* **Model** — a pre-norm transformer encoder (`MSDeltaModel`) plus two task heads:
  * `MSDeltaForPreTraining` — masked-peak **relative-intensity** prediction (a KL objective over
    a distribution across masked peaks, not a token-classification cross-entropy).
  * `MSDeltaForDenoising` — per-peak signal/noise binary classifier on a frozen encoder.
* **Preprocessing** — a Hugging Face `FeatureExtractionMixin` processor that thresholds, top-K
  filters, log-transforms and pads raw peak lists, plus a masking collator.
* **Training** — a thin `Trainer` subclass driven entirely by HF `TrainingArguments`, with
  DeepSpeed ZeRO-2 and a multi-node PBS launcher for ALCF Polaris.
* **Diagnostics (the interesting part)** — five families of callbacks that run *during* training:
  linear probes, Fourier-frequency health, chemical alignment of the bias curves, replicate
  spectrum retrieval, and a nested denoising probe.
* **Configs** — five parameter tiers from 50M to 1B.

Everything is registered with the HF auto-classes (`AutoConfig`, `AutoModel`,
`AutoModelForPreTraining`, `AutoModelForTokenClassification`, `AutoProcessor`), so checkpoints
round-trip through `save_pretrained` / `from_pretrained` and can be pushed to the Hub.

---

## Install

Requires **Python ≥ 3.13** and [uv](https://docs.astral.sh/uv/). Torch is pinned to the
CUDA 12.8 wheel index on Linux/Windows.

```bash
cd msdelta
uv sync --frozen        # or: uv sync
uv run pre-commit install
```

Smoke test (works without a GPU):

```bash
uv run python -c "import msdelta, torch; print(torch.__version__); print(msdelta.__all__)"
```

## Quickstart

### Encode a spectrum

```python
import torch
from msdelta import MSDeltaConfig, MSDeltaForPreTraining, MSDeltaProcessor

processor = MSDeltaProcessor.from_pretrained("configs/msdelta-base-50m")
model = MSDeltaForPreTraining(MSDeltaConfig.from_pretrained("configs/msdelta-base-50m"))

mz = [101.07, 119.08, 175.12, 246.16]
intensity = [1000.0, 310.0, 880.0, 120.0]

batch = processor(mz, intensity, return_tensors="pt")
with torch.no_grad():
    out = model(**batch)  # -> MSDeltaForPreTrainingOutput(loss=None, logits=(1, 4))
```

### Train

Every run is fully specified by an args file; nothing is hard-coded in Python.

```bash
uv run msdelta-train --args_file configs/msdelta-base-50m/training.args
```

Multi-node on Polaris:

```bash
qsub -A PROJECT -q QUEUE -l select=2:system=polaris -l place=scatter \
     -l walltime=12:00:00 -l filesystems=home:eagle -j oe -o pbs/logs \
     -v ARGS_FILE=configs/msdelta-base-200m/training.args,CHECKPOINT_DIR=/eagle/PROJECT/ckpt \
     pbs/polaris-pretrain.pbs
```

The launcher derives `gradient_accumulation_steps` from `GLOBAL_BATCH_SIZE` and the detected GPU
count, exports the MPI→torch.distributed rank variables, and shares one W&B run across nodes.

---

## Repository layout

The package is split by role into four subpackages, each with its own README.

```
msdelta/
├── README.md                     ← you are here
├── docs/
│   ├── ARCHITECTURE.md           ← illustrated model internals + how to modify them
│   └── API.md                    ← full function index
│
├── msdelta/                      ← the Python package  ▸ msdelta/README.md
│   ├── __init__.py                 public API + HF auto-class registration
│   │
│   ├── model/                    ▸ msdelta/model/README.md
│   │   ├── configuration.py        MSDeltaConfig, MSDeltaDenoisingConfig
│   │   ├── modeling.py             PeakEmbed, DeltaMZBias, BiasedMHA, EncoderBlock,
│   │   │                           MSDeltaModel, and the two task heads
│   │   ├── fourier.py              FourierFeatures + frequency-health metrics
│   │   └── experimental.py         ⚠ scratch variant — does not run, see ARCHITECTURE.md §7
│   │
│   ├── data/                     ▸ msdelta/data/README.md
│   │   ├── processing.py           MSDeltaProcessor + masking collator
│   │   ├── loading.py              dataset resolution, preprocessing, collation
│   │   └── chemistry.py            residue / isotope / neutral-loss mass tables (pyteomics)
│   │
│   ├── train/                    ▸ msdelta/train/README.md
│   │   ├── cli.py                  main() + MSDeltaTrainer  (`msdelta-train`)
│   │   ├── __main__.py             enables `python -m msdelta.train`
│   │   ├── args.py                 ModelArguments, DataArguments, MSDeltaTrainingArguments
│   │   ├── callbacks.py            all diagnostic callbacks + build_callbacks()
│   │   └── wandb_distributed.py    one W&B client per node, one shared run
│   │
│   └── eval/                     ▸ msdelta/eval/README.md
│       ├── alignment.py            statistical test: do bias peaks land on real chemistry?
│       ├── probe.py                linear probes on the frozen encoder
│       ├── retrieval.py            replicate-spectrum retrieval vs a binned baseline
│       ├── denoising.py            nested Trainer that fits a fresh denoising head
│       ├── embedding.py            token pooling + batched spectrum embedding
│       └── viz.py                  matplotlib bias-curve panels
│
├── configs/                      five size tiers + DeepSpeed ZeRO-2  ▸ configs/README.md
└── pbs/                          Polaris multi-node launcher and venv setup
```

Dependencies run one way only — `model/` imports nothing internal, `data/` only its own
`chemistry`, `eval/` uses `model/` + `data/`, and `train/` sits on top of all three. Nothing
imports `train/`. See [msdelta/README.md](msdelta/README.md) for the import conventions, the
old→new file map, and why two files in `model/` deliberately use relative imports.

---

## End-to-end data flow

```
Parquet shards (HF: chrisagrams/massive_kb_v1_shuffled)
  columns: "m/z", "int", "peptide_charge"
        │
        │  data.build_preprocessed_dataset  →  MSDeltaProcessor.__call__
        │    · drop peaks below 1% of the base peak
        │    · keep the top `max_peaks` (150) by intensity, restored to original order
        │    · log_intensity = log1p(i) / max(log1p(i))          ∈ (0, 1]
        │    · labels        = i / Σi                             (relative abundance)
        │    · also derives charge, precursor_mz, log_tic from the peptide_charge string
        ▼
  cached HF dataset: mz, log_intensity, labels, charge, precursor_mz, log_tic, peptide_charge
        │
        │  MSDeltaDataCollatorForPreTraining
        │    · pad to the longest spectrum in the batch
        │    · sample `mask_ratio` (0.50 in the shipped args) of real peaks → mask_positions
        ▼
  batch: mz (B,K) · log_intensity (B,K) · attention_mask (B,K)
         mask_positions (B,K) · labels (B,K)
        │
        ▼
  MSDeltaForPreTraining  →  logits (B,K)  →  KL(softmax over masked ‖ normalized true intensity)
```

Note the split of responsibility: **intensity is the token content and the prediction target;
m/z is used only to build the attention bias.** Masking replaces a peak's intensity token with a
learned `mask_token`, but the masked peak's m/z is still visible to every other peak through the
bias — so the task is "given where this peak sits in mass space relative to everything else, how
big is it?"

---

## Diagnostics

All of these run inline during pretraining on rank 0 and log to W&B (see
[configs/README.md](configs/README.md) for the interval flags).

| Callback | Question it answers | Key metrics |
| --- | --- | --- |
| `BiasPanelCallback` | What do the per-head Δm/z curves look like? | PNG panels, fine `[-5, 5]` Da and coarse `[-200, 200]` Da, with reference masses overlaid |
| `AlignmentCallback` | Are the bias peaks landing on **real chemistry** more often than chance? | `align/n_sig05`, `align/best_p`, `align/*_max_enrich`, `align/*_coverage` — a binomial test per head against the axis fraction covered by reference masses |
| `LinearProbeCallback` | Is precursor m/z, charge, TIC, fragment m/z, neutral-loss membership, or isotope rank linearly decodable from the frozen encoder? | `probe/*_r2`, `probe/*_acc`, `probe/*_auc`, each with a baseline |
| `FourierProbeCallback` | Are the learnable Fourier frequencies healthy — resolving real data, not dead, not drifting? | `fourier/{int,dm}_{mae,dead,drift_log10,f_min,f_max}` |
| `RetrievalCallback` / `ReplicateRetrievalCallback` | Do replicate spectra of the same peptide embed near each other, and does the model beat a 1-Da binned-vector baseline? | `retrieval/mAP`, `retrieval/gap_vs_binned`, `replicate_retrieval/{Hit@1,MAP,R@5}` (± all-but-top-16 whitening) |
| `DenoisingProbeCallback` | Does the frozen encoder support downstream signal/noise classification? | `denoise/{auroc,auprc,f1,balanced_accuracy,…}` from a **fresh head trained in a nested Trainer**, with RNG/grad state forked and restored |

The alignment test is the load-bearing scientific claim: `align/n_sig01_bonf` counts heads whose
peak↔chemistry hit rate survives a Bonferroni correction.

---

## Known rough edges

* The root `README.md` was empty until these docs were added; this is the first prose in the repo.
* **There are no tests.** No `tests/`, no CI. Nothing verifies the processor invariants, the KL
  loss, or the collator's masking.
* `configs/msdelta-base-400m/config.json` still carries `"zero_bias_diagonal": true`, but that
  option was deleted from the code in commit `22c0a14`. `PretrainedConfig` silently absorbs it as
  an unused attribute, so the 400M tier is **not** doing what its config implies.
* `msdelta/model/experimental.py` is imported by nothing and deliberately not re-exported from
  `msdelta/model/__init__.py`. It references a
`config.delta_bias_scale` field that no longer exists on `MSDeltaConfig`, so it
  raises `AttributeError` on construction, and its `MSDeltaModel_2` residual loop has a bug. Details and fixes in
  [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md#7-the-scratch-variant-modelexperimentalpy).
* Because the attention bias is a dense float `attn_mask`, `scaled_dot_product_attention` cannot
  use the FlashAttention kernel — the `(B, heads, K, K)` bias tensor is materialized in memory
  and dominates activation cost. See the memory section in the architecture doc.

`TODO.txt` tracks the open research questions (precursor leakage, regularization, ablations,
scaling laws, SAE interpretability).
