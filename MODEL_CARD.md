---
license: apache-2.0
library_name: transformers
pipeline_tag: feature-extraction
tags:
  - mass-spectrometry
  - proteomics
  - tandem-mass-spectrometry
  - custom_code
datasets:
  - Gaolaboratory/MSConsensus-100M
---

# Iona

Iona is a transformer encoder for tandem mass spectra (MS/MS). It treats each centroided peak as a
token and learns how peaks relate to each other through a per-head attention bias over the signed
m/z difference (Δm/z) between every pair of peaks.

## Model details

- **Input:** a centroided spectrum given as parallel `mz` and `intensity` arrays, with at most 512
  peaks.
- **Peak tokens:** built from the normalized log intensity only. Tokens carry no absolute m/z; the
  model sees mass only through pairwise Δm/z.
- **Δm/z attention bias:** each attention head gets a learned bias curve over Δm/z. The curve is
  computed by a small MLP on Fourier features of Δm/z (256 frequencies, 0.001–190).
- **Output:** one contextual embedding per peak (`last_hidden_state`).

| Size | Hidden size | Layers | Heads | FFN size |
|---|---|---|---|---|
| iona-base-50m | 640 | 10 | 10 | 2560 |
| iona-base-100m | 800 | 13 | 10 | 3200 |
| iona-base-200m | 1024 | 16 | 16 | 4096 |
| iona-base-400m | 1280 | 20 | 20 | 5120 |

## Usage

The checkpoint includes its modeling code, so it loads with `trust_remote_code=True`. You need
`torch`, `transformers`, `sentence-transformers` and `pytorch-metric-learning` installed.

```python
import torch
from transformers import AutoModelForPreTraining, AutoProcessor

repo = "Gaolaboratory/iona-base-50m"
processor = AutoProcessor.from_pretrained(repo, trust_remote_code=True)
model = AutoModelForPreTraining.from_pretrained(repo, trust_remote_code=True).eval()

mz = [175.119, 262.151, 363.198, 476.282]
intensity = [1200.0, 5300.0, 800.0, 2400.0]
inputs = processor(mz, intensity, return_tensors="pt")

with torch.no_grad():
    peak_embeddings = model.iona(**inputs).last_hidden_state  # (batch, peaks, hidden_size)
```

The processor normalizes intensities, so pass raw intensity values. For a single embedding per
spectrum, pool `peak_embeddings` over the peaks where `inputs["attention_mask"]` is 1.

## Training

- **Data:** [Gaolaboratory/MSConsensus-100M](https://huggingface.co/datasets/Gaolaboratory/MSConsensus-100M).
- **Objective:** masked-intensity modeling. 50% of peaks have their intensity token replaced by a
  learned mask token. The model predicts how intensity is distributed across the masked peaks,
  trained with a KL-divergence loss against the true normalized intensities.
- **Optimization:** AdamW (β₁ = 0.9, β₂ = 0.95), learning rate 1.3e-4 with 2,000 warmup steps and
  cosine decay, weight decay 0.01, bf16, 3 epochs.
- **Code:** <https://github.com/chrisagrams/msdelta>

## Evaluation
<!-- TODO: add results for the published checkpoint. -->

## Limitations

- Inputs must be centroided spectra with between 1 and 512 peaks. Profile-mode data needs to be
  centroided first.
- The model was pretrained on the spectra in MSConsensus-100M. Performance on other instruments,
  fragmentation methods or sample types hasn't been characterized.

