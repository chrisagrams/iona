---
library_name: pytorch
tags: [mass-spectrometry, proteomics, peptide-embedding, cross-modal, msdelta]
---

# iona-peptide-embedder-400m

Maps a (modified) peptide + precursor charge to a 2560-d unit vector in the **spectrum
embedding space of [`Gaolaboratory/iona-contrastive-400m`](https://huggingface.co/Gaolaboratory/iona-contrastive-400m)**,
so a spectrum and a candidate peptide can be compared directly by cosine
(spectrum -> peptide retrieval, candidate scoring for rescoring).

A small transformer (4 layers x 256, 4.77M parameters): residue + position + charge
embeddings, Fourier features of each residue's modification mass, mean+max pooling, MLP
projection. Trained as a student to regress the frozen `iona-contrastive-400m` embedding of
each peptide's spectra (MSE on unit vectors).

## Training

`chrisagrams/ms-contrastive-100k` train split (experimental spectra; peptides of
`chrisagrams/ms2-peptide-replicate-retrieval` excluded), targets = `iona-contrastive-400m`
spectrum embeddings; 3 epochs, lr 1e-3, seed 0. Selected as the best of 3 seeds by
validation cross-modal Hit@1 (0.946).

## Results: `chrisagrams/ms-contrastive-100k` test split

Every experimental test spectrum (25,848) ranked against every test peptide+charge
(9,771 candidates), spectrum embedded by `iona-contrastive-400m`:

| Hit@1 | Hit@5 | MRR |
|---|---|---|
| 0.9229 | 0.9489 | 0.9354 |

(3 seeds: 0.9226-0.9229.) The same recipe with a weaker 50m teacher reaches 0.898: student
quality follows the teacher.

Caveat: trained and tested on splits of the same dataset. On low-resolution ion-trap MS2
spectra (not in training) retrieval degrades strongly for this family of models.

## Usage

```python
import torch
from huggingface_hub import snapshot_download
import sys; path = snapshot_download("Gaolaboratory/iona-peptide-embedder-400m"); sys.path.insert(0, path)
from peptide_embedder import PeptideEmbedder

peptides = PeptideEmbedder.from_pretrained(path).eval()
p = peptides.embed(["PEPTIDEK", "AC[57.0215]M[15.9949]K"], charges=[2, 2])   # (2, 2560), unit norm

# spectrum side: see the iona-contrastive-400m card (mean+max of the last hidden state, L2-normalised)
# score = s @ p.T   (cosine)
```

Peptide notation: modification mass in brackets after its residue (`C[57.0215]`,
`M[15.9949]`, `N[0.9840]`); N-terminal modification as a leading `[mass]` (`[42.0106]PEPTIDE`).
Charge is the precursor charge (clamped to 0-7).

Internal release; not yet reviewed for publication.
