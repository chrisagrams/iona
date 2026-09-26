# Spectrum embeddings: zero-shot (frozen pretrained encoder)

Retrieval with the pretrained encoder and no contrastive training: embed each spectrum with a frozen
encoder, optionally post-process the embeddings, and retrieve by cosine similarity.

## Files

| file | content |
|---|---|
| `C_zeroshot.png` | left: all 24 frozen encoders on ms-contrastive-100k test, MAP@R vs pretraining steps, one line per size (solid: best layer + ABTT, dashed: best layer raw); right: the 6 encoders scored on yeast 20k, raw vs ABTT |
| `C_zeroshot.csv` | the plotted values plus the chosen layer and number of components; also the two frozen 400M encoders scored on mouse 20k (used in `../C_transfer`, not plotted here) |
| `plot_zeroshot.py` | regenerates the figure from the CSV (`python plot_zeroshot.py`; matplotlib + numpy) |

CSV columns: `benchmark`, `encoder`, `raw_final` (last layer, raw), `raw_best_layer` (best layer, raw),
`abtt_best` (best layer after ABTT), `abtt_D` (components removed), `abtt_layer`, `abtt_fit` (data the
ABTT statistics were fitted on).

## Setup

- **Embedding**: mean+max pooling over peaks of each transformer block's output ("block NN") or the final layer.
- **ABTT** (all-but-the-top; Mu & Viswanath, 2018): subtract the mean embedding and project out the top
  D principal components, D in {8, 32, 64, 128}. The mean and components are fitted on training spectra,
  never on the test set: the ms-contrastive-100k train split for the in-distribution panel, and 25,000
  spectra from the train split of the eight other species for the yeast panel.
- **Benchmarks and metric**: MAP@R over experimental spectra, as in `../README.md`
  (ms-contrastive-100k test, 25,137 queries; yeast 20k subset, 20,019 queries).

## Results

- ABTT roughly doubles retrieval for every encoder. In-distribution the best is 0.66 (400M@330k); every
  400M checkpoint from 120k on (0.54-0.66) is above every smaller encoder (at most 0.44). Yeast best:
  0.71 (400M@220k).
- Frozen retrieval does not rise steadily with pretraining: most sizes peak at 120k-330k and dip later
  (400M@430k 0.54).
- On unseen yeast the frozen 400M encoder with ABTT (0.71) beats every fine-tuned model we have there
  (at most 0.66), though not GLEAMS (0.77, see `../benchmarks/`).

## Caveats

- The layer and D shown for each encoder are the best **on that benchmark's test set**, so in principle
  upper bounds. A validation pass (choose on ms-contrastive-100k validation, report test) was completed
  for three encoders (50M@220k/330k/540k): validation picked the same layer and D as test every time,
  so for those the reported numbers are unchanged. The remaining encoders were not re-checked.
- Updated 2026-09-26: the in-distribution panel previously showed 8 encoders from an early summary
  (best 0.43, 200M@540k); it now shows all 24.
- The two panels cover different encoder sets (what was evaluated on each benchmark).

Provenance: msdelta repository, `sweeps/package_contrastive.py` (from `results/finetune/contrastive/zeroshot-layers-abtt/`,
summarised by `sweeps/summarise_zeroshot.py`, `results/finetune/contrastive/nine20k_zeroshot/` and
`results/finetune/contrastive/mouse20k_zeroshot_*/`).
