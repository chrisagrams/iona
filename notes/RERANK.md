# PSM rescoring with iona embeddings — handoff guide

Rescore database-search results (e.g. MSFragger's top-10 candidates per spectrum) using the
lab's per-candidate features plus **our spectrum/peptide embedding similarity**. Output: the
best PSM per spectrum with a target-decoy q-value.

## What you need

| | where |
|---|---|
| Spectrum encoder | `Gaolaboratory/iona-contrastive-400m` (private Hub repo; ask for access) |
| Peptide encoder | `Gaolaboratory/iona-peptide-embedder-400m` (private) |
| Global rescorer (optional, plug-and-play) | `Gaolaboratory/iona-rerank-400m` (private) |
| Input data | one parquet per run in the `Gaolaboratory/psm-rerank-hek-hct116` schema (`spectra/<dataset>/<run>.parquet`: MS2 peaks + the candidate list with MSFragger's scores, `is_decoy`) **and** the lab's feature table for the run (`features/<dataset>/<run>.parquet`, keyed on `candidate_id`; produced by the lab's `psm_features.py`) |
| Software | this repo (`msdelta`), Python 3.12, torch, transformers, pandas, pyarrow, scikit-learn, safetensors, huggingface_hub; a GPU for step 1 (CPU works, slowly) |

`HF_TOKEN` must be set for the private repos. On Aurora (Intel XPU), after `module load frameworks`, set
`export ONEAPI_DEVICE_SELECTOR=level_zero:gpu ZE_FLAT_DEVICE_HIERARCHY=FLAT` -- the module's default
selector makes torch segfault on import.

## Steps

```bash
# 1. embed every candidate of a run (GPU; ~40 min for a 40k-spectrum run on one Aurora tile)
python -m msdelta.rerank_psm_embed --run my_run.parquet --out rows/my_run.parquet \
    --encoder Gaolaboratory/iona-contrastive-400m \
    --student Gaolaboratory/iona-peptide-embedder-400m

# 2a. rescore per run (RECOMMENDED; Percolator-style: 3-fold by spectrum within each run,
#     iterative labels, linear model; no pretrained rescorer needed)
python -m msdelta.psm_rerank score --rows rows/ --labfeat features/ --out psms.parquet

# 2b. or apply the pretrained global model (scores a new run without training on it)
python -m msdelta.psm_rerank score --mode global --model Gaolaboratory/iona-rerank-400m \
    --rows rows/ --labfeat features/ --out psms.parquet

# control: the same rescoring WITHOUT the embedding features
python -m msdelta.psm_rerank score --no-embedding --rows rows/ --labfeat features/ --out base.parquet
```

`psms.parquet`: `spectrum_id, run_id, candidate, peptide, is_decoy, score, q_value` (one row
per spectrum). Accept targets with `q_value <= 0.01` for 1% FDR. The command also prints
PSMs/peptides at 1% FDR for MSFragger's own ranking and for the rescored list.

## What to expect (measured; notes/OBSERVATIONS.md)

On 8 runs of `psm-rerank-hek-hct116`, PSMs at 1% FDR:

| method | PSMs |
|---|---|
| MSFragger e-value | 89,693 |
| MS2Rescore full (MS2PIP + DeepLC) | 128,211 |
| **per-run, lab features** | 129,041 |
| **per-run, lab + embedding** | **+0.53%** on top (3 seeds, over a random-spectrum control) |
| global model, lab + embedding | ~105.7k |

On 16 held-out HCT116 runs (1,867,364 spectra; the global model was trained on the 8 runs
above and never saw them), PSMs at 1% FDR:

| method | PSMs |
|---|---|
| MSFragger e-value | 329,807 |
| global model (`--mode global --model Gaolaboratory/iona-rerank-400m`) | 411,607 |
| per-run, lab features (`--no-embedding`) | 515,674 |
| **per-run, lab + embedding** | **523,720** (+8,046, +1.6%) |

Per-run is much stronger than the global model; use global only when a run cannot be
trained on (e.g. very few spectra).

## Notes and caveats

- Spectra above 512 peaks are reduced to their 512 most intense peaks for the encoder.
- The embedding is weakest on low-resolution (ion-trap) MS2; gains there are small.
- The lab's feature columns vary slightly by run (dynamic `mod_count_*`); missing columns are
  filled (0 for mod counts, the training median otherwise).
- Tests: `pytest tests/test_psm_rerank.py tests/test_rerank_r4.py`.
