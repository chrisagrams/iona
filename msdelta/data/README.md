# `msdelta/data/` — spectra in, tensors out

Raw centroided peak lists → padded, masked model inputs. Also holds the chemical mass tables that
the evaluation code tests the model against.

| file | contents |
| --- | --- |
| `processing.py` | `MSDeltaProcessor` (a HF `FeatureExtractionMixin`) and `MSDeltaDataCollatorForPreTraining` |
| `loading.py` | Dataset resolution (local Parquet or the Hub), preprocessing `.map`, and `collate_preprocessed`; plus `charge_index` / `precursor_mz` parsed from the `peptide_charge` label |
| `chemistry.py` | `pyteomics`-derived constants only, no functions: `RESIDUE_MASSES`, `RESIDUES_AA20`, `ISOTOPES`, `NEUTRAL_LOSSES`, `WATER_MASS`, `PROTON_MASS` |

> Named `loading.py` rather than `datasets.py` so it does not visually shadow the `datasets`
> library it imports.

## The preprocessing contract

`MSDeltaProcessor._process_one` is the single place where a spectrum is normalized, and every
downstream assumption traces back to it:

1. Validate — finite, non-negative, equal lengths, at least one positive intensity.
2. Drop peaks below `intensity_threshold_frac` (0.01) × base peak.
3. Keep the top `max_peaks` (150) **by intensity, then restore m/z order**.
4. `log_intensity = log1p(i) / max(log1p(i))` → ∈ (0, 1]. *This is the token content.*
5. `labels = i / Σi` → relative abundance. *This is the prediction target.*
6. Return the surviving indices so callers can carry per-peak annotations through the filter.

Step 6 is what makes `process_denoising_example` correct: it re-indexes the noise labels by
`selected` so peak↔label alignment survives thresholding and top-K.

## Two processors per run

Pretraining and the denoising probe use **different** processor settings, built separately in
`train/cli.py`:

| | pretraining | denoising probe |
| --- | --- | --- |
| `max_peaks` | 150 | 1024 |
| `intensity_threshold_frac` | 0.01 | 0.0 |

The denoising probe keeps everything — noise peaks are the thing being classified, so
thresholding them away would defeat the task. Its 1024-peak spectra are why
`eval/denoising.py` batches by a **peak-pair budget** instead of a fixed batch size: the padded
attention bias is `batch × max_len²`.

## Masking

`MSDeltaDataCollatorForPreTraining` pads to the longest spectrum in the batch and samples
`round(len × mask_ratio)` positions per row (at least `min_masked`). The dataclass default is
`0.15`; **every shipped `training.args` uses `0.50`.**

Masking replaces the peak's *intensity* token with the learned `mask_token` — but the masked
peak's m/z is still visible to every other peak through the attention bias. The task is therefore
"given where this peak sits in mass space relative to everything else, how big is it?"

## Gotcha

`loading._preprocess_example` swallows processor `ValueError`s into an empty spectrum, which
`build_preprocessed_dataset` then filters out. Malformed rows vanish silently — if your dataset
shrinks unexpectedly, that is where it went.
