# Data

This directory holds only preparation scripts and this registry. The data itself lives on the Hugging
Face Hub or on Aurora's `/flare` (`$S` = `/lus/flare/projects/UIC-HPC/khuss/msdelta`); nothing under
`data/` other than `*.py`, `*.sh` and `*.md` is tracked.

| script | what it does |
|---|---|
| `prepare_eval_splits.sh` | ms-contrastive-100k test / validation / train-10k evaluation splits (flattened, preprocessed, replicate-corpus peptides excluded) and the 5k validation subset |
| `subsample_prepared.py` | shrink a prepared split to >= N experimental spectra, keeping whole peptide groups |

## Training and fine-tuning datasets (Hugging Face Hub)

| dataset | used for |
|---|---|
| `chrisagrams/ms-denoise-100k` | denoising fine-tuning and its test set (per-peak noise labels) |
| `chrisagrams/ms-contrastive-100k` | contrastive fine-tuning, alignment (peptide encoder), in-distribution retrieval evaluation |
| `chrisagrams/ms2-peptide-replicate-retrieval` | the replicate corpus: stage 1 of the two-stage contrastive recipe |
| `Gaolaboratory/psm-rerank-hek-hct116` | PSM rescoring (HEK / HCT116 search results, pinned revision in `msdelta.rescoring`) |

Cached under `$S/huggingface` (`HF_HOME`); jobs run with `HF_HUB_OFFLINE=1`.

## Prepared evaluation data (`$S/eval-data`)

| directory | rows | built by |
|---|---|---|
| `ms-contrastive-100k-test-mp512` | 35,734 | `prepare_eval_splits.sh` |
| `ms-contrastive-100k-validation-mp512` | 35,654 | `prepare_eval_splits.sh` |
| `ms-contrastive-100k-train10k-mp512` | 35,731 | `prepare_eval_splits.sh` (ABTT / PCA fit sample) |
| `ms-contrastive-100k-validation-mp512-sub5k` | >= 5,000 experimental | `subsample_prepared.py` |
| `nine-species-train-fit25k` | 25k | ABTT fit sample from the 8 non-yeast nine-species train spectra |
| `nine-species-noble/nine-species-balanced.zip` | – | Noble lab nine-species (Zenodo 10.5281/zenodo.12819175), raw download |

## Transfer benchmarks (`$S/baselines/<name>/{prepared,export}`)

Built by the benchmark builders in `baselines_wip/` (kept there with the baselines):

| directory | spectra | source | builder |
|---|---|---|---|
| `c11_cap20` (HEK) | 27,637 | `Gaolaboratory/psm-rerank-hek-hct116` confident PSMs | `c11_build.py` |
| `c14_hct116_20k` | 20,002 | same, HCT116 runs, trimmed to 512 peaks | `c11_build.py` (`C11_TRIM=1`) |
| `nine_yeast` (**canonical yeast test set**, K76-C: always this one; `CANONICAL.txt` with sha256 alongside), `nine_yeast20k` (older figures only) | 86,184 / 20,019 | `InstaDeepAI/ms_ninespecies_benchmark` (yeast test) | `nine_build.py` |
| `nine_oodval20k` | 20,004 | same, 8 non-yeast species (train split): OOD validation | `nine_build.py` |
| `noble_mouse20k`, `noble_human20k` | 20,003 / 20,000 | Noble nine-species-balanced, Mus musculus / H. sapiens | `noble_build.py` (`NOBLE_SPECIES=...`) |

## Models and runs

| location | content |
|---|---|
| `/flare/UIC-HPC/khuss/msdelta/pretrained/msdelta-<size>-production-01-checkpoint-<step>` | pretrained encoders (25M-400M, checkpoints 10k-540k) |
| `$S/runs/sweep-<arm>-<job>/final` | fine-tuned models (every sweep arm); released ones are on the Hub as `Gaolaboratory/iona-*` |
| `$S/shelf/` | code shelved from this repository (`portable_eval/`, the first Hub peptide-embedder module, i.e. the peptide encoder's standalone loader) |
| `$S/shelf/data-synthetic/` | 4 untracked parquet shards (3,000 spectra each) that sat in `data/synthetic/`; never used by our code, likely a remnant from master (shelved 2026-09-27, K4-S, SHA256SUMS alongside) |
