# Data

This directory holds only preparation scripts and this registry. The data itself lives on the Hugging
Face Hub or on Aurora's `/flare` (`$S` = `/lus/flare/projects/UIC-HPC/khuss/msdelta`); nothing under
`data/` other than `*.py`, `*.sh` and `*.md` is tracked.

| script | what it does |
|---|---|
| `prepare_eval_splits.sh` | ms-contrastive-100k test / validation / train-10k evaluation splits (flattened, preprocessed, replicate-corpus peptides excluded) and the 5k validation subset |
| `subsample_prepared.py` | shrink a prepared split to >= N experimental spectra, keeping whole peptide groups |
| `prepare_massive_kb.py` | MassIVE-KB (`chrisagrams/massive_kb_v1_shuffled`) -> contrastive training data with every evaluation peptide removed; run by `pbs/prepare_massive_kb.pbs` (see below) |

## Training and fine-tuning datasets (Hugging Face Hub)

| dataset | used for |
|---|---|
| `chrisagrams/ms-denoise-100k` | denoising fine-tuning and its test set (per-peak noise labels) |
| `chrisagrams/ms-contrastive-100k` | contrastive fine-tuning, alignment (peptide encoder), in-distribution retrieval evaluation |
| `chrisagrams/ms2-peptide-replicate-retrieval` | the replicate corpus: stage 1 of the two-stage contrastive recipe |
| `Gaolaboratory/psm-rerank-hek-hct116` | PSM rescoring (HEK / HCT116 search results, pinned revision in `msdelta.rescoring`) |

Cached under `$S/huggingface` (`HF_HOME`); jobs run with `HF_HUB_OFFLINE=1`.

## MassIVE-KB for contrastive training (`$S/data/massive-kb-*`, C18-C)

`prepare_massive_kb.py` has three subcommands; `pbs/prepare_massive_kb.pbs` runs them on a
compute node (tests, then exclusion, then prepare, then check). Card: `notes/C18_prepare_card.md`.

- `exclusion --out $S/data/massive-kb-exclusion`: every peptide of every evaluation set (ms-contrastive-100k
  validation + test, the replicate corpus, `nine_oodval20k`, `noble_mouse20k`, `noble_human20k`,
  `nine_yeast` (canonical), `c11_cap20`, `c14_hct116_20k`) with per-source counts and the sha256 of
  each prepared file (`exclusion.json`, `exclusion_counts.json`).
- `prepare --out $S/data/massive-kb-contrastive --exclusion ...`: reads the hub-cache parquet shards
  (revision `891f42f7`) directly, one process per shard. Per spectrum: parse `SEQ_z`; drop it if its
  sequence (mods stripped, I/L collapsed by default: `--no-il-collapse`) is in any eval set; map the
  modification notation to ours (`C[+57.021]` -> `C[57.0215]`, `M[+15.995]` -> `M[15.9949]`,
  `N/Q[+0.984]` -> `[0.9840]`, N-terminal `[+42.011]`/`[+43.006]`/`[-17.027]` -> `[42.0106]`/`[43.0058]`/
  `[-17.0265]`; any other token drops the spectrum and is counted by its exact string); precursor m/z
  is THEORETICAL (none is stored), so filter-failure analyses are empty on this data; spectra over
  512 peaks are dropped, as training does; `log_intensity` = log1p(int)/max, stored processed (float32),
  like the prepared eval sets. Splits: `--split-policy peptide` (default) re-splits by a hash of the
  I/L-collapsed sequence (3.3% validation, 3.3% test), so the splits are peptide-disjoint like
  ms-contrastive-100k's; `source` keeps the source's spectrum-level splits.
  `--shards N` / `--limit N` make small dry runs; `--resume` continues `<out>.partial`; an existing
  output is never overwritten.
- `check <out>`: counts vs manifest, schema, no excluded sequence, split disjointness, group sizes,
  processed values on a sample.

Output (`<out>/`): `{train,validation,test}/<srcsplit>-<shard>.parquet` (columns `analyte_id`,
`peptide`, `charge`, `precursor`, `mz`, `source`, `log_intensity`, one spectrum per row, grouped
downstream by `analyte_id` = peptide_key), `group_sizes.parquet`, `overlap_report.{json,md}`
(how many MassIVE-KB spectra / sequences each eval set removes), `exclusion.json`, `manifest.json`
(source revision, counts, drops by reason, unknown tokens, group-size summary, code commit),
`_shards/` (per-shard stats). Load with
`datasets.load_dataset("parquet", data_files={"train": f"{out}/train/*.parquet", ...})`; the
contrastive trainer does not read this format yet (it needs a `--dataset_format` for it).

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
