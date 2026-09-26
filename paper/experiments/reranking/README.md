# Reranking: extra classification power from our embeddings

Rescoring database-search results: for each spectrum, MSFragger returns up to 10 candidate
peptides; a classifier rescores every candidate from its features, the best candidate per spectrum
is kept, and targets are counted at 1% FDR (target-decoy). The question is whether adding our
spectrum/peptide embedding similarity as a feature lets the classifier identify more spectra.

## Files

| file | content |
|---|---|
| `R_embedding_gain.png`, `R_embedding_gain.csv` | % more PSMs at 1% FDR from adding the embedding features vs adding a null control |
| `R_benchmark.png`, `R_benchmark.csv` | PSMs at 1% FDR for MSFragger, MS2Rescore and Iona-rerank, with a grid showing which features each method uses |
| `plot_reranking.py` | regenerates both figures from the CSVs (`python plot_reranking.py`; matplotlib + numpy) |

`R_embedding_gain.csv`: one row per (dataset, embedding, feature set, arm, seed) with the PSMs of the
classifier without the added features (`base_psms`), with them (`psms`), and the difference
(`gain_psms`, `gain_pct`). `R_benchmark.csv`: one row per method (and seed for ours). `source` names
the result file each number was read from.

## Data

- `Gaolaboratory/psm-rerank-hek-hct116` (revision `87f5c27`): MS2 spectra with MSFragger's top-10
  candidates and search scores, per run; lab feature tables from the same repository (`features/`,
  revision `a6df794`).
- **8 runs**: 6 HEK293 runs (05032011-7, 0718-5, 09122011-5, HEK-0628-5, HEK-U100ug-0408-5,
  HEK-U100ug-V2-5) and 2 HCT116 runs (HILIC14, HILIC15); 332,822 spectra, 2,818,394 candidates.
- **HCT116, 18 runs**: all 18 HCT116 runs, 1,954,190 spectra; none of our models were trained on
  this data.

## Methods

- **Iona-rerank** (our per-run classifier): Percolator-style rescoring trained separately on each run: 3-fold
  cross-validation by spectrum, iterative target/decoy labels, linear model. PSMs at 1% FDR are
  counted over all runs.
  - **rich features**: the lab's per-candidate feature table: scores from four search engines (MSFragger,
    Comet, SEQUEST, ProLuCID), fragment-ion coverage/intensity, cross-candidate context and peptide features.
    Its predicted-spectrum and retention-time columns are empty in the published table and are not used.
  - **MSFragger features**: MSFragger's search scores alone.
- **+ Iona embedding**: the cosine similarity between the spectrum's embedding and the candidate peptide's
  embedding, plus four features of that cosine relative to the spectrum's other candidates (rank,
  difference to the best other candidate, z-score within the spectrum, gap between the top two).
  Spectrum embeddings come from the fine-tuned spectrum encoder (the teacher); peptide embeddings from
  the peptide embedder trained on it (the student).
  - **Iona embedding (50M)**: fine-tuned 50M spectrum encoder (step 600) and its peptide student.
  - **Iona embedding (400M)**: fine-tuned 400M spectrum encoder (end of epoch, seed 0) and its peptide student
    (released as `Gaolaboratory/iona-peptide-embedder-400m`).
- **+ null control**: the same five features computed with the embedding of a random other spectrum
  in place of the query's. It adds the same number of features with the same distributions but no
  information about the match, so any gain it gives is an artefact of adding features.
- **MS2Rescore** (v4.0.2): the same 8 runs, pooled 1% FDR. "+ Iona embedding" adds the five Iona embedding
  features as extra rescoring columns (MSFragger features + Iona embedding: 102,751 vs 100,974 plain and 101,007
  with the null control, one run each). "MSFragger features" uses MSFragger's search
  features only; "+ MS2PIP + DeepLC" adds MS2PIP (predicted fragment intensities), DeepLC (predicted
  retention time) and MS2Rescore's own fragment-match features. MS2Rescore does NOT use the rich features.
- **Seeds**: our classifier uses 3 seeds (fold assignment and initialisation) on the 8 runs; the
  18-run HCT116 result is one seed.

## Results

Embedding minus null control (the net gain), PSMs at 1% FDR, 8 runs, per seed:

| features | Iona embedding (50M) | Iona embedding (400M) |
|---|---|---|
| MSFragger features | +1,495 / +1,215 / +1,186 (+1.3%) | +2,705 / +2,373 / +2,386 (+2.5%) |
| rich features | +398 / +738 / +473 (+0.4%) | +564 / +924 / +544 (+0.5%) |

HCT116, 18 runs, Iona embedding (400M): +20,763 (+5.6%) with MSFragger features, +8,624 (+1.65%) with rich
features.

- The null control adds nothing (-0.1% to +0.2%); the Iona embedding adds identifications in every setting.
- The better teacher (400M) gives the larger gain; the gain is largest on the unseen HCT116 runs and
  with MSFragger features, and persists on top of the rich features; Iona-rerank with rich features alone
  matches MS2Rescore with MS2PIP + DeepLC (128,909 vs 128,211) using different information.

## Controls and caveats

- With shuffled target/decoy labels the classifier finds 0 PSMs at 1% FDR.
- The embedding cosine alone barely separates targets from decoys (AUROC 0.52-0.53; null 0.50), so the
  gain is not a leak of the target/decoy label.
- Across the 3 seeds, Iona-rerank without the embedding ranges 128,751-129,041 PSMs (rich features)
  and 99,154-99,405 (MSFragger features); gains are compared within each seed.
- Oktoberfest was run on the 6 HEK runs only and is not comparable with these 8-run numbers, so it is
  not shown.
- MS2Rescore full with the embedding features added (one run): 128,909 PSMs, against 128,658 with the
  null control, a net +251 (+0.2%). On the 6 HEK runs the gain is clean (+257 vs +6 for the null); on
  the 2 HCT116 runs the null gained more than the embedding (+452 vs +343), i.e. within MS2Rescore's
  sensitivity to added features on those small runs. `R_benchmark.png` shows the embedding row;
  the null-control row is omitted there (it is noise-dominated, see above). The 18-run HCT116 MS2Rescore
  comparison is pending. (MS2Rescore + MS2PIP + DeepLC + Iona embedding and Iona-rerank
  with rich features both read 128,909: the first is one run, the second the mean of 129,041 / 128,751 / 128,934.)

Provenance: msdelta repository, `sweeps/package_rerank.py` (from `results/rerank/psm/` and
`baselines_wip/results_ms2rescore.json`, `baselines_wip/results_ms2rescore_emb8_a2.json`,
`baselines_wip/results_ms2rescore_searchonly8.json`).
