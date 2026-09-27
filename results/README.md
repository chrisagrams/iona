# Results

    raw/         eval outputs exactly as the jobs wrote them -- never hand-edited
      finetune/{align,contrastive,denoise}/   per-model / per-eval JSONs, shard logs, text dumps
      rerank/psm/                             PSM rescoring outputs
    processed/   regenerated from raw/ by the scripts in sweeps/ and paper/
      figures/   PNGs (+ the CSV behind each summary figure), one folder per track; index in figures/README.md
      tables/    aggregated CSVs and SUMMARY_TABLES.md

Training run directories, checkpoints and embeddings are not here: they stay on /flare at
`/lus/flare/projects/UIC-HPC/khuss/msdelta/{runs,embeddings}`.

To change a number, rerun the eval (it writes to raw/) and then the packaging/plotting script
(`sweeps/package_*.py`, `sweeps/plot_*.py`, `sweeps/summarise_*.py`), which rewrites processed/.
Exception: the `raw/finetune/denoise/*.txt` grid tables are written by `sweeps/summarise_denoise.py` from the
sweep run directories on /flare, not from anything in this repo, so they are the committed record and live in raw/.
