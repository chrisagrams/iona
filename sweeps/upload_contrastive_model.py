"""Upload the best contrastive spectrum encoder per scale to Gaolaboratory/iona-contrastive-{scale},
PRIVATE.

    source pbs/load_keys.sh    # HF_TOKEN, never printed
    .venv/bin/python sweeps/upload_contrastive_model.py --scale 400m --dry-run
    .venv/bin/python sweeps/upload_contrastive_model.py --scale 400m

SELECTION (2026-09-25): C7 (two-stage contrastive) FINAL, the seed with the best
VALIDATION experimental MAP@R on ms-contrastive-100k. 400m: seed 0 (0.8639; seeds 1/2
0.8629 / 0.8634). 50m: seed 1 (0.8357; 0.8314 / 0.8342). C7 exists only at 50m and 400m.
Test numbers are reported, never selected on. Numbers are from our own evaluation of these
saved weights (eval_grouped_retrieval), not from the training log.

Uploads final/ as is (weights, config, processor, remote code) plus a model card.
Refuses to touch a repo that already exists.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

RUNS = Path("/lus/flare/projects/UIC-HPC/khuss/msdelta/runs")
RES = Path(__file__).resolve().parent.parent / "results/finetune/contrastive"
PICKS = {
    "400m": dict(run="sweep-cont400m_ep01_seed0-8860522", name="c7b_cont400m_final_seed0",
                 seed=0, arch="20 layers x 1280", width=2560, stage1_epochs=12,
                 stage1_batch="64 groups x 4 replicates", replicate_only="0.714 / 0.810"),
    "50m": dict(run="sweep-cont050m_ep01_seed1-8860522", name="c7b_cont050m_final_seed1",
                seed=1, arch="10 layers x 640", width=1280, stage1_epochs=24,
                stage1_batch="64 groups x 4 replicates", replicate_only="0.655 / 0.764"),
}


def metrics(name: str, split: str) -> dict:
    return json.loads((RES / f"grouped100k-{split}" / f"{name}.json").read_text())["metrics"]


def card(scale: str) -> str:
    p = PICKS[scale]; repo = f"Gaolaboratory/iona-contrastive-{scale}"
    v, t = metrics(p["name"], "validation"), metrics(p["name"], "test")
    row = lambda k: f"| {k} | {v[k]:.4f} | {t[k]:.4f} |"   # noqa: E731
    return f"""---
library_name: transformers
tags: [mass-spectrometry, proteomics, spectrum-embedding, contrastive, msdelta]
---

# iona-contrastive-{scale}

Spectrum embedding model: an MSDelta encoder ({scale} scale, {p["arch"]}) fine-tuned so
that tandem mass spectra of the same peptide + charge land close together. Embedding = the
masked **mean and max** of the encoder's last hidden state over the spectrum's peaks,
concatenated ({p["width"]} dims); compare with **cosine**. No projection head.

## Training

1. Self-supervised pretraining: `iona-base-{scale}` (MSDelta on `chrisagrams/MSConsensus-100M`,
   checkpoint 220,000).
2. Supervised contrastive (SupCon, temperature 0.002) on
   `chrisagrams/ms2-peptide-replicate-retrieval` for {p["stage1_epochs"]} epochs
   ({p["stage1_batch"]}), plus a KL term (weight 10) to the frozen pretrained intensity head.
3. One epoch of the same objective on `chrisagrams/ms-contrastive-100k` train (85 groups x
   3 experimental replicates; peptides of the replicate corpus excluded), lr 1e-4, seed {p["seed"]}.

## Results: `chrisagrams/ms-contrastive-100k`

Spectrum -> spectrum retrieval; a hit is a spectrum of the same peptide + charge.
`experimental` ranks experimental spectra among themselves (~25k queries); `all` also
includes the consensus spectra. Spectra above 512 peaks are out of scope.

| metric | validation | test |
|---|---|---|
{row("experimental/MAP@R")}
{row("experimental/Hit@1")}
{row("all/MAP@R")}
{row("all/Hit@1")}

Selected as the best of 3 seeds by **validation** experimental MAP@R; the test split was
never used for selection.

Reference points on the same test split and metric (experimental MAP@R / Hit@1):
binned cosine (0.1 Da) 0.730 / 0.819; PCA of binned spectra 0.723 / 0.814; pretrained
GLEAMS 0.646 / 0.746. This model trained on this dataset's train split, GLEAMS did not;
the same encoder after steps 1-2 only (never trained on this dataset) reaches
{p["replicate_only"]}.

## Usage

```python
import torch
from transformers import AutoModelForPreTraining, AutoProcessor

repo = "{repo}"
model = AutoModelForPreTraining.from_pretrained(repo, trust_remote_code=True).eval()
processor = AutoProcessor.from_pretrained(repo, trust_remote_code=True)

def embed(mz, intensity):
    x = processor(torch.as_tensor(mz, dtype=torch.float32),
                  torch.as_tensor(intensity, dtype=torch.float32), return_tensors="pt")
    with torch.no_grad():
        h = model.msdelta(mz=x["mz"], log_intensity=x["log_intensity"],
                          attention_mask=x["attention_mask"]).last_hidden_state
    m = x["attention_mask"].bool().unsqueeze(-1)
    mean = (h * m).sum(1) / m.sum(1).clamp_min(1)
    mx = h.masked_fill(~m, float("-inf")).max(1).values
    return torch.nn.functional.normalize(torch.cat([mean, mx], -1), dim=-1)
```

Internal release; not yet reviewed for publication.
"""


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scale", required=True, choices=sorted(PICKS))
    ap.add_argument("--dry-run", action="store_true")
    cli = ap.parse_args()
    run = RUNS / PICKS[cli.scale]["run"]; final = run / "final"
    repo = f"Gaolaboratory/iona-contrastive-{cli.scale}"
    files = sorted(p.name for p in final.iterdir())
    print(f"{repo}  <-  {run.name}/final  files: {', '.join(files)}", flush=True)
    text = card(cli.scale)
    if cli.dry_run:
        print(text)
        return 0
    from huggingface_hub import HfApi
    api = HfApi()
    if api.repo_exists(repo):
        raise SystemExit(f"{repo} already exists; refusing to overwrite")
    api.create_repo(repo, repo_type="model", private=True)
    api.upload_folder(repo_id=repo, folder_path=str(final),
                      commit_message=f"{run.name} final/ (best validation MAP@R of 3 seeds)")
    api.upload_file(repo_id=repo, path_in_repo="README.md", path_or_fileobj=text.encode(),
                    commit_message="model card")
    info = api.model_info(repo, files_metadata=True)
    print(f"  uploaded; private={info.private}", flush=True)
    for s in info.siblings:
        if s.rfilename == "model.safetensors":
            print(f"  model.safetensors sha256 {s.lfs.sha256 if s.lfs else '?'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
