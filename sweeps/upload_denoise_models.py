"""Upload the best denoise fine-tune per scale to Gaolaboratory/iona-denoise-{scale}, PRIVATE.

    source pbs/load_keys.sh    # HF_TOKEN, never printed
    .venv/bin/python sweeps/upload_denoise_models.py --dry-run
    .venv/bin/python sweeps/upload_denoise_models.py

SELECTION (2026-09-23): per scale, the arm with the best VALIDATION AUPRC -- the metric
load_best_model_at_end already selects on -- across every checkpoint-ladder arm we have
(3 seeds x the pretraining checkpoints available at that scale). Test metrics are
reported, never selected on. Within a scale the top arms differ by < 0.001 test AUROC.

Uploads each arm's final/ as is (weights, config, processor, the remote code), minus
training_args.bin (a pickle of our local training arguments), plus a model card.
Refuses to touch a repo that already exists.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

RUNS = Path("/lus/flare/projects/UIC-HPC/khuss/msdelta/runs")
ORG = "Gaolaboratory"
PICKS = {
    "50m": "sweep-50m_ck540k_seed1-8856549",
    "100m": "sweep-100m_ck540k_seed2-8856549",
    "200m": "sweep-200m_ck330k_seed2-8850494",
    "400m": "sweep-400m_ck220k_seed1-8850494",
}
SKIP = ["training_args.bin"]


def param_count(final: Path) -> int:
    """From the safetensors header only; no tensors are read."""
    raw = (final / "model.safetensors").read_bytes()[:8]
    n = int.from_bytes(raw, "little")
    with open(final / "model.safetensors", "rb") as f:
        f.seek(8)
        header = json.loads(f.read(n))
    total = 0
    for name, meta in header.items():
        if name == "__metadata__":
            continue
        count = 1
        for dim in meta["shape"]:
            count *= dim
        total += count
    return total


def best_val_auprc(arm: Path) -> float:
    log = (arm / "logs" / "train.log").read_text(errors="ignore")
    return max(float(x) for x in re.findall(r"'eval_auprc': '?([0-9.]+)", log))


def card(scale: str, arm: Path, params: int) -> str:
    m = re.match(r"sweep-(\d+m)_ck(\d+)k_seed(\d+)-(\d+)$", arm.name)
    ckpt = {"540": "540,423", "430": "430,000", "330": "330,000", "220": "220,000",
            "120": "120,000", "10": "10,000"}[m[2]]
    test = json.loads((arm / "test_results.json").read_text())
    return f"""---
library_name: transformers
tags: [mass-spectrometry, proteomics, denoising, msdelta]
---

# iona-denoise-{scale}

Per-peak noise classifier for tandem mass spectra: an MSDelta encoder ({scale} scale,
{params / 1e6:.1f}M parameters including the head) pretrained on
`chrisagrams/MSConsensus-100M` for {ckpt} steps, then fine-tuned with a token-classification
head on `chrisagrams/ms-denoise-100k`. One logit per peak; **noise is the positive class
(label 1)**. Spectra above 512 peaks are out of scope (they were dropped, not truncated,
in training).

## Results (held-out test split, peptide-disjoint from train)

| metric | value |
|---|---|
| AUROC | {test['test_auroc']:.4f} |
| AUPRC | {test['test_auprc']:.4f} |
| F1 (threshold logit >= 0) | {test['test_f1']:.4f} |
| per-spectrum AUROC (mean) | {test['test_auroc_per_spectrum']:.4f} |

Selected as the best of 3 seeds x the available pretraining checkpoints at this scale by
**validation** AUPRC (best: {best_val_auprc(arm):.4f}); the test split was never used
for selection.

## Training

lr 2e-4 (encoder at 0.5x), 4 epochs, head width 512, effective batch 12 (12 tiles,
DeepSpeed ZeRO-2), bf16, cosine schedule, best checkpoint by validation AUPRC. Seed
{m[3]}. Run `{arm.name}`.

## Usage

```python
from transformers import AutoConfig, AutoModelForTokenClassification, AutoProcessor
repo = "{ORG}/iona-denoise-{scale}"
model = AutoModelForTokenClassification.from_pretrained(repo, trust_remote_code=True)
processor = AutoProcessor.from_pretrained(repo, trust_remote_code=True)
```

Internal release; not yet reviewed for publication.
"""


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--scales", default=",".join(PICKS))
    cli = ap.parse_args()
    from huggingface_hub import HfApi

    api = HfApi()
    for scale in cli.scales.split(","):
        arm = RUNS / PICKS[scale]
        final = arm / "final"
        repo = f"{ORG}/iona-denoise-{scale}"
        files = sorted(p.name for p in final.iterdir() if p.name not in SKIP)
        params = param_count(final)
        print(f"{repo}  <-  {arm.name}/final  ({params / 1e6:.1f}M params)  files: "
              f"{', '.join(files)}", flush=True)
        if cli.dry_run:
            print(card(scale, arm, params).split("## Training")[0][-420:])
            continue
        if api.repo_exists(repo):
            raise SystemExit(f"{repo} already exists; refusing to overwrite")
        api.create_repo(repo, repo_type="model", private=True)
        api.upload_folder(repo_id=repo, folder_path=str(final), ignore_patterns=SKIP,
                          commit_message=f"{arm.name} final/ (best validation AUPRC)")
        api.upload_file(repo_id=repo, path_in_repo="README.md",
                        path_or_fileobj=card(scale, arm, params).encode(),
                        commit_message="model card")
        info = api.model_info(repo)
        print(f"  uploaded; private={info.private}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
