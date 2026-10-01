"""K156-P: P2 half-epoch Pairformer pretraining arms (K148-P (a)+(c), K150-P design; arms chosen by Claude
under the user's 2026-09-30 delegation: "schedule subsequent experiments autonomously using your best
judgement"; see notes/DECISIONS.md K156-P).

    python sweeps/make_p2.py            # write configs/p2/<arm>/
    python sweeps/make_p2.py --check

Fixed (K148/K150, approved): data = MSConsensus-100M shards 0-199 at cap 150 (drop), 13,470,623 train
spectra; validation = the Stage 0 shard (67,933); the transformer's LR schedule (lr 1.3e-4, AdamW 0.9/0.95,
wd 0.01, warmup 2000, cosine over its 540,423 steps) stopped at 0.5 epoch = 23,387 steps of global batch 576
(16 nodes x 12 tiles x micro 3) via MSDELTA_STOP_AT_STEP; bf16; mask ratio 0.5; probes off;
W&B CS_Pharm/pairformer_pretrain.
Model: the Stage 0 Pairformer (hidden 512, 10 layers, 8 heads) with the user's pair choices (K151-K153):
c_z = c_t = 32, c_o = 16, outgoing triangle multiplication only, factored write-back.
Arms: k1 (pair update every layer), k1_triattn (+ triangle attention, K148 (c)), k5 (pair update on layers
1 and 6: pair cost ~ single cost, the user's K114 balancing rule, width profile 8880628).
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "configs" / "p2"
BASE = REPO / "configs" / "stage0" / "pairformer"
S = "/lus/flare/projects/UIC-HPC/khuss/msdelta"
PAIR = {"pair_channels": 32, "pair_tri_channels": 32, "pair_opm_channels": 16, "pair_tri_mul": "outgoing",
        "pair_writeback": "outer", "pair_writeback_impl": "factored"}
ARMS = {
    "p2-cz32-k1": {},
    "p2-cz32-k1-triattn": {"pair_use_triangle_attention": True},
    "p2-cz32-k5": {"pair_update_every": 5},
    # K180-P (user-approved k sweep, 2026-10-01): same recipe, compiled on 2 nodes x micro 24 (global 576 as above);
    # fixed padding to max_peaks so torch.compile sees one shape.
    "p2-L10-k3": {"pair_update_every": 3},
    "p2-L10-k10": {"pair_update_every": 10},
    "p2-L14-k7": {"num_hidden_layers": 14, "pair_update_every": 7},
    "p2-L14-k5": {"num_hidden_layers": 14, "pair_update_every": 5},
    # K181-P: p2-cz32-k5 rerun under the K180 setup (compiled, 2 nodes x micro 24) -- does the setup alone move the loss?
    "p2-L10-k5": {"pair_update_every": 5},
    # K185-P (user-approved 2026-10-01): no pair updates -- the initial pair state is only read out (per-layer bias).
    "p2-L10-static": {"pair_update": "static"},
    "p2-L20-static": {"num_hidden_layers": 20, "pair_update": "static"},
    # K186-P (user-approved 2026-10-01): 20 layers with 1 / 2 / 4 pair updates.
    "p2-L20-k20": {"num_hidden_layers": 20, "pair_update_every": 20},
    "p2-L20-k10": {"num_hidden_layers": 20, "pair_update_every": 10},
    "p2-L20-k5": {"num_hidden_layers": 20, "pair_update_every": 5},
}
# K182-P (d) (user 2026-10-01: "Sure ok"): plain-transformer baselines under the K180 setup, matching the Pairformer
# depths run (10 / 14 / 20 layers) at P2's single-stream width: transformer-50m's config (configs/msdelta-base-50m:
# learned delta-m/z attention bias, no pair stream) with hidden 512, 8 heads, FFN 2048; P2's processor (max_peaks 150).
TBASE = REPO / "configs" / "msdelta-base-50m"
TSHAPE = {"hidden_size": 512, "num_attention_heads": 8, "intermediate_size": 2048}
TARMS = {f"p2-T-L{n}": {**TSHAPE, "num_hidden_layers": n} for n in (10, 14, 20)}
EXTRA_ARGS = {a: "--pad_to_multiple_of 150\n" for a in TARMS} | {a: "--pad_to_multiple_of 150\n" for a in ("p2-L10-k3", "p2-L10-k10", "p2-L14-k7", "p2-L14-k5", "p2-L10-k5",
                                     "p2-L10-static", "p2-L20-static", "p2-L20-k20", "p2-L20-k10", "p2-L20-k5")}
TRAINING = """--config_name configs/p2/{arm}
--processor_name_or_path configs/p2/{arm}
--output_dir ./runs/{arm}
--run_name {arm}
--dataset_repo_id {S}/data/p2-cap150-half/raw
--dataset_train_split train
--dataset_validation_split validation
--preprocessing_num_workers 24
--max_peaks 150
--mask_ratio 0.50
--per_device_train_batch_size 3
--per_device_eval_batch_size 32
--gradient_accumulation_steps 1
--dataloader_num_workers 6
--learning_rate 1.3e-4
--adam_beta1 0.9
--adam_beta2 0.95
--lr_scheduler_type cosine
--warmup_steps 2000
--max_steps 540423
--weight_decay 0.01
--max_grad_norm 1.0
--bf16 true
--gradient_checkpointing false
--seed 0
--logging_steps 50
--logging_first_step true
--eval_strategy steps
--eval_steps 1000
--save_strategy steps
--save_steps 5000
--save_total_limit 3
--remove_unused_columns false
--report_to wandb
--wandb_project pairformer_pretrain
--probe_execution off
--bias_curve_steps 0
--probe_steps 0
--denoise_steps 0
--retrieval_steps 0
"""


def arms() -> dict[str, dict[str, str]]:
    base = json.loads((BASE / "config.json").read_text())
    out = {}
    tbase = json.loads((TBASE / "config.json").read_text())
    for arm, over in [*ARMS.items(), *TARMS.items()]:
        cfg = {**tbase, **over} if arm in TARMS else {**base, **PAIR, **over}
        out[arm] = {"config.json": json.dumps(dict(sorted(cfg.items())), indent=2) + "\n",
                    "preprocessor_config.json": (BASE / "preprocessor_config.json").read_text(),
                    "training.args": TRAINING.format(arm=arm, S=S) + EXTRA_ARGS.get(arm, "")}
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    cli = ap.parse_args()
    want = arms()
    if cli.check:
        bad = [f"{a}/{f}" for a, files in want.items() for f, text in files.items()
               if not (OUT / a / f).exists() or (OUT / a / f).read_text() != text]
        print("stale or missing: " + str(bad) if bad else f"{len(want)} arms match")
        return 1 if bad else 0
    for arm, files in want.items():
        shutil.rmtree(OUT / arm, ignore_errors=True)
        (OUT / arm).mkdir(parents=True)
        for f, text in files.items():
            (OUT / arm / f).write_text(text)
    print(f"wrote {len(want)} arms to {OUT.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
