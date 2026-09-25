"""A1: a peptide encoder aligned to the spectrum encoder, trained on ms-contrastive-100k.
PLAN.md A1.

    python sweeps/make_a1.py

The only aligned peptide encoder so far (R1's student) trained on ~855 peptides of the
replicate corpus and was scored on 94. This one trains on ms-contrastive-100k's ~88k
train peptides (257k experimental spectra, replicate-corpus peptides excluded, same
flags as C7 so teacher and student see the same rows).

TEACHER: the best spectrum encoder on ms-contrastive-100k's test split so far,
sweep-s400m_t0002_pk256_ep12_seed0-8857593 (experimental MAP@R 0.711; C1 recipe, trained
on the replicate corpus). A C7 teacher trained on this corpus replaces it once C7 lands;
the config differs only in --pretrained_path so the two students are comparable.

Otherwise configs/finetune-align-contrastive, except what 25x more data forces:
  - batch 64 per device (was 4): the loss is per-pair L2, so batch size changes step count
    and noise, not the objective; 4 would be 64k steps per epoch.
  - 3 epochs (was 10): ~12k steps at batch 64, about R1's step count.
  - eval/save every 1000 steps (was 200) and eval batch 64: 26k validation rows.

Run with pbs/aurora-finetune.pbs (MODULE=msdelta.finetune_align, XPUS_PER_HOST=1,
PRECOMPUTE_CACHE set).
"""
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
RUNS = "/lus/flare/projects/UIC-HPC/khuss/msdelta/runs"
BASE = REPO / "configs" / "finetune-align-contrastive" / "training.args"
TEACHERS = {
    "400m-c1": f"{RUNS}/sweep-s400m_t0002_pk256_ep12_seed0-8857593/final",
    # C7 best-first 50m, continued on ms-contrastive-100k, step 600 of 1062, seed 1:
    # 0.831 exp MAP@R on the 100k test (best of 3 seeds; binned cosine 0.730).
    "050m-c7s600": f"{RUNS}/sweep-cont050m_ep01_seed1-8860522/checkpoint-600/encoder",
    # C7 best-first 400m, FINAL (one epoch on ms-contrastive-100k), best seed on the 100k VALIDATION split.
    "400m-c7final": f"{RUNS}/sweep-cont400m_ep01_seed0-8860522/final",
    # teacher chosen automatically (teacher_downstream.sh)
    "400m-oodsel": "/lus/flare/projects/UIC-HPC/khuss/msdelta/runs/sweep-cont400m_ep01_seed1-8860522/checkpoint-600/encoder",
    # teacher chosen automatically (teacher_downstream.sh)
    "400m-frozen220k": "/flare/UIC-HPC/khuss/msdelta/pretrained/msdelta-400m-production-01-checkpoint-220000",
    # teacher chosen automatically (teacher_downstream.sh)
    "400m-c1rep": "/lus/flare/projects/UIC-HPC/khuss/msdelta/runs/sweep-s400m_t0002_pk256_ep12_seed0-8857593/final",
    # A7 lower bound: the 400m architecture at RANDOM init (seed 0), never trained.
    "400m-random": "/lus/flare/projects/UIC-HPC/khuss/msdelta/pretrained-random/msdelta-400m-random-init-seed0",
}
OVERRIDES = {
    "--dataset_repo": "chrisagrams/ms-contrastive-100k",
    "--dataset_format": "grouped",
    "--include_consensus": "false",
    "--exclude_replicate_peptides": "true",
    "--num_train_epochs": "3",
    "--per_device_train_batch_size": "64",
    "--per_device_eval_batch_size": "64",
    "--eval_steps": "1000",
    "--save_steps": "1000",
    "--wandb_project": "msdelta-finetune-align",
}


def main() -> int:
    tokens = BASE.read_text().split()
    for tag, teacher in TEACHERS.items():
        over = dict(OVERRIDES, **{"--pretrained_path": teacher,
                                  "--run_name": f"a1-align-100k-{tag}",
                                  "--output_dir": f"./runs/a1-align-100k-{tag}"})
        lines, seen = [], set()
        for f, v in zip(tokens[::2], tokens[1::2]):
            lines.append(f"{f} {over.get(f, v)}")
            seen.add(f)
        lines += [f"{f} {v}" for f, v in over.items() if f not in seen]
        out = REPO / "configs" / f"a1-align-100k-{tag}"
        out.mkdir(parents=True, exist_ok=True)
        (out / "training.args").write_text("\n".join(lines) + "\n")
        (out / "DESCRIPTION.md").write_text(
            f"A1 ALIGNMENT STUDENT on ms-contrastive-100k, teacher "
            f"{Path(teacher).parent.name}. The student maps a peptide sequence to the "
            f"frozen teacher's spectrum embedding (L2 on normalised vectors); scored by "
            f"peptide->spectrum Hit@1. Same data flags as C7 (experimental spectra only, "
            f"replicate-corpus peptides excluded). PLAN.md A1.\n")
        print(f"  wrote {out.relative_to(REPO)}/training.args  (teacher {Path(teacher).parent.name})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
