"""R1: do the encoders that win on contrastive improve the reranker? PLAN.md R1.

    python sweeps/make_r1.py

The reranking chain is teacher -> alignment student -> embedding_cosine -> rescorer. The
last measurement of it (5 paired seeds) found the embedding COST 0.109 Hit@1 against the
feature-only rescorer's 0.889, but its teacher was sweep-lr5e4_kl10_t02-8840715 -- one of
the first contrastive runs, trained at a batch of four with the recipe since shown to be
barely better than not training. The current encoders are ~2.3x better on MAP@100.

This writes one alignment config per teacher, IDENTICAL to configs/finetune-align-
contrastive except for the teacher, the run name and output dir, so any change in the
rescoring result is the teacher. Two teachers, the best seed-0 cell at each end of the
scale range (220k, batch width 64):

    r1-align-050m   50m  t0.005   MAP@R 0.748
    r1-align-400m   400m t0.003   MAP@R 0.821

Run each with pbs/aurora-finetune.pbs (MODULE=msdelta.finetune_align, XPUS_PER_HOST=1,
PRECOMPUTE_CACHE set), then pbs/rescoring.pbs on the cache and the student.
"""
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
RUNS = "/lus/flare/projects/UIC-HPC/khuss/msdelta/runs"
BASE = REPO / "configs" / "finetune-align-contrastive" / "training.args"
TEACHERS = {
    "050m": f"{RUNS}/sweep-s050m_t0005_pk064_seed0-8856643/final",
    "400m": f"{RUNS}/sweep-s400m_t0003_pk064_seed0-8857336/final",
}


def main() -> int:
    tokens = BASE.read_text().split()
    for tag, teacher in TEACHERS.items():
        over = {"--pretrained_path": teacher,
                "--run_name": f"r1-align-{tag}",
                "--output_dir": f"./runs/r1-align-{tag}"}
        lines = [f"{f} {over.get(f, v)}" for f, v in zip(tokens[::2], tokens[1::2])]
        out = REPO / "configs" / f"r1-align-{tag}"
        out.mkdir(parents=True, exist_ok=True)
        (out / "training.args").write_text("\n".join(lines) + "\n")
        (out / "DESCRIPTION.md").write_text(
            f"R1 ALIGNMENT STUDENT, teacher {Path(teacher).parent.name}. Identical to "
            f"configs/finetune-align-contrastive except the teacher, run name and output "
            f"dir, so a change in the rescoring result is attributable to the teacher. The "
            f"student maps a peptide sequence to the teacher's spectrum embedding; its "
            f"cosine to a spectrum becomes the rescorer's embedding_cosine feature. "
            f"PLAN.md R1.\n")
        print(f"  wrote {out.relative_to(REPO)}/training.args  (teacher {Path(teacher).parent.name})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
