"""Generate contrastive fine-tune arms.

    python sweeps/make_contrastive_grid.py
    python sweeps/make_contrastive_grid.py --check

Twelve arms, one per tile, one node.

Chosen after a smoke test in which the contrastive term sat at its CHANCE value for
three epochs and the embedding space did not move (`clean` 0.010 -> 0.000, margin
+0.0255 -> +0.0270). That test cannot distinguish "the objective does not work here" from
"743 steps at lr 2e-5 on 2,000 spectra was never going to move a 50M encoder", so these
axes are picked to separate exactly those explanations.

`learning_rate` is the prime suspect and gets three levels spanning 25x. `kl_weight` gets
0 and 10: zero is the clean test of whether the contrastive term can separate anything
when nothing holds it back, and 10 sits between the 1.0 that was inert (0.3% of the loss)
and the 100 that may be a leash. `temperature` gets the SupCon default and a softer
value, because 0.07 is tuned for batches with thousands of negatives and we have 44.

Batch is NOT an axis: DeltaMZBias is O(batch * peaks^2) and the tile caps it at 8 spectra
at 512 peaks. Decoupling negatives from memory needs GradCache, which is the follow-up if
these arms show the objective working at all.
"""

from __future__ import annotations

import argparse
import itertools
import shutil
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
TEMPLATE = REPO / "configs" / "finetune-contrastive-50m" / "training.args"
OUT = REPO / "configs" / "sweep-contrastive"
STAMP = OUT / ".template"
RUN_PREFIX = "v2_con50m-"

LEARNING_RATES = ("2e-5", "1e-4", "5e-4")
KL_WEIGHTS = ("0", "10")
TEMPERATURES = ("0.07", "0.2")


def arm_name(lr: str, kl: str, temperature: str) -> str:
    return (f"lr{lr.replace('-', '')}_kl{kl.replace('.', '')}"
            f"_t{temperature.replace('.', '')}")


def render_arm(lr: str, kl: str, temperature: str) -> tuple[str, str]:
    name = arm_name(lr, kl, temperature)
    overrides = {
        "--learning_rate": lr,
        "--kl_weight": kl,
        "--temperature": temperature,
        "--run_name": f"{RUN_PREFIX}{name}",
        "--output_dir": f"./runs/{RUN_PREFIX}{name}",
    }
    lines, seen = [], set()
    tokens = TEMPLATE.read_text().split()
    for flag, value in zip(tokens[::2], tokens[1::2]):
        lines.append(f"{flag} {overrides.get(flag, value)}")
        seen.add(flag)
    for flag, value in overrides.items():
        if flag not in seen:
            lines.append(f"{flag} {value}")
    return name, "\n".join(lines) + "\n"


def description(lr: str, kl: str, temperature: str) -> str:
    shared = TEMPLATE.parent / "DESCRIPTION.md"
    lead = " ".join(shared.read_text().split()) + " " if shared.exists() else ""
    regulariser = ("KL regularisation DISABLED, so this is the clean test of whether the "
                   "contrastive term can separate replicates with nothing holding the "
                   "encoder back" if kl == "0" else
                   f"KL weight {kl}, between the 1.0 that was 0.3% of the loss and the "
                   f"100 that may prevent the encoder moving at all")
    return (f"{lead}GRID ARM: lr={lr}, {regulariser}, temperature {temperature} "
            f"({'the SupCon default, tuned for batches with thousands of negatives' if temperature == '0.07' else 'softer than the default, which suits the 44 negatives a batch of 8 provides'}). "
            f"The question these arms answer is whether the flat contrastive loss in the "
            f"smoke test meant the objective does not work or that the run was too small "
            f"to move a 50M encoder.\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--clean", action="store_true")
    cli = parser.parse_args()

    combos = list(itertools.product(LEARNING_RATES, KL_WEIGHTS, TEMPERATURES))
    if cli.check:
        stale = [n for n, text in (render_arm(*c) for c in combos)
                 if not (OUT / n / "training.args").exists()
                 or (OUT / n / "training.args").read_text() != text
                 or not (OUT / n / "DESCRIPTION.md").exists()]
        if stale:
            print(f"  {len(stale)} stale or missing: {', '.join(stale[:6])}")
            print(f"\nregenerate: python {Path(__file__).name} --clean")
            return 1
        print(f"  {len(combos)} arms match {TEMPLATE.relative_to(REPO)}")
        return 0

    if cli.clean and OUT.exists():
        shutil.rmtree(OUT)
    for combo in combos:
        name, text = render_arm(*combo)
        (OUT / name).mkdir(parents=True, exist_ok=True)
        (OUT / name / "training.args").write_text(text)
        (OUT / name / "DESCRIPTION.md").write_text(description(*combo))
    STAMP.write_text(f"{TEMPLATE.relative_to(REPO)}\ncontrastive\n")
    print(f"arms={len(combos)} written under {OUT.relative_to(REPO)}/")
    for combo in combos:
        print(f"  {arm_name(*combo)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
