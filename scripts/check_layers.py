"""Check that every encoder layer actually moved during training.

A run can finish cleanly, report a falling loss, and still have most of its layers
frozen -- this repo has produced that exact failure twice (see TODO.md issues 0 and 1),
where a residual-loop mistake left 75-89% of parameters with no gradient at all.

Usage:
    # compare two checkpoints from the same run (preferred)
    python scripts/check_layers.py runs/<run>/checkpoint-1000 runs/<run>/checkpoint-5000

    # or compare one checkpoint against a fresh init of the same config
    python scripts/check_layers.py runs/<run>/final

A healthy run: every block shows a clearly non-zero relative change, of broadly similar
magnitude. A dead block reads ~0.000000.
"""

from __future__ import annotations

import os

# Must precede the torch/sklearn imports below. On an ALCF login node OpenBLAS otherwise
# tries to start one thread per core (64) and dies with "Resource temporarily
# unavailable" against the per-user process limit.
for _var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import argparse  # noqa: E402
import sys  # noqa: E402

import torch  # noqa: E402

from msdelta.model.factory import load_model_class  # noqa: E402


def load(path: str, model_class: str | None):
    cls = load_model_class(model_class)
    return cls.from_pretrained(path)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("early", help="checkpoint directory (or the only checkpoint)")
    ap.add_argument("late", nargs="?", help="later checkpoint directory")
    ap.add_argument(
        "--model_class",
        default=None,
        help="dotted path, e.g. msdelta.model.experiments.pair_stream."
        "MSDeltaPairStreamForPreTraining",
    )
    args = ap.parse_args(argv)

    late_model = load(args.late or args.early, args.model_class)
    if args.late:
        early_model = load(args.early, args.model_class)
        label = f"{args.early} -> {args.late}"
    else:
        cls = load_model_class(args.model_class)
        early_model = cls(late_model.config)  # fresh init, same config
        label = f"fresh init -> {args.early}"

    early = dict(early_model.named_parameters())
    print(f"relative parameter change, {label}\n")
    print(f"  {'module':38s} {'||delta|| / ||early||':>22s}")

    dead = []
    encoder = late_model.msdelta
    groups: list[tuple[str, torch.nn.Module]] = [("embed", encoder.embed)]
    groups += [(f"blocks[{i}]", b) for i, b in enumerate(encoder.blocks)]
    groups.append(("bias_module", encoder.bias_module))

    for name, module in groups:
        num = den = 0.0
        for pname, p in module.named_parameters():
            key = f"msdelta.{name}.{pname}".replace("blocks[", "blocks.").replace("]", "")
            if key not in early:
                continue
            num += (p.detach() - early[key].detach()).pow(2).sum().item()
            den += early[key].detach().pow(2).sum().item()
        rel = (num**0.5) / (den**0.5) if den else float("nan")
        flag = ""
        if den and rel < 1e-6:
            flag = "   <-- DEAD, never trained"
            dead.append(name)
        print(f"  {name:38s} {rel:22.6f}{flag}")

    print()
    if dead:
        print(f"PROBLEM: {len(dead)} module(s) did not move: {', '.join(dead)}")
        return 1
    print("All modules moved. (Necessary, not sufficient -- also check the loss and align/*.)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
