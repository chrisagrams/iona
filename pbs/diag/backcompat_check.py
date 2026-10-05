"""Backward compatibility of the msdelta reorganisation and the standard PeptideEncoderModel, on real
models (run from the NEW tree; the OLD tree's path is given for the spectrum comparison).

    python pbs/diag/backcompat_check.py --old-repo /home/khuss/code/msdelta --out $MSDELTA_DIAG/backcompat.json

1. Peptide encoder, released Hub model (Gaolaboratory/iona-peptide-embedder-400m, local HF cache):
   new PeptideEncoderModel.from_pretrained  vs  the old loading path (PeptideEncoder built by hand,
   student_readout, sequence_encoder. prefix stripped -- as portable_eval/mouse/embed_ours.py did)
   vs  the standalone module shipped on the Hub (peptide_embedder.py from the snapshot).
2. Peptide encoder, an alignment run's raw final/ (the run behind the Hub release): new class vs old path,
   plus a save_pretrained -> from_pretrained round trip.
3. Spectrum encoder (iona-contrastive-400m, final/ of the released run): embeddings of validation
   spectra computed by the OLD tree (subprocess, old import paths) and the NEW tree must be identical.
All comparisons are max |difference| on the same inputs, CPU float32 (deterministic).
"""
import argparse
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import torch

HUB_REPO = "Gaolaboratory/iona-peptide-embedder-400m"
ALIGN_RUN = "/lus/flare/projects/UIC-HPC/khuss/msdelta/runs/v2_align-ft-a1-align-100k-400m-c7final-8866356/final"
SPECTRUM_MODEL = "/lus/flare/projects/UIC-HPC/khuss/msdelta/runs/sweep-cont400m_ep01_seed0-8860522/final"
DATA = "/lus/flare/projects/UIC-HPC/khuss/msdelta/eval-data/ms-contrastive-100k-validation-mp512"

SPECTRUM_SNIPPET = r'''
import sys, torch, numpy as np
from datasets import load_from_disk
from msdelta.modeling_msdelta import MSDeltaForPreTraining          # OLD-style import path
from msdelta.reranking import AlignmentCollator, pool_sequence
torch.manual_seed(0)
d = load_from_disk(sys.argv[1]).select(range(500))
enc = MSDeltaForPreTraining.from_pretrained(sys.argv[2]).eval()
core = getattr(enc, "msdelta", enc)
feats = [{"mz": r["mz"], "log_intensity": r["log_intensity"], "peptide": "A", "charge": 2} for r in d]
coll = AlignmentCollator(pad_spectra_to=512)
out = []
with torch.no_grad():
    for s in range(0, len(feats), 25):
        b = coll(feats[s:s + 25])
        h = core(mz=b["mz"], log_intensity=b["log_intensity"], attention_mask=b["attention_mask"]).last_hidden_state
        out.append(torch.nn.functional.normalize(pool_sequence(h, b["attention_mask"], "mean+max").float(), dim=-1))
np.save(sys.argv[3], torch.cat(out).numpy())
'''


def peptides_for_test(n=2000):
    from datasets import load_from_disk
    d = load_from_disk(DATA)
    peps = d["peptide"][:n]; charges = [int(c) for c in d["charge"][:n]]
    return list(peps), charges


def old_path_embed(model_dir, peptides, charges):
    """What every loader did before the standard class (portable_eval/mouse/embed_ours.py)."""
    from safetensors.torch import load_file

    from msdelta.reranking import PeptideCollator, PeptideEncoder, student_readout
    state = load_file(str(Path(model_dir) / "model.safetensors"))
    width = int(state["sequence_encoder.projection.3.weight"].shape[0])
    enc = PeptideEncoder(embedding_size=width, hidden_size=256, num_layers=4, num_heads=8,
                         pooling="mean+max", readout=student_readout(state))
    enc.load_state_dict({k.removeprefix("sequence_encoder."): v for k, v in state.items()
                         if k.startswith("sequence_encoder.")})
    enc.eval(); coll = PeptideCollator(); out = []
    with torch.no_grad():
        for s in range(0, len(peptides), 512):
            b = coll(peptides[s:s + 512], [min(max(c, 0), 7) for c in charges[s:s + 512]])
            out.append(torch.nn.functional.normalize(enc(**b).float(), dim=-1))
    return torch.cat(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--old-repo", required=True)
    ap.add_argument("--out", required=True)
    cli = ap.parse_args()
    torch.manual_seed(0)
    from huggingface_hub import snapshot_download

    from msdelta.models.peptide_encoder import PeptideEncoderModel
    report = {}
    peps, charges = peptides_for_test()

    # 1. released Hub model
    snap = Path(snapshot_download(HUB_REPO))
    new = PeptideEncoderModel.from_pretrained(str(snap)).eval().embed(peps, charges)
    old = old_path_embed(snap, peps, charges)
    spec = importlib.util.spec_from_file_location("hub_peptide_embedder", snap / "peptide_embedder.py")
    hubmod = importlib.util.module_from_spec(spec); spec.loader.exec_module(hubmod)
    hub = hubmod.PeptideEmbedder.from_pretrained(str(snap)).eval().embed(peps, charges)
    report["hub_release"] = {"peptides": len(peps), "new_vs_old_path": float((new - old).abs().max()),
                             "new_vs_hub_standalone_module": float((new - hub).abs().max())}

    # 2. an alignment run's raw final/
    new2 = PeptideEncoderModel.from_pretrained(ALIGN_RUN).eval()
    e_new = new2.embed(peps, charges); e_old = old_path_embed(ALIGN_RUN, peps, charges)
    tmp = Path(cli.out).with_suffix("") / "peptide_encoder_roundtrip"
    new2.save_pretrained(str(tmp))
    e_rt = PeptideEncoderModel.from_pretrained(str(tmp)).eval().embed(peps, charges)
    report["alignment_final_dir"] = {"new_vs_old_path": float((e_new - e_old).abs().max()),
                                     "save_load_round_trip": float((e_new - e_rt).abs().max()),
                                     "hub_release_vs_this_run": float((e_new - new).abs().max())}

    # 3. spectrum encoder: old tree vs new tree
    outs = {}
    for tag, repo in (("old", cli.old_repo), ("new", str(Path(__file__).resolve().parents[2]))):
        path = Path(cli.out).with_suffix("") / f"spectrum_{tag}.npy"
        path.parent.mkdir(parents=True, exist_ok=True)
        r = subprocess.run([sys.executable, "-c", SPECTRUM_SNIPPET, DATA, SPECTRUM_MODEL, str(path)],
                           env={**os.environ, "PYTHONPATH": repo, "HF_HUB_OFFLINE": "1",
                                "HF_DATASETS_OFFLINE": "1"},
                           capture_output=True, text=True, cwd=repo)
        if r.returncode:
            report[f"spectrum_{tag}_error"] = r.stderr[-800:]
        else:
            import numpy as np
            outs[tag] = np.load(path)
    if len(outs) == 2:
        report["spectrum_encoder_old_vs_new_tree"] = {"spectra": int(outs["old"].shape[0]),
                                                      "max_abs_diff": float(abs(outs["old"] - outs["new"]).max())}
    ok = all(v == 0.0 for sec in report.values() if isinstance(sec, dict)
             for k, v in sec.items() if k not in ("peptides", "spectra", "hub_release_vs_this_run"))
    report["ALL_IDENTICAL"] = ok and "spectrum_encoder_old_vs_new_tree" in report
    Path(cli.out).write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
