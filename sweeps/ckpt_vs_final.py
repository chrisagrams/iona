"""Is contrastive final/ the best checkpoint, or just the last one?

Contrastive trains with --eval_strategy no, so nothing ever compared an intermediate
checkpoint against the end. save_total_limit 2 rotated all but the last two away, so
this can only see the tail -- but if even the tail is non-monotonic, final/ is not a
safe choice and the early checkpoints we are about to delete may hold better models.
"""
import sys, os, glob, json, torch
sys.path.insert(0, "/home/khuss/code/msdelta")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
os.environ.setdefault("HF_HOME", "/lus/flare/projects/UIC-HPC/khuss/msdelta/huggingface")
from msdelta.eval_retrieval import score_encoder
from msdelta.finetune_contrastive import ContrastiveCollator
from msdelta.processing_msdelta import MSDeltaProcessor
from msdelta.reranking import build_alignment_datasets
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import homes  # noqa: E402  (data homes, configs/homes.env)

R = "/lus/flare/projects/UIC-HPC/khuss/msdelta/runs"
arms = sorted(glob.glob(f"{R}/sweep-50m_ck330k_seed*-8851663"))[:3]
device = torch.device("xpu" if torch.xpu.is_available() else "cpu")
proc = MSDeltaProcessor.from_pretrained(arms[0] + "/final", max_peaks=512)
ds = build_alignment_datasets("chrisagrams/ms2-peptide-replicate-retrieval", proc,
                              validation_fraction=0.1, seed=0)
coll = ContrastiveCollator(max_peptide_length=64, pad_spectra_to=512)
out = {}
for a in arms:
    name = os.path.basename(a)
    # checkpoint-N/model.safetensors is the WRAPPER's bare state dict -- model.* keys,
    # no config.json -- so loading the checkpoint root silently yields a random encoder
    # scoring MAP@R 0.0029 and ratio nan. SaveEncoderCallback writes the loadable HF
    # encoder into checkpoint-N/encoder; that is the artifact to score.
    subs = [c + "/encoder" for c in sorted(glob.glob(a + "/checkpoint-*"))
            if os.path.isdir(c + "/encoder")] + [a + "/final"]
    for sub in subs:
        tag = ("/".join(sub.rstrip("/").split("/")[-2:])
               if sub.endswith("encoder") else os.path.basename(sub))
        try:
            s = score_encoder(sub, ds, coll, device, 2000, 16)
            out[f"{name}/{tag}"] = s
            print(f"  {name:32s} {tag:16s} MAP@R {s['retrieval/MAP@R']:.4f}  "
                  f"P@1 {s['retrieval/Precision@1']:.4f}  ratio {s['sep_spectrum/ratio']:.2f}",
                  flush=True)
        except Exception as e:
            print(f"  {name} {tag}: FAILED {type(e).__name__}: {e}", flush=True)
json.dump(out, open(homes.EVAL / "contrastive" / "ckpt_vs_final.json", "w"), indent=2)
print("  wrote ckpt_vs_final.json")
