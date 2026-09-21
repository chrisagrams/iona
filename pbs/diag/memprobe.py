"""Attribute 200m contrastive memory stage by stage, on one tile."""
import os, sys, torch
sys.path.insert(0, "/home/khuss/code/msdelta")

def mem(tag):
    a = torch.xpu.memory_allocated()/1e9
    r = torch.xpu.memory_reserved()/1e9
    p = torch.xpu.max_memory_allocated()/1e9
    print(f"  {tag:<34} live {a:6.2f}  peak {p:6.2f}  reserved {r:6.2f} GB", flush=True)

CKPT = sys.argv[1] if len(sys.argv) > 1 else \
    "/flare/UIC-HPC/khuss/msdelta/pretrained/msdelta-200m-production-01-checkpoint-192799"
torch.xpu.set_device(0)
mem("start")

from msdelta.modeling_msdelta import MSDeltaForPreTraining
from msdelta.contrastive import MSDeltaForContrastive
enc = MSDeltaForPreTraining.from_pretrained(CKPT).to("xpu")
mem("encoder loaded")
n = sum(p.numel() for p in enc.parameters())
print(f"  ({n/1e6:.0f}M params, dtype {next(enc.parameters()).dtype})", flush=True)

ref = MSDeltaForPreTraining.from_pretrained(CKPT).to("xpu")
mem("+ frozen reference")

model = MSDeltaForContrastive(enc, ref, pooling="mean+max", temperature=0.07,
                              kl_weight=10.0).to("xpu")
mem("+ wrapper")

opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=2e-5)
mem("+ optimizer (states lazy)")

B, L = 4, 512
mz = (torch.rand(B, L, device="xpu")*1000+100)
li = torch.rand(B, L, device="xpu")
am = torch.ones(B, L, dtype=torch.bool, device="xpu")
grp = torch.tensor([0,0,1,1], device="xpu")
mem("+ inputs")

out = model(mz=mz, log_intensity=li, attention_mask=am, group=grp)
mem("+ FORWARD")
out["loss"].backward()
mem("+ BACKWARD")
opt.step()
mem("+ optimizer.step (states live)")
print(f"\n  loss {float(out['loss']):.4f}", flush=True)
