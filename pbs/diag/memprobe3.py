"""Does gradient checkpointing actually reduce activation memory here?"""
import sys, torch
sys.path.insert(0, "/home/khuss/code/msdelta")
CKPT = "/flare/UIC-HPC/khuss/msdelta/pretrained/msdelta-200m-production-01-checkpoint-192799"
torch.xpu.set_device(0)
from msdelta.modeling_msdelta import MSDeltaForPreTraining

def trial(ckpt_on, peaks, batch=4):
    m = MSDeltaForPreTraining.from_pretrained(CKPT).to("xpu")
    enc = m.msdelta
    if ckpt_on:
        m.gradient_checkpointing_enable()
        enc.train()
    torch.xpu.empty_cache(); torch.xpu.reset_peak_memory_stats()
    base = torch.xpu.memory_allocated()/1e9
    mz = torch.rand(batch, peaks, device="xpu")*1000+100
    li = torch.rand(batch, peaks, device="xpu")
    am = torch.ones(batch, peaks, dtype=torch.bool, device="xpu")
    h = enc(mz=mz, log_intensity=li, attention_mask=am).last_hidden_state
    live = torch.xpu.memory_allocated()/1e9 - base
    peak = torch.xpu.max_memory_allocated()/1e9 - base
    flag = getattr(enc, "gradient_checkpointing", None)
    print(f"  ckpt={str(ckpt_on):<5} peaks={peaks:<4} batch={batch}  "
          f"activations live {live:6.2f} GB  peak {peak:6.2f} GB   "
          f"(enc.gradient_checkpointing={flag}, training={enc.training})", flush=True)
    del m, enc, h, mz, li, am
    torch.xpu.empty_cache()

for ckpt_on in (False, True):
    for peaks in (512, 256):
        trial(ckpt_on, peaks)
print(flush=True)
trial(True, 512, batch=16)
