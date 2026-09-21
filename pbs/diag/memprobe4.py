"""50m at the batch sizes the pair grid used -- does memory explain those faults?"""
import sys, torch
sys.path.insert(0, "/home/khuss/code/msdelta")
torch.xpu.set_device(0)
from msdelta.modeling_msdelta import MSDeltaForPreTraining
C50 = "/flare/UIC-HPC/khuss/msdelta/pretrained/msdelta-50m-production-01-checkpoint-133233"
C200 = "/flare/UIC-HPC/khuss/msdelta/pretrained/msdelta-200m-production-01-checkpoint-192799"

def trial(ckpt, tag, batch, peaks=512):
    m = MSDeltaForPreTraining.from_pretrained(ckpt).to("xpu")
    m.gradient_checkpointing_enable(); enc = m.msdelta; enc.train()
    torch.xpu.empty_cache(); torch.xpu.reset_peak_memory_stats()
    base = torch.xpu.memory_allocated()/1e9
    try:
        mz = torch.rand(batch, peaks, device="xpu")*1000+100
        li = torch.rand(batch, peaks, device="xpu")
        am = torch.ones(batch, peaks, dtype=torch.bool, device="xpu")
        h = enc(mz=mz, log_intensity=li, attention_mask=am).last_hidden_state
        h.sum().backward()
        print(f"  {tag:<6} batch {batch:<3} peaks {peaks}  OK   "
              f"activations {torch.xpu.max_memory_allocated()/1e9-base:6.2f} GB", flush=True)
        del h, mz, li, am
    except Exception as e:
        print(f"  {tag:<6} batch {batch:<3} peaks {peaks}  {type(e).__name__}: {str(e)[:60]}", flush=True)
    del m, enc; torch.xpu.empty_cache()

for b in (2, 4, 8, 16):
    trial(C50, "50m", b)
print(flush=True)
for b in (2, 4):
    trial(C200, "200m", b)
trial(C200, "200m", 4, peaks=256)
trial(C200, "200m", 2, peaks=512)
