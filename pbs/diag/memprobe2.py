"""Which part of the contrastive forward allocates 15 GB?"""
import sys, torch
sys.path.insert(0, "/home/khuss/code/msdelta")
def mem(tag):
    print(f"  {tag:<40} live {torch.xpu.memory_allocated()/1e9:6.2f}  "
          f"peak {torch.xpu.max_memory_allocated()/1e9:6.2f} GB", flush=True)

CKPT = "/flare/UIC-HPC/khuss/msdelta/pretrained/msdelta-200m-production-01-checkpoint-192799"
torch.xpu.set_device(0)
from msdelta.modeling_msdelta import MSDeltaForPreTraining
m = MSDeltaForPreTraining.from_pretrained(CKPT).to("xpu")
cfg = m.config
print(f"  config: hidden {cfg.hidden_size} layers {cfg.num_hidden_layers} "
      f"heads {cfg.num_attention_heads}", flush=True)
for attr in ("n_freqs", "num_freqs", "bias_num_freqs", "fourier_features"):
    if hasattr(cfg, attr): print(f"  cfg.{attr} = {getattr(cfg, attr)}", flush=True)

B, L = 4, 512
mz = torch.rand(B, L, device="xpu")*1000+100
li = torch.rand(B, L, device="xpu")
am = torch.ones(B, L, dtype=torch.bool, device="xpu")
torch.xpu.reset_peak_memory_stats(); mem("before anything")

enc = m.msdelta
with torch.no_grad():
    bias = enc.bias_module(mz)
print(f"  DeltaMZBias tensor: {tuple(bias.shape)}  {bias.element_size()*bias.nelement()/1e9:.2f} GB", flush=True)
del bias; torch.xpu.empty_cache(); torch.xpu.reset_peak_memory_stats()

with torch.no_grad():
    h = enc(mz=mz, log_intensity=li, attention_mask=am).last_hidden_state
mem("encoder forward, NO GRAD")
print(f"  hidden {tuple(h.shape)}", flush=True)
logits = m.intensity_head(h)
print(f"  intensity_head output {tuple(logits.shape)}  "
      f"{logits.element_size()*logits.nelement()/1e9:.3f} GB", flush=True)
del h, logits; torch.xpu.empty_cache(); torch.xpu.reset_peak_memory_stats()

h = enc(mz=mz, log_intensity=li, attention_mask=am).last_hidden_state
mem("encoder forward, WITH GRAD")
logits = m.intensity_head(h)
mem("+ intensity_head")
