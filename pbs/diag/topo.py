import torch
n = torch.xpu.device_count()
p = torch.xpu.get_device_properties(0)
print(f"  visible tiles: {n}")
print(f"  per-tile reported memory: {p.total_memory/1e9:.1f} GB")
print(f"  sum over tiles: {n*p.total_memory/1e9:.0f} GB")
print(f"  props: {p}")
