"""K138-I: DataLoader workers returning tensors (fd sharing -> multiprocessing resource_sharer socket in TMPDIR)."""
import os, tempfile, torch
from torch.utils.data import DataLoader, Dataset
class D(Dataset):
    def __len__(self): return 64
    def __getitem__(self, i): return torch.full((512, 2), float(i))
n = sum(b.shape[0] for b in DataLoader(D(), batch_size=8, num_workers=6))
print(f"rank {os.environ.get('PMIX_RANK')} TMPDIR={tempfile.gettempdir()} ({len(tempfile.gettempdir())} chars) -> {n} rows OK", flush=True)
