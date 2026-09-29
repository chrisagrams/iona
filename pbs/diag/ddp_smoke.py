"""K138-I: minimal multi-rank DDP on XPU with xccl, launched like aurora-pretrain.pbs (mpiexec --pmi=pmix).

SMOKE_MPI4PY=1 imports mpi4py (MPI_Init) before init_process_group, as ALCF's DDP example does.
Prints one line per stage per rank so a hang/crash shows where it stopped."""
import os
import sys
import time

if os.environ.get("SMOKE_MPI4PY") == "1":
    from mpi4py import MPI  # noqa: F401  (initializes MPI)

import torch
import torch.distributed as dist

rank = int(os.environ["RANK"])
world = int(os.environ["WORLD_SIZE"])
local = int(os.environ["LOCAL_RANK"])


def say(msg):
    print(f"[smoke rank {rank}/{world}] {msg}", flush=True)


say(f"start mpi4py={os.environ.get('SMOKE_MPI4PY', '0')} kvs={os.environ.get('CCL_KVS_MODE')}")
torch.xpu.set_device(local)
dist.init_process_group("xccl", rank=rank, world_size=world, device_id=torch.device("xpu", local))
say("init_process_group ok")
t = torch.ones(4, device="xpu") * (rank + 1)
dist.all_reduce(t)
say(f"all_reduce ok -> {t[0].item()} (expect {world * (world + 1) / 2})")
model = torch.nn.parallel.DistributedDataParallel(torch.nn.Linear(64, 64).to("xpu"), device_ids=[local])
say("DDP wrap ok")
opt = torch.optim.SGD(model.parameters(), lr=0.1)
for _ in range(3):
    opt.zero_grad()
    model(torch.randn(8, 64, device="xpu")).sum().backward()
    opt.step()
torch.xpu.synchronize()
say("3 DDP steps ok")
dist.barrier()
dist.destroy_process_group()
say("done")
sys.exit(0)
