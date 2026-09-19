"""Which part of the peptide student faults on twelve tiles? Bisect by deletion.

One variant per process, chosen by --variant, so a fault kills only that variant and the
job script can run the next. Each builds a progressively larger slice of PeptideEncoder
and takes a few optimizer steps under whatever parallelism the launcher set up.

The point is to name a module, not to train anything. Denoise runs fine on twelve tiles
and shares every module type here EXCEPT nn.TransformerEncoder, so the ordering below is
chosen to reach that as late as possible: if 'embeddings' already faults, the suspect is
wrong and the cause is something all these variants share.
"""

from __future__ import annotations

import argparse
import os
import sys

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from msdelta.finetune_denoise import select_device  # noqa: E402
from msdelta.fourier import FourierFeatures  # noqa: E402

VARIANTS = ("linear", "embeddings", "fourier", "transformer1", "transformer4", "student")


def build(variant: str, hidden: int = 256) -> nn.Module:
    if variant == "linear":
        # The floor: dense matmuls and nothing else. If this faults on twelve tiles the
        # problem is not in our model at all.
        class M(nn.Module):
            def __init__(self):
                super().__init__()
                self.stack = nn.Sequential(nn.Linear(hidden, hidden), nn.GELU(),
                                           nn.Linear(hidden, 64))

            def forward(self, ids):
                x = ids.float().unsqueeze(-1).expand(-1, -1, hidden)
                return self.stack(x).mean(1)
        return M()

    if variant == "embeddings":
        class M(nn.Module):
            def __init__(self):
                super().__init__()
                self.residue = nn.Embedding(23, hidden, padding_idx=0)
                self.position = nn.Embedding(64, hidden)
                self.charge = nn.Embedding(8, hidden)
                self.out = nn.Linear(hidden, 64)

            def forward(self, ids):
                pos = torch.arange(ids.shape[1], device=ids.device)
                h = self.residue(ids) + self.position(pos)[None]
                h = h + self.charge(torch.zeros_like(ids[:, 0]))[:, None]
                return self.out(h.mean(1))
        return M()

    if variant == "fourier":
        class M(nn.Module):
            def __init__(self):
                super().__init__()
                self.features = FourierFeatures(16, 1e-2, 1e3)
                self.project = nn.Linear(self.features.out_dim, hidden)
                self.out = nn.Linear(hidden, 64)

            def forward(self, ids):
                h = self.project(self.features(ids.float()).to(torch.float32))
                return self.out(h.mean(1))
        return M()

    if variant.startswith("transformer"):
        layers = int(variant.removeprefix("transformer"))

        class M(nn.Module):
            def __init__(self):
                super().__init__()
                self.embed = nn.Embedding(23, hidden, padding_idx=0)
                layer = nn.TransformerEncoderLayer(
                    d_model=hidden, nhead=8, dim_feedforward=4 * hidden, dropout=0.1,
                    batch_first=True, norm_first=True, activation="gelu")
                self.encoder = nn.TransformerEncoder(layer, num_layers=layers,
                                                     enable_nested_tensor=False)
                self.out = nn.Linear(hidden, 64)

            def forward(self, ids):
                mask = ids.eq(0)
                h = self.encoder(self.embed(ids), src_key_padding_mask=mask)
                return self.out(h.mean(1))
        return M()

    from msdelta.reranking import PeptideEncoder

    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.inner = PeptideEncoder(embedding_size=64, hidden_size=hidden,
                                        num_layers=4, num_heads=8)

        def forward(self, ids):
            return self.inner(ids, torch.zeros_like(ids, dtype=torch.float32),
                              ids.ne(0).long(), torch.full((ids.shape[0],), 2,
                                                           device=ids.device))
    return M()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=VARIANTS, required=True)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--batch", type=int, default=4)
    cli = parser.parse_args()

    select_device()
    rank = int(os.environ.get("RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    device = "xpu" if torch.xpu.is_available() else "cpu"
    if world > 1:
        torch.distributed.init_process_group(backend="xccl", rank=rank, world_size=world)

    model = build(cli.variant).to(device)
    if world > 1:
        import deepspeed
        # An explicit optimizer, not just model_parameters: ZeRO-2 partitions optimizer
        # STATE, so with only parameters it is handed a DummyOptim and asserts
        # "zero stage 2 requires an optimizer". The real training path gets one from the
        # Trainer, which is why this only showed up here.
        model, optimizer, _, _ = deepspeed.initialize(
            model=model, model_parameters=model.parameters(),
            optimizer=torch.optim.AdamW(model.parameters(), lr=1e-3),
            config={"train_batch_size": cli.batch * world,
                    "train_micro_batch_size_per_gpu": cli.batch,
                    "gradient_accumulation_steps": 1,
                    "torch_autocast": {"enabled": True, "dtype": "bfloat16"},
                    "zero_optimization": {"stage": 2, "contiguous_gradients": True,
                                          "overlap_comm": True}})
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    if rank == 0:
        n = sum(p.numel() for p in model.parameters())
        print(f"[bisect] {cli.variant}: {n/1e6:.3f}M params, world={world}", flush=True)

    torch.manual_seed(rank)
    ids = torch.randint(1, 23, (cli.batch, 32), device=device)
    for step in range(cli.steps):
        loss = build_loss(model, ids)
        if world > 1:
            model.backward(loss)
            model.step()
        else:
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        if device == "xpu":
            torch.xpu.synchronize()      # localise the fault to THIS step
        if rank == 0 and step % 5 == 0:
            print(f"[bisect] {cli.variant} step {step} loss {float(loss):.4f}", flush=True)

    if rank == 0:
        print(f"[bisect] {cli.variant}: SURVIVED {cli.steps} steps", flush=True)
    return 0


def build_loss(model, ids):
    return model(ids).float().pow(2).mean()


if __name__ == "__main__":
    raise SystemExit(main())
