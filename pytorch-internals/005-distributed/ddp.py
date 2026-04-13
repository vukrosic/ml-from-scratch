"""
Compare single-GPU vs DistributedDataParallel training speedup.

DDP wraps a model and shards gradient computation across ranks.
Each rank computes its portion of the batch, then all_reduce syncs
the gradients before the optimizer step. This script measures
the effective speedup from using DDP across multiple GPUs.

Run with:
    torchrun --nproc_per_node=2 ddp.py
"""

import time
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP


class SimpleModel(nn.Module):
    """Two-layer MLP for benchmarking."""

    def __init__(self, input_dim=512, hidden_dim=1024, output_dim=10):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        return x


def step(model, optimizer, inputs, targets, gradient_scaling=True):
    """One training step: forward, backward, optimizer update."""
    optimizer.zero_grad()
    output = model(inputs)
    loss = nn.functional.cross_entropy(output, targets)
    loss.backward()

    if gradient_scaling:
        # DDP scales loss by world_size for correct gradient averaging
        # Manual implementation needs the same treatment
        for p in model.parameters():
            if p.grad is not None:
                p.grad /= dist.get_world_size()

    optimizer.step()


def benchmark_single_gpu(model, batch_size, steps=100, warmup=20):
    """Time a single-GPU training loop."""
    device = torch.device("cuda:0")
    model = model.to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    inputs = torch.randn(batch_size, 512, device=device)
    targets = torch.randint(0, 10, (batch_size,), device=device)

    # Warmup
    for _ in range(warmup):
        step(model, optimizer, inputs, targets, gradient_scaling=False)

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(steps):
        step(model, optimizer, inputs, targets, gradient_scaling=False)
    torch.cuda.synchronize()

    return (time.perf_counter() - t0) / steps


def benchmark_ddp(rank, world_size, model, batch_size, steps=100, warmup=20):
    """Time a DDP training loop on one rank."""
    device = torch.device(f"cuda:{rank}")
    model = model.to(device)
    ddp_model = DDP(model, device_ids=[rank])
    optimizer = torch.optim.SGD(ddp_model.parameters(), lr=0.01)

    inputs = torch.randn(batch_size, 512, device=device)
    targets = torch.randint(0, 10, (batch_size,), device=device)

    # Warmup
    for _ in range(warmup):
        step(ddp_model, optimizer, inputs, targets, gradient_scaling=True)

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(steps):
        step(ddp_model, optimizer, inputs, targets, gradient_scaling=True)
    torch.cuda.synchronize()

    return (time.perf_counter() - t0) / steps


def main():
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device_count = torch.cuda.device_count()

    if world_size == 1:
        # Single-process fallback for testing without torchrun
        model = SimpleModel()
        single_time = benchmark_single_gpu(model, batch_size=256)
        print(f"Single GPU: ~{single_time*1000:.2f} ms/step")
        return

    if rank == 0:
        print(f"DDP speedup — {world_size} GPUs, batch per GPU = 256")
        print("-" * 50)

    # Per-rank batch size
    batch_per_gpu = 256

    model = SimpleModel()

    # Only measure on rank 0 to avoid redundant output
    if rank == 0:
        ddp_time = benchmark_ddp(rank, world_size, model, batch_per_gpu)

    # Also run single-GPU for comparison (on rank 0's device for fairness)
    if rank == 0 and device_count > 0:
        model_single = SimpleModel()
        single_time = benchmark_single_gpu(model_single, batch_size=batch_per_gpu * world_size)

        speedup = single_time / ddp_time

        print(f"Single GPU (total batch={batch_per_gpu*world_size}): ~{single_time*1000:.2f} ms/step")
        print(f"DDP ({world_size} GPUs, batch={batch_per_gpu} each):     ~{ddp_time*1000:.2f} ms/step")
        print(f"Speedup: ~{speedup:.2f}x")
        print()
        print("Note: DDP speedup assumes compute-bound workloads.")
        print("For small models, communication overhead may dominate.")

    dist.barrier()


if __name__ == "__main__":
    # torchrun auto-sets rank and world_size via environment variables
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    main()
    dist.destroy_process_group()
