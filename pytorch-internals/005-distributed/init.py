"""
Benchmark torch.distributed init methods: nccl, gloo, and cpu backends.

Distributed training starts with initialization. The choice of backend
affects communication speed between processes. NCCL is fastest for GPUs,
Gloo is the fallback for CPU or multi-machine setups.

Run with:
    torchrun --nproc_per_node=2 init.py
"""

import time
import torch
import torch.distributed as dist


def benchmark_init(backend, init_method, world_size, rank):
    """Time the cost of initializing the distributed process group."""
    dist.init_process_group(
        backend=backend,
        init_method=init_method,
        world_size=world_size,
        rank=rank,
        timeout=time.delta(seconds=30),
    )

    # Warmup: a trivial all_reduce to ensure the group is functional
    tensor = torch.zeros(1, device="cuda" if backend == "nccl" else "cpu")
    for _ in range(10):
        dist.all_reduce(tensor)

    # Benchmark: timed all_reduce calls
    torch.cuda.synchronize() if backend == "nccl" else None
    t0 = time.perf_counter()
    for _ in range(100):
        dist.all_reduce(tensor)
    torch.cuda.synchronize() if backend == "nccl" else None
    elapsed = time.perf_counter() - t0

    avg_latency_ms = (elapsed / 100) * 1000

    dist.destroy_process_group()

    return avg_latency_ms


def main():
    # Only run on rank 0 — the benchmark output is collected there
    if dist.get_rank() != 0:
        dist.init_process_group("nccl")
        dist.destroy_process_group()
        return

    world_size = dist.get_world_size()
    rank = dist.get_rank()

    backends = [
        ("nccl", "env://"),
        ("gloo", "env://"),
        ("gloo", "tcp://localhost:29500"),
    ]

    print(f"torch.distributed init methods — {world_size} ranks")
    print("-" * 50)
    for backend, init_method in backends:
        # Re-init for each backend to get fresh measurements
        dist.init_process_group(
            backend=backend,
            init_method=init_method,
            world_size=world_size,
            rank=rank,
        )
        latency = benchmark_init(backend, init_method, world_size, rank)
        print(f"{backend:6s}  {init_method:20s}  avg all_reduce: ~{latency:.3f} ms")
        dist.destroy_process_group()


if __name__ == "__main__":
    # Use torchrun to set up the environment variables automatically
    # torchrun --nproc_per_node=2 init.py
    main()
