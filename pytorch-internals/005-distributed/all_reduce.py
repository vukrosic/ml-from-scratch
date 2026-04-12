"""
Measure all_reduce bandwidth scaling with tensor size and world size.

all_reduce sums a tensor across all ranks, returning the result to every rank.
Bandwidth scales with tensor size and network speed. This script measures
the effective bandwidth you get on your hardware.

Run with:
    torchrun --nproc_per_node=2 all_reduce.py
"""

import time
import torch
import torch.distributed as dist


def benchmark_all_reduce(tensor_size, steps=200, warmup=20):
    """Time all_reduce for a given tensor size in bytes."""
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tensor = torch.randn(tensor_size, device=device)

    # Warmup
    for _ in range(warmup):
        dist.all_reduce(tensor)
    if device == "cuda":
        torch.cuda.synchronize()

    # Benchmark
    t0 = time.perf_counter()
    for _ in range(steps):
        dist.all_reduce(tensor)
    if device == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    avg_time_ms = (elapsed / steps) * 1000
    bytes_per_tensor = tensor.nelement() * tensor.element_size()
    bandwidth_gb = (bytes_per_tensor * steps) / (elapsed * 1e9)

    return avg_time_ms, bandwidth_gb


def main():
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    if rank == 0:
        print(f"all_reduce bandwidth — {world_size} ranks")
        print("-" * 60)

    # Test tensor sizes: 1 MB to 256 MB
    sizes = [2**i for i in range(10, 19)]  # 1K to 256K elements

    results = []
    for size in sizes:
        tensor_size = size  # elements
        dist.barrier()
        avg_ms, bw_gb = benchmark_all_reduce(tensor_size)
        results.append((tensor_size, avg_ms, bw_gb))

        if rank == 0:
            mb = tensor_size * 4 / 1e6  # float32
            print(f"  {mb:6.1f} MB  avg: ~{avg_ms:.3f} ms  bandwidth: ~{bw_gb:.2f} GB/s")

    dist.barrier()

    if rank == 0:
        print("-" * 60)
        print("Scaling observation: bandwidth typically plateaus at ~8-16 MB")
        print("for NCCL on modern GPUs with NVLink. Smaller tensors are")
        print("latency-bound; larger tensors are bandwidth-bound.")


if __name__ == "__main__":
    # torchrun sets up rank and world_size automatically via env vars
    dist.init_process_group(backend="nccl" if torch.cuda.is_available() else "gloo")
    main()
    dist.destroy_process_group()
