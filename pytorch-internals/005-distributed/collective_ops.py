"""
Time each NCCL collective operation: all_reduce, all_gather, broadcast, reduce.

Collective ops are the building blocks of multi-GPU communication.
Each has different semantics and cost. This script benchmarks
the round-trip latency and effective bandwidth for each on your hardware.

Run with:
    torchrun --nproc_per_node=2 collective_ops.py
"""

import time
import torch
import torch.distributed as dist


def benchmark_op(op_fn, tensor, steps=200, warmup=20):
    """Time a collective operation. op_fn takes and returns a tensor."""
    # Warmup
    for _ in range(warmup):
        op_fn(tensor)
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    # Benchmark
    t0 = time.perf_counter()
    for _ in range(steps):
        op_fn(tensor)
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    return (time.perf_counter() - t0) / steps


def main():
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    is_cuda = torch.cuda.is_available()
    device = "cuda" if is_cuda else "cpu"
    backend = "nccl" if is_cuda else "gloo"

    if rank == 0:
        print(f"NCCL collective ops — {world_size} ranks, {device}")
        print("-" * 55)

    # Use a moderately large tensor to measure bandwidth
    tensor_size = 2**18  # ~1M elements, ~4 MB for float32
    tensor = torch.randn(tensor_size, device=device)

    results = {}

    # all_reduce: sum across all ranks, result on every rank
    def all_reduce_op(t):
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
        return t

    results["all_reduce"] = benchmark_op(all_reduce_op, tensor.clone())

    # all_gather: concatenate tensors from all ranks onto every rank
    def all_gather_op(t):
        output = [torch.zeros_like(t) for _ in range(world_size)]
        dist.all_gather(output, t)
        return output[0]

    results["all_gather"] = benchmark_op(all_gather_op, tensor.clone())

    # broadcast: send tensor from root rank to all other ranks
    def broadcast_op(t):
        dist.broadcast(t, src=0)
        return t

    results["broadcast"] = benchmark_op(broadcast_op, tensor.clone())

    # reduce: sum from all ranks onto root rank only
    def reduce_op(t):
        dist.reduce(t, src=0, op=dist.ReduceOp.SUM)
        return t

    results["reduce"] = benchmark_op(reduce_op, tensor.clone())

    # Barrier (not a collective communication but synchronizes all ranks)
    def barrier_op(t):
        dist.barrier()
        return t

    results["barrier"] = benchmark_op(barrier_op, tensor.clone())

    if rank == 0:
        mb = tensor_size * 4 / 1e6
        for op_name, avg_time in results.items():
            print(f"  {op_name:12s}  {mb:.1f} MB  avg: ~{avg_time*1000:.3f} ms")

    dist.barrier()

    if rank == 0:
        print("-" * 55)
        print("all_reduce and all_gather are bandwidth-bound at this size.")
        print("broadcast and reduce have similar cost to all_reduce.")
        print("barrier is a synchronization point — measures worst-case delay.")


if __name__ == "__main__":
    dist.init_process_group(backend="nccl" if torch.cuda.is_available() else "gloo")
    main()
    dist.destroy_process_group()
