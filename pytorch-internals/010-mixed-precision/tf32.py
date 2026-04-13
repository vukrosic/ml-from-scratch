"""
Measure TF32 speedup vs FP32 on Ampere+ GPUs.

Benchmarks matrix multiplication throughput with and without TF32 enabled.
Only meaningful on Ampere+ hardware (A100, RTX 30xx, RTX 40xx).
Run: python tf32.py [--m N] [--k N] [--n N] [--iterations N]
"""

import argparse
import time
import torch


def benchmark_matmul(m, k, n, iterations, warmup, use_tf32):
    """Benchmark a single matmul operation."""
    if use_tf32:
        torch.set_float32_matmul_precision("high")
    else:
        torch.set_float32_matmul_precision("highest")

    device = torch.device("cuda")

    # Allocate matrices
    a = torch.randn(m, k, device=device)
    b = torch.randn(k, n, device=device)

    # Warmup
    for _ in range(warmup):
        c = a @ b

    torch.cuda.synchronize()

    # Timing
    t0 = time.perf_counter()
    for _ in range(iterations):
        c = a @ b
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    # Reset to default
    torch.set_float32_matmul_precision("highest")

    return elapsed / iterations


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--m", type=int, default=4096, help="Rows of first matrix")
    parser.add_argument("--k", type=int, default=4096, help="Columns of first / rows of second")
    parser.add_argument("--n", type=int, default=4096, help="Columns of second matrix")
    parser.add_argument("--iterations", type=int, default=500)
    parser.add_argument("--warmup", type=int, default=100)
    args = parser.parse_args()

    print(f"Matrix sizes: ({args.m} x {args.k}) @ ({args.k} x {args.n})")
    print(f"Iterations: {args.iterations}, Warmup: {args.warmup}")
    print()

    device_name = torch.cuda.get_device_name(0)
    compute_capability = torch.cuda.get_device_capability(0)
    print(f"GPU: {device_name}")
    print(f"Compute capability: {compute_capability[0]}.{compute_capability[1]}")
    print()

    if compute_capability[0] < 8:
        print("WARNING: TF32 is only supported on Ampere+ (compute capability 8.0+)")
        print("This benchmark will not show meaningful speedup on your hardware.")
        print()

    # Benchmark FP32
    print("Benchmarking FP32 matmul...")
    fp32_time = benchmark_matmul(args.m, args.k, args.n, args.iterations, args.warmup, use_tf32=False)
    print(f"  FP32: {fp32_time*1000:.3f} ms")

    # Benchmark TF32
    print("Benchmarking TF32 matmul...")
    tf32_time = benchmark_matmul(args.m, args.k, args.n, args.iterations, args.warmup, use_tf32=True)
    print(f"  TF32: {tf32_time*1000:.3f} ms")

    print()
    print(f"TF32 speedup: {fp32_time/tf32_time:.2f}x faster than FP32")

    # Compute throughput
    flops = 2 * args.m * args.k * args.n  # multiply-add counts as 1 FLOP but we count 2
    tf32_throughput = flops / tf32_time / 1e12
    fp32_throughput = flops / fp32_time / 1e12
    print(f"FP32 throughput: {fp32_throughput:.2f} TFLOPS")
    print(f"TF32 throughput: {tf32_throughput:.2f} TFLOPS")


if __name__ == "__main__":
    main()
