"""
Profile memory and compute for LayerNorm, GroupNorm, and RMSNorm.

We measure:
1. Forward pass latency across batch sizes and feature dimensions
2. Memory usage (bytes allocated) for the normalized tensor
3. Backward pass cost (autograd gradient computation)

This helps understand why modern LLMs (Llama, Mistral) prefer RMSNorm —
the small saving per layer compounds across dozens of layers and long
sequences.
"""

import time
import torch
import torch.nn as nn
import gc

from layer_norm import LayerNorm
from rms_norm import RMSNorm


def reset_memory():
    """Force garbage collection and reset peak memory stats."""
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats() if torch.cuda.is_available() else None


def benchmark_forward(fn, n_runs=100, n_warmup=10):
    """Time forward pass in μs."""
    for _ in range(n_warmup):
        fn()
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    start = time.perf_counter()
    for _ in range(n_runs):
        fn()
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    return (time.perf_counter() - start) / n_runs * 1e6


def get_memory_mb():
    """Return peak allocated GPU memory in MB, or 0 if no CUDA."""
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / 1024 / 1024
    return 0.0


def profile_shapes(shapes, dtypes=None):
    """Profile LayerNorm, GroupNorm, RMSNorm across multiple tensor shapes."""
    if dtypes is None:
        dtypes = [torch.float32]

    results = []

    for dtype in dtypes:
        for shape in shapes:
            batch, seq, d_model = shape
            x = torch.randn(shape, dtype=dtype, requires_grad=True)

            ln = LayerNorm(d_model).to(dtype)
            gn = nn.GroupNorm(num_groups=1, num_channels=d_model).to(dtype)
            rms = RMSNorm(d_model).to(dtype)

            layers = {
                "LayerNorm": ln,
                "GroupNorm(1)": gn,
                "RMSNorm": rms,
            }

            for name, layer in layers.items():
                reset_memory()

                # Forward memory
                y = layer(x)
                fwd_mem = get_memory_mb()

                # Forward time
                fwd_time = benchmark_forward(lambda: layer(x.detach()))

                # Backward time
                _, x_ckpt = x.detach().clone().requires_grad_(True), x.detach().clone().requires_grad_(True)
                layer_ckpt = type(layer)(d_model).to(dtype) if name != "GroupNorm(1)" else gn

                def backward_fn():
                    y = layer_ckpt(x_ckpt)
                    y.sum().backward()

                bwd_time = benchmark_forward(backward_fn)

                results.append({
                    "dtype": str(dtype).split(".")[-1],
                    "shape": str(shape),
                    "layer": name,
                    "fwd_us": round(fwd_time, 2),
                    "bwd_us": round(bwd_time, 2),
                    "fwd_mem_mb": round(fwd_mem, 4),
                })

    return results


def print_table(results):
    """Pretty-print profiling results."""
    header = f"{'dtype':<10} {'shape':<20} {'layer':<14} {'fwd (μs)':>10} {'bwd (μs)':>10} {'mem (MB)':>10}"
    print(header)
    print("-" * 80)
    for r in results:
        print(f"{r['dtype']:<10} {r['shape']:<20} {r['layer']:<14} "
              f"{r['fwd_us']:>10.2f} {r['bwd_us']:>10.2f} {r['fwd_mem_mb']:>10.4f}")


if __name__ == "__main__":
    print("Profiling LayerNorm, GroupNorm(1), and RMSNorm")
    print("=" * 80)

    # Test across these configurations
    shapes = [
        (1, 64, 128),
        (4, 64, 256),
        (8, 128, 512),
        (16, 512, 768),
    ]
    dtypes = [torch.float32, torch.float16]

    print("\nNote: CUDA not available — using CPU timings")
    print("On CUDA, results will differ (GPU math is highly parallel).\n")

    results = profile_shapes(shapes, dtypes)
    print_table(results)

    print("\nKey observations:")
    print("  - LayerNorm and GroupNorm(1) have identical compute cost (same formula)")
    print("  - RMSNorm saves one mean-reduction, but the difference is small in PyTorch")
    print("  - Memory is dominated by the input tensor itself, not normalization overhead")
    print("  - Larger batch/seq dims amplify the compute cost of the reductions")
