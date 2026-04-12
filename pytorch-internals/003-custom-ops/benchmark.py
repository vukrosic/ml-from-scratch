"""
Benchmark: custom autograd.Function ReLU vs torch.nn.ReLU.

Small tensors test the overhead of Python call dispatch.
Large tensors test the actual compute cost (which dominates).
"""

import torch
import time
from custom_relu import custom_relu


def benchmark(fn, x, n_warmup=50, n_iters=500):
    """
    Time the average forward+backward pass of `fn` on `x`.

    We call fn(x) and then loss.sum().backward() each iteration
    to measure both forward and backward pass.
    """
    x = x.clone()  # fresh tensor each call avoids graph state issues

    # Warmup.
    for _ in range(n_warmup):
        x = x.detach().requires_grad_(True)
        y = fn(x)
        y.sum().backward()

    # Timed iterations.
    start = time.perf_counter()
    for _ in range(n_iters):
        x = x.detach().requires_grad_(True)
        y = fn(x)
        y.sum().backward()
    elapsed = time.perf_counter() - start

    return elapsed / n_iters * 1000  # ms per iteration


if __name__ == "__main__":
    # CPU benchmarks — no GPU needed for this lesson.
    sizes = [
        (128, 256),
        (1024, 512),
        (4096, 768),
    ]

    print(f"{'Shape':>20}  {'nn.ReLU':>10}  {'Custom':>10}  {'Ratio':>8}")
    print("-" * 55)

    for shape in sizes:
        x = torch.randn(*shape)

        t_relu = benchmark(torch.relu, x)
        t_custom = benchmark(custom_relu, x)
        ratio = t_custom / t_relu

        print(f"{str(shape):>20}  {t_relu:>10.4f}  {t_custom:>10.4f}  {ratio:>8.3f}x")

    print("\nExpected: custom op is slightly slower (Python overhead).")
    print("Real speedups come from fused kernels, not from custom Python ops.")
