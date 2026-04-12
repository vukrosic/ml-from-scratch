"""Benchmark all torch.compile modes across five batch sizes.

What it does:
  - Tests a 4-layer MLP with hidden=4096 across batch sizes [1, 4, 16, 64, 256].
  - Compares four modes: default, reduce-overhead, max-autotune, max-autotune-no-cudagraphs.
  - Each cell is speedup over eager. Higher is better.

Run:
  python benchmark_modes.py

Expected output:
  Batch      default    reduce-overhead    max-autotune    max-autotune-no-cudagraphs
  ------------------------------------------------------------------------------
      1        1.23x           1.45x            1.51x                  1.48x
      4        1.87x           2.01x            2.34x                  2.19x
     ...

Note:
  reduce-overhead uses CUDA graph capture and is not guaranteed to work for all models.
  If it fails on your model, you will see a warning or error at runtime.
"""

import time
import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, d=1024, hidden=4096):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, d),
        )

    def forward(self, x):
        return self.net(x)


def benchmark(fn, x, steps=200, warmup=30):
    """Measure milliseconds per step after warmup."""
    for _ in range(warmup):
        fn(x)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(steps):
        fn(x)
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / steps * 1000  # ms


if __name__ == "__main__":
    device = "cuda"
    torch.set_float32_matmul_precision("high")

    batch_sizes = [1, 4, 16, 64, 256]
    modes = ["default", "reduce-overhead", "max-autotune", "max-autotune-no-cudagraphs"]

    model = MLP().to(device).eval()

    # Print header
    print(f"{'Batch':>6}", end="")
    for m in modes:
        print(f" {m:>24}", end="")
    print()
    print("-" * 110)

    for bs in batch_sizes:
        x = torch.randn(bs, 1024, device=device)

        with torch.no_grad():
            eager = benchmark(model, x)   # eager baseline in ms

        print(f"{bs:>6}", end="")
        print(f" {eager:>24.3f}", end="")  # show eager as baseline column

        for mode in modes:
            torch.compiler.reset()   # reset cache between mode tests
            compiled = torch.compile(model, mode=mode)
            with torch.no_grad():
                ms = benchmark(compiled, x)
            speedup = eager / ms
            print(f" {speedup:>23.2f}x", end="")

        print()

    print()
    print("Each cell = speedup over eager. Higher is better.")
    print("reduce-overhead is not guaranteed to work for all models.")
