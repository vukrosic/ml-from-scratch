"""Benchmark all torch.compile modes across batch sizes."""

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
    for _ in range(warmup):
        fn(x)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(steps):
        fn(x)
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / steps


if __name__ == "__main__":
    device = "cuda"
    torch.set_float32_matmul_precision("high")

    batch_sizes = [1, 4, 16, 64, 256]
    modes = ["default", "reduce-overhead", "max-autotune", "max-autotune-no-cudagraphs"]

    model = MLP().to(device).eval()

    # Eager baseline
    print(f"{'Batch':>6}", end="")
    for m in modes:
        print(f" {m:>20}", end="")
    print()
    print("-" * 90)

    for bs in batch_sizes:
        x = torch.randn(bs, 1024, device=device)

        with torch.no_grad():
            eager = benchmark(model, x) * 1000

        print(f"{bs:>6}", end="")
        print(f" {eager:>20.3f}", end="")

        for mode in modes:
            torch.compiler.reset()
            compiled = torch.compile(model, mode=mode)
            with torch.no_grad():
                ms = benchmark(compiled, x) * 1000
            speedup = eager / ms
            print(f" {speedup:>19.2f}x", end="")

        print()

    print()
    print("Each cell = speedup over eager. Higher is better.")
    print("Note: reduce-overhead is not guaranteed to work for all models.")
