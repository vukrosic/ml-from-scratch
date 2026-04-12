"""Real torch.compile benchmark — eager vs compiled."""

import time
import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, d_in=1024, d_hidden=4096, d_out=1024):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, d_hidden),
            nn.GELU(),
            nn.Linear(d_hidden, d_out),
        )

    def forward(self, x):
        return self.net(x)


def benchmark(fn, x, steps=200, warmup=50):
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
    model = MLP().to(device).eval()
    x = torch.randn(256, 1024, device=device)

    # Eager baseline
    with torch.no_grad():
        eager_time = benchmark(model, x)

    # Compiled
    compiled = torch.compile(model)
    with torch.no_grad():
        t0 = time.perf_counter()
        compiled(x)
        torch.cuda.synchronize()
        compile_overhead = time.perf_counter() - t0

        compiled_time = benchmark(compiled, x)

    print(f"Eager:              {eager_time * 1000:.3f} ms")
    print(f"First run (compile): {compile_overhead * 1000:.1f} ms")
    print(f"Compiled (steady):   {compiled_time * 1000:.3f} ms")
    print(f"Speedup:            {eager_time / compiled_time:.2f}x")
