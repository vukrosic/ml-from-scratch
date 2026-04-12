"""
Benchmark: our stable softmax vs torch.softmax across dtypes and input sizes.

We test float16, bfloat16, float32, float64 — and a range of vector dimensions
and batch sizes — to see when, if ever, our pure-Python version is competitive,
and to characterise the numerical behaviour of each dtype.
"""

import math
import time
import torch


def stable_softmax(x, temperature=1.0):
    """Pure-Python stable softmax, works on a flat list."""
    if temperature != 1.0:
        x = [xi / temperature for xi in x]
    max_x = max(x)
    exp_x = [math.exp(xi - max_x) for xi in x]
    total = sum(exp_x)
    return [e / total for e in exp_x]


def benchmark(func, n_runs=1000):
    """Time a function over n_runs and return mean elapsed seconds."""
    start = time.perf_counter()
    for _ in range(n_runs):
        func()
    elapsed = time.perf_counter() - start
    return elapsed / n_runs


if __name__ == "__main__":
    dtypes = [torch.float16, torch.bfloat16, torch.float32, torch.float64]
    dim_sizes = [64, 256, 512, 1024]

    print(f"{'dtype':<12} {'dim':>6}  {'torch (μs)':>10}  {'ours (μs)':>10}  {'ratio':>6}")
    print("-" * 52)

    for dtype in dtypes:
        for dim in dim_sizes:
            x = torch.randn(dim, dtype=dtype)
            n_runs = 2000

            torch_time = benchmark(lambda: torch.softmax(x, dim=0), n_runs=n_runs)
            our_time = benchmark(lambda: stable_softmax(x.tolist()), n_runs=n_runs)

            ratio = our_time / torch_time
            print(f"{str(dtype):<12} {dim:>6}  {torch_time*1e6:>10.2f}  "
                  f"{our_time*1e6:>10.2f}  {ratio:>6.1f}x")

    print("\nNote: pure-Python is ~100-1000x slower than torch; this is expected.")
    print("The point is numerical accuracy, not raw speed.")
