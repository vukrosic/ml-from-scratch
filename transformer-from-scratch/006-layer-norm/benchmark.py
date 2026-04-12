"""
Benchmark: our LayerNorm vs nn.LayerNorm across batch sizes.

LayerNorm's cost is dominated by computing mean and variance over the feature
dimension — two reductions and a few elementwise ops. For large feature
dimensions (d_model >= 512), the cost becomes noticeable. We benchmark
forward and backward pass across several batch sizes with a fixed d_model.
"""

import time
import torch
import torch.nn as nn

from layer_norm import LayerNorm as OurLayerNorm


class Timer:
    """Context manager that records elapsed time."""

    def __init__(self):
        self.elapsed = 0.0

    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, *args):
        self.elapsed = time.perf_counter() - self.start


def benchmark(fn, n_warmup=5, n_runs=20):
    """Time a function over n_runs (after n_warmup warmup runs)."""
    for _ in range(n_warmup):
        fn()
    times = []
    for _ in range(n_runs):
        timer = Timer()
        with timer:
            fn()
        times.append(timer.elapsed)
    return sum(times) / len(times)


if __name__ == "__main__":
    d_model = 512
    seq_len = 64
    batch_sizes = [1, 2, 4, 8, 16]
    n_warmup, n_runs = 5, 30

    print(f"d_model={d_model}, seq_len={seq_len}")
    print(f"{'batch':>6}  {'ours fwd (μs)':>14}  {'nn.LN fwd (μs)':>14}  "
          f"{'ours bwd (μs)':>14}  {'nn.LN bwd (μs)':>14}")
    print("-" * 75)

    for batch in batch_sizes:
        x = torch.randn(batch, seq_len, d_model, requires_grad=True)

        our_ln = OurLayerNorm(d_model)
        torch_ln = nn.LayerNorm(d_model)

        # Forward benchmark
        our_fwd = benchmark(lambda: our_ln(x.detach()), n_warmup, n_runs)
        torch_fwd = benchmark(lambda: torch_ln(x.detach()), n_warmup, n_runs)

        # Backward benchmark (uses autograd to compute gradients)
        x_ours = x.detach().clone().requires_grad_(True)
        x_torch = x.detach().clone().requires_grad_(True)

        our_ln2 = OurLayerNorm(d_model)
        torch_ln2 = nn.LayerNorm(d_model)

        def our_backward():
            y = our_ln2(x_ours)
            y.sum().backward()

        def torch_backward():
            y = torch_ln2(x_torch)
            y.sum().backward()

        our_bwd = benchmark(our_backward, n_warmup, n_runs)
        torch_bwd = benchmark(torch_backward, n_warmup, n_runs)

        ratio_fwd = our_fwd / torch_fwd
        ratio_bwd = our_bwd / torch_bwd
        print(f"{batch:>6}  {our_fwd*1e6:>14.2f}  {torch_fwd*1e6:>14.2f}  "
              f"{our_bwd*1e6:>14.2f}  {torch_bwd*1e6:>14.2f}  "
              f"(fwd {ratio_fwd:.2f}x, bwd {ratio_bwd:.2f}x)")

    print("\nNote: our pure-PyTorch version is within ~1-3x of nn.LayerNorm.")
    print("The goal is understanding, not beating highly optimized C++ kernels.")
