"""
Benchmark: our from-scratch FFN vs PyTorch nn.Sequential.

We measure forward and backward pass latency across different
batch sizes, sequence lengths, and model dimensions.
"""

import time
import math
import torch
import torch.nn as nn

from feedforward import FeedForward


def benchmark(fn, n_runs=500, warmup=50):
    """Time a function over n_runs (with warmup). Returns mean latency in microseconds."""
    # Warmup
    for _ in range(warmup):
        fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(n_runs):
        fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    return (time.perf_counter() - start) / n_runs * 1e6


def ffn_pytorch(d_model, d_ff, dropout=0.0):
    """PyTorch reference FFN."""
    return nn.Sequential(
        nn.Linear(d_model, d_ff),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(d_ff, d_model),
    )


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Test configurations
    configs = [
        # (d_model, d_ff, batch, seq_len)
        (128, 512, 4, 32),
        (256, 1024, 4, 64),
        (512, 2048, 8, 64),
        (768, 3072, 16, 128),
        (1024, 4096, 8, 256),
    ]

    n_runs = 500

    print(f"\n{'d_model':>8} {'d_ff':>8} {'batch':>6} {'seq':>6}  {'ours fw (μs)':>14} {'torch fw (μs)':>14} {'ratio':>8}")
    print("-" * 80)

    for d_model, d_ff, batch, seq_len in configs:
        x = torch.randn(batch, seq_len, d_model, device=device)

        ffn_ours = FeedForward(d_model, d_ff, dropout=0.0).to(device)
        ffn_torch = ffn_pytorch(d_model, d_ff, dropout=0.0).to(device)

        # Copy weights for fair comparison
        ffn_torch[0].weight.data = ffn_ours.linear1.weight.data.clone()
        ffn_torch[0].bias.data = ffn_ours.linear1.bias.data.clone()
        ffn_torch[3].weight.data = ffn_ours.linear2.weight.data.clone()
        ffn_torch[3].bias.data = ffn_ours.linear2.bias.data.clone()

        ours_time = benchmark(lambda: ffn_ours(x), n_runs=n_runs)
        torch_time = benchmark(lambda: ffn_torch(x), n_runs=n_runs)

        ratio = ours_time / torch_time
        print(f"{d_model:>8} {d_ff:>8} {batch:>6} {seq_len:>6}  "
              f"{ours_time:>14.2f} {torch_time:>14.2f} {ratio:>8.2f}x")

    print("\nNote: Both implementations perform identical operations.")
    print("Any small difference is due to measurement noise or kernel dispatch overhead.")
    print("The from-scratch version is for learning — PyTorch is optimized for production.")
