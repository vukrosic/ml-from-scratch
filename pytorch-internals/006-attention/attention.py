"""Naive attention vs. torch.nn.functional.scaled_dot_product_attention.

Run:
    python attention.py
"""

import torch
import torch.nn.functional as F
import time


def naive_attention(Q, K, V, scale=None):
    """Scaled dot-product attention, computed naively.

    Q, K, V: (batch, seq, d_head)
    Returns: (batch, seq, d_head)

    This materializes the full (batch, seq, seq) attention matrix explicitly.
    """
    d = Q.shape[-1]
    if scale is None:
        scale = d ** -0.5

    # Step 1: attention scores
    scores = Q @ K.transpose(-2, -1)  # (batch, seq, seq)
    scores = scores * scale

    # Step 2: softmax
    weights = F.softmax(scores, dim=-1)

    # Step 3: apply to values
    return weights @ V


def fast_attention(Q, K, V):
    """Scaled dot-product attention using PyTorch's fused SDPA kernel.

    Q, K, V: (batch, seq, d_head)
    Returns: (batch, seq, d_head)

    Fuses the entire operation. Avoids explicitly materializing the
    (batch, seq, seq) attention matrix.
    """
    return F.scaled_dot_product_attention(Q, K, V)


def benchmark(fn, *args, steps=200, warmup=50):
    """Time a function call on GPU with proper synchronization."""
    for _ in range(warmup):
        fn(*args)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(steps):
        fn(*args)
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / steps


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Vary sequence length to show O(n²) scaling
    batch, d_head = 4, 64
    seq_lengths = [256, 512, 1024]

    for seq in seq_lengths:
        Q = torch.randn(batch, seq, d_head, device=device)
        K = torch.randn(batch, seq, d_head, device=device)
        V = torch.randn(batch, seq, d_head, device=device)

        naive_time = benchmark(naive_attention, Q, K, V)
        fast_time = benchmark(fast_attention, Q, K, V)

        print(f"\nSeq length {seq}:")
        print(f"  Naive:   {naive_time*1000:.2f} ms")
        print(f"  SDPA:    {fast_time*1000:.2f} ms")
        print(f"  Speedup: {naive_time/fast_time:.2f}x")


if __name__ == "__main__":
    main()
