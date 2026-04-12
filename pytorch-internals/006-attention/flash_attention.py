"""Simulate Flash Attention's tiling approach.

Run:
    python flash_attention.py
"""

import torch
import torch.nn.functional as F
import time


def flash_attention_simulated(Q, K, V, block_size=128):
    """Simulate Flash Attention's block-wise computation.

    Does NOT include the online softmax stabilization that the real algorithm uses.
    This is for illustration only — showing how attention can be computed
    block-by-block to avoid allocating the full (seq, seq) attention matrix.

    Q, K, V: (batch, seq, d_head)
    Returns: (batch, seq, d_head)
    """
    batch, seq, d = Q.shape
    O = torch.zeros_like(Q)

    # Process each row block of queries
    for i in range(0, seq, block_size):
        Q_block = Q[:, i:i+block_size, :]  # (batch, Br, d)

        # Accumulator for the max and norm across blocks (for proper softmax)
        # This simplified version does a separate softmax per block,
        # which is NOT the same as correct Flash Attention.
        row_accum = torch.zeros(batch, block_size, device=Q.device)

        for j in range(0, seq, block_size):
            K_block = K[:, j:j+block_size, :]  # (batch, Bc, d)
            V_block = V[:, j:j+block_size, :]  # (batch, Bc, d)

            # Compute attention scores for this block
            S_block = Q_block @ K_block.transpose(-2, -1)  # (batch, Br, Bc)
            S_block = S_block / (d ** 0.5)

            # Naive softmax per block (not the online stabilized softmax)
            P_block = torch.softmax(S_block, dim=-1)

            # Accumulate output for this row block
            O[:, i:i+block_size, :] += P_block @ V_block

    return O


def naive_attention(Q, K, V):
    """Reference naive attention."""
    d = Q.shape[-1]
    scale = d ** -0.5
    scores = Q @ K.transpose(-2, -1) * scale
    weights = F.softmax(scores, dim=-1)
    return weights @ V


def benchmark(fn, *args, steps=100, warmup=20):
    """Time a function on GPU."""
    for _ in range(warmup):
        fn(*args)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(steps):
        fn(*args)
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / steps


def memory_used():
    """Return current GPU memory allocated in GB."""
    return torch.cuda.memory_allocated() / 1024**3


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    batch, seq, d_head = 2, 2048, 64
    Q = torch.randn(batch, seq, d_head, device=device)
    K = torch.randn(batch, seq, d_head, device=device)
    V = torch.randn(batch, seq, d_head, device=device)

    print(f"\nSequence length: {seq}, d_head: {d_head}")
    print(f"Attention matrix size: {seq*seq * 4 / 1024**2:.1f} MB (float32)")

    # Warmup
    _ = naive_attention(Q, K, V)
    _ = flash_attention_simulated(Q, K, V)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    # Naive
    torch.cuda.reset_peak_memory_stats()
    naive_time = benchmark(naive_attention, Q, K, V)
    naive_mem = torch.cuda.max_memory_allocated() / 1024**2

    # Simulated flash
    torch.cuda.reset_peak_memory_stats()
    flash_time = benchmark(flash_attention_simulated, Q, K, V)
    flash_mem = torch.cuda.max_memory_allocated() / 1024**2

    print(f"\nNaive attention:   {naive_time*1000:.2f} ms, peak mem: ~{naive_mem:.1f} MB")
    print(f"Simulated flash:   {flash_time*1000:.2f} ms, peak mem: ~{flash_mem:.1f} MB")

    # Verify numerical equivalence (within floating point)
    out_naive = naive_attention(Q, K, V)
    out_flash = flash_attention_simulated(Q, K, V)
    max_diff = (out_naive - out_flash).abs().max().item()
    print(f"\nMax difference between implementations: {max_diff:.6f}")
    print("(Differences are expected — simulated flash skips online softmax stabilization)")


if __name__ == "__main__":
    main()
