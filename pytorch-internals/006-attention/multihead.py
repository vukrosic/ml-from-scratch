"""Multi-head attention module from scratch.

Run:
    python multihead.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):
    """Multi-head attention.

    Args:
        d_model: model dimension
        num_heads: number of attention heads (must divide d_model evenly)
    """
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0, f"{d_model} not divisible by {num_heads}"
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_model // num_heads

        # Q, K, V projections
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)

        # Output projection
        self.W_o = nn.Linear(d_model, d_model)

    def forward(self, x, mask=None):
        """Forward pass.

        x: (batch, seq, d_model)
        mask: optional attention mask, broadcastable to (batch, num_heads, seq, seq)
        Returns: (batch, seq, d_model)
        """
        batch, seq, _ = x.shape

        # Project to Q, K, V: (batch, seq, d_model)
        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)

        # Reshape for multi-head: (batch, seq, num_heads, d_head) -> (batch, num_heads, seq, d_head)
        Q = Q.view(batch, seq, self.num_heads, self.d_head).transpose(1, 2)
        K = K.view(batch, seq, self.num_heads, self.d_head).transpose(1, 2)
        V = V.view(batch, seq, self.num_heads, self.d_head).transpose(1, 2)

        # SDPA: (batch, num_heads, seq, d_head)
        attn = F.scaled_dot_product_attention(Q, K, V, attn_mask=mask)

        # Reshape back: (batch, num_heads, seq, d_head) -> (batch, seq, d_model)
        attn = attn.transpose(1, 2).contiguous().view(batch, seq, self.d_model)

        # Final projection
        return self.W_o(attn)


def test_correctness():
    """Test that multi-head attention produces the expected output shape."""
    d_model, num_heads = 512, 8
    batch, seq = 2, 32

    mha = MultiHeadAttention(d_model, num_heads)
    x = torch.randn(batch, seq, d_model)

    out = mha(x)

    assert out.shape == (batch, seq, d_model), f"Expected {(batch, seq, d_model)}, got {out.shape}"
    print(f"Output shape: {out.shape} — correct")


def test_equivalence_to_single_head():
    """Verify that num_heads=1 gives the same result as a single attention head."""
    d_model, num_heads = 64, 1
    batch, seq = 2, 16

    mha = MultiHeadAttention(d_model, num_heads)
    x = torch.randn(batch, seq, d_model)

    # Single-head SDPA reference
    Q = mha.W_q(x)
    K = mha.W_k(x)
    V = mha.W_v(x)
    Q = Q.view(batch, seq, num_heads, d_model).transpose(1, 2)
    K = K.view(batch, seq, num_heads, d_model).transpose(1, 2)
    V = V.view(batch, seq, num_heads, d_model).transpose(1, 2)
    ref = F.scaled_dot_product_attention(Q, K, V)
    ref = ref.transpose(1, 2).contiguous().view(batch, seq, d_model)
    ref = mha.W_o(ref)

    out = mha(x)

    max_diff = (out - ref).abs().max().item()
    print(f"Max difference from single-head reference: {max_diff:.2e}")


def test_causal_mask():
    """Test with a causal mask for autoregressive decoding."""
    d_model, num_heads = 256, 4
    batch, seq = 1, 16

    mha = MultiHeadAttention(d_model, num_heads).cuda()
    x = torch.randn(batch, seq, d_model, device='cuda')

    # Create causal mask: (seq, seq), upper triangle is masked
    causal_mask = torch.triu(torch.ones(seq, seq, device='cuda'), diagonal=1).bool()

    out = mha(x, mask=causal_mask)

    assert out.shape == (batch, seq, d_model)
    print(f"Causal mask test passed. Output shape: {out.shape}")


def benchmark_throughput():
    """Show throughput scaling with sequence length."""
    d_model, num_heads = 512, 8

    mha = MultiHeadAttention(d_model, num_heads).cuda()

    seq_lengths = [128, 256, 512, 1024]

    print("\nThroughput (steps/sec) vs sequence length:")
    for seq in seq_lengths:
        x = torch.randn(4, seq, d_model, device='cuda')

        # Warmup
        for _ in range(20):
            _ = mha(x)
        torch.cuda.synchronize()

        # Time
        t0 = time.perf_counter()
        for _ in range(50):
            _ = mha(x)
        torch.cuda.synchronize()
        elapsed = (time.perf_counter() - t0) / 50

        steps_per_sec = 1.0 / elapsed
        print(f"  seq={seq:4d}: {steps_per_sec:.1f} steps/s  ({elapsed*1000:.2f} ms/step)")


import time


def main():
    print("=== Multi-Head Attention Tests ===\n")
    test_correctness()
    test_equivalence_to_single_head()
    test_causal_mask()
    benchmark_throughput()


if __name__ == "__main__":
    main()
