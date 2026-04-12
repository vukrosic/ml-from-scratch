"""
ALiBi: Attention with Linear Biases — from "Train Short, Test Long: Tuning AI"

ALiBi avoids positional embeddings entirely. Instead, it adds a linear bias to
the attention score that is proportional to the distance between query and key.
The bias for query at position i attending to key at position j is: -|i-j| * m
where m is a head-specific slope, often powers of 2: 1/2, 1/4, 1/8, ...

Key insight: no learned position table, works for any sequence length,
and naturally encodes relative position through the bias slope per head.

Attention score with ALiBi:
  score[i,j] = (q_i · k_j) / sqrt(d_head) - |i-j| * m_head
"""

import math
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Build the ALiBi bias matrix
# ---------------------------------------------------------------------------

def build_alibi_bias(n_heads, seq_len, m=None):
    """
    Build the (n_heads, seq_len, seq_len) ALiBi bias matrix.
    Each head gets a different slope m: head h uses slope m^h.
    m is often 1/2 (powers of 2 per head).
    Returns: (n_heads, seq_len, seq_len) bias to SUBTRACT from attention scores.
    """
    if m is None:
        m = 0.5  # standard: powers of 1/2

    # slopes per head: m^0, m^1, ..., m^(n_heads-1)
    slopes = torch.tensor([m ** h for h in range(n_heads)])  # (n_heads,)

    # Distance matrix: |i - j| for all pairs
    positions_i = torch.arange(seq_len).unsqueeze(1)  # (seq, 1)
    positions_j = torch.arange(seq_len).unsqueeze(0)  # (1, seq)
    distances = (positions_i - positions_j).abs()    # (seq, seq)

    # Bias per head: -|i-j| * slope[head]
    bias = -distances.unsqueeze(0) * slopes.view(n_heads, 1, 1)  # (n_heads, seq, seq)
    return bias  # negative because we SUBTRACT from attention scores


# ---------------------------------------------------------------------------
# ALiBi Multi-Head Attention
# ---------------------------------------------------------------------------

class ALiBiMultiHeadAttention(nn.Module):
    """
    Multi-head attention with ALiBi (Attention with Linear Biases).
    No positional embeddings — position is encoded through a distance-based
    linear penalty applied to attention scores before softmax.
    """
    def __init__(self, d_model, n_heads, m=0.5):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.m = m
        self.qkv_proj = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x):
        batch, seq_len, d_model = x.shape

        qkv = self.qkv_proj(x)
        qkv = qkv.view(batch, seq_len, self.n_heads, 3 * self.d_head)
        qkv = qkv.transpose(1, 2)
        q, k, v = qkv.chunk(3, dim=-1)

        scale = math.sqrt(self.d_head)
        scores = q @ k.transpose(-2, -1) / scale  # (batch, heads, seq, seq)

        # Build ALiBi bias — independent of the input values, only depends on geometry
        alibi_bias = build_alibi_bias(self.n_heads, seq_len, self.m)
        alibi_bias = alibi_bias.to(scores.dtype).to(scores.device)

        # Subtract bias (more distant positions get penalised more)
        scores = scores + alibi_bias.unsqueeze(0)  # broadcast over batch

        attn = torch.softmax(scores, dim=-1) @ v
        attn = attn.transpose(1, 2).contiguous().view(batch, seq_len, d_model)
        return self.out_proj(attn)


# ---------------------------------------------------------------------------
# Visualise the ALiBi bias pattern
# ---------------------------------------------------------------------------

def plot_alibi_bias(n_heads=8, seq_len=16, save_path="alibi_bias.png"):
    """Visualise the ALiBi bias matrix — distance penalty per head."""
    import matplotlib.pyplot as plt

    bias = build_alibi_bias(n_heads, seq_len)  # (n_heads, seq, seq)
    # bias is negative (penalty), take absolute for heatmap
    bias_abs = bias.abs()

    fig, axes = plt.subplots(2, 4, figsize=(14, 6), sharex=True, sharey=True)
    axes = axes.flatten()
    for h in range(n_heads):
        axes[h].imshow(bias_abs[h].numpy(), cmap="Reds")
        axes[h].set_title(f"head {h}\nslope = 0.5^{h} = {0.5**h:.4f}")
        axes[h].set_xlabel("key position")
        axes[h].set_ylabel("query position")
    plt.suptitle("ALiBi Bias Magnitude |i-j| × slope — darker = more penalty",
                 fontsize=11)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved {save_path}")


# ---------------------------------------------------------------------------
# Verify: more distant positions get lower attention
# ---------------------------------------------------------------------------

def verify_alibi_attention_pattern(n_heads=4, seq_len=8):
    """
    With ALiBi, attention to distant tokens is penalised.
    Verify that for each head, attention weights decrease with distance.
    """
    alibi = ALiBiMultiHeadAttention(d_model=64, n_heads=n_heads, m=0.5)
    alibi.eval()
    x = torch.randn(1, seq_len, 64)

    with torch.no_grad():
        qkv = alibi.qkv_proj(x)
        qkv = qkv.view(1, seq_len, n_heads, 3 * alibi.d_head)
        qkv = qkv.transpose(1, 2)
        q, k, _ = qkv.chunk(3, dim=-1)
        scale = math.sqrt(alibi.d_head)
        scores = q @ k.transpose(-2, -1) / scale
        alibi_bias = build_alibi_bias(n_heads, seq_len, alibi.m).to(scores)
        scores = scores + alibi_bias.unsqueeze(0)
        attn = torch.softmax(scores, dim=-1)

    # For head 0, position 3: check that attention decreases with distance
    attn_head0 = attn[0, 0, 3]  # attending from position 3 to all positions
    print("Head 0, query at position 3 — attention to each key position:")
    for j in range(seq_len):
        dist = abs(3 - j)
        print(f"  to pos {j} (dist={dist}): {attn_head0[j].item():.4f}")

    # Verify monotonic: closer positions get more attention
    monotonic = all(attn_head0[j] >= attn_head0[j+1] for j in range(seq_len-1))
    print(f"\nAttention decreases monotonically with distance: {monotonic}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    n_heads, seq_len = 4, 8

    # ALiBi attention
    alibi = ALiBiMultiHeadAttention(d_model=64, n_heads=n_heads)
    x = torch.randn(2, seq_len, 64)
    out = alibi(x)
    print(f"Input shape:  {x.shape}")
    print(f"Output shape: {out.shape}")
    assert out.shape == x.shape

    # Visualise bias
    plot_alibi_bias(n_heads=8, seq_len=16)

    # Verify distance penalty
    print("\n--- ALiBi distance penalty verification ---")
    verify_alibi_attention_pattern(n_heads, seq_len)
    print("\nNo positional embedding table — position is a property of")
    print("attention geometry, not a learned representation.")
