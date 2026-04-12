"""
Cross-Attention Head Analysis.

Measures which encoder tokens each cross-attention head attends to.
This analysis reveals how different heads specialize — some may attend
to specific syntactic roles, positional patterns, or entity mentions.

Extended topic: paid Skool video explains the analysis methodology
and what these attention patterns mean for translation/generation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# ---------------------------------------------------------------------------
# Cross-Attention with accessible weights
# ---------------------------------------------------------------------------

class CrossAttentionWithWeights(nn.Module):
    """
    Cross-attention that stores attention weights for analysis.
    """

    def __init__(self, d_model, n_heads, bias=True):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.q_proj = nn.Linear(d_model, d_model, bias=bias)
        self.k_proj = nn.Linear(d_model, d_model, bias=bias)
        self.v_proj = nn.Linear(d_model, d_model, bias=bias)
        self.out_proj = nn.Linear(d_model, d_model, bias=bias)
        self.attn_weights = None  # Store for analysis

    def forward(self, dec_x, enc_x):
        batch, dec_seq, d_model = dec_x.shape
        enc_seq = enc_x.shape[1]
        q = self.q_proj(dec_x).view(batch, dec_seq, self.n_heads, self.d_head).transpose(1, 2)
        k = self.k_proj(enc_x).view(batch, enc_seq, self.n_heads, self.d_head).transpose(1, 2)
        v = self.v_proj(enc_x).view(batch, enc_seq, self.n_heads, self.d_head).transpose(1, 2)
        scale = math.sqrt(self.d_head)
        scores = q @ k.transpose(-2, -1) / scale
        self.attn_weights = F.softmax(scores, dim=-1)
        attn = self.attn_weights @ v
        attn = attn.transpose(1, 2).contiguous().view(batch, dec_seq, d_model)
        return self.out_proj(attn)


class DecoderBlockWithCrossAnalysis(nn.Module):
    """
    Decoder block with cross-attention that stores weights for analysis.
    """

    def __init__(self, d_model, n_heads, bias=True):
        super().__init__()
        self.masked_attn = MaskedSelfAttention(d_model, n_heads, bias=bias)
        self.cross_attn = CrossAttentionWithWeights(d_model, n_heads, bias=bias)
        self.ffn = FeedForward(d_model, 4 * d_model, bias=bias)
        self.ln1 = LayerNorm(d_model)
        self.ln2 = LayerNorm(d_model)
        self.ln3 = LayerNorm(d_model)

    def forward(self, x, enc_x=None):
        x = self.ln1(x + self.masked_attn(x))
        if enc_x is not None:
            x = self.ln2(x + self.cross_attn(x, enc_x))
        x = self.ln3(x + self.ffn(x))
        return x


class LayerNorm(nn.Module):
    def __init__(self, d_model, eps=1e-5):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        return self.gamma * (x - mean) / torch.sqrt(var + self.eps) + self.beta


class MaskedSelfAttention(nn.Module):
    def __init__(self, d_model, n_heads, bias=True):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.qkv_proj = nn.Linear(d_model, 3 * d_model, bias=bias)
        self.out_proj = nn.Linear(d_model, d_model, bias=bias)

    def forward(self, x):
        batch, seq_len, d_model = x.shape
        qkv = self.qkv_proj(x)
        qkv = qkv.view(batch, seq_len, self.n_heads, 3 * self.d_head).transpose(1, 2)
        q, k, v = qkv.chunk(3, dim=-1)
        scale = math.sqrt(self.d_head)
        scores = q @ k.transpose(-2, -1) / scale
        causal_mask = torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool, device=x.device), diagonal=1)
        scores = scores.masked_fill(causal_mask, float('-inf'))
        attn = F.softmax(scores, dim=-1) @ v
        attn = attn.transpose(1, 2).contiguous().view(batch, seq_len, d_model)
        return self.out_proj(attn)


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff=None, bias=True):
        super().__init__()
        d_ff = d_ff or 4 * d_model
        self.w1 = nn.Linear(d_model, d_ff, bias=bias)
        self.w2 = nn.Linear(d_ff, d_model, bias=bias)

    def forward(self, x):
        return self.w2(F.gelu(self.w1(x)))


# ---------------------------------------------------------------------------
# Analysis functions
# ---------------------------------------------------------------------------

def analyze_cross_attention_heads(block, dec_x, enc_x, batch_idx=0):
    """
    Analyze which encoder tokens each cross-attention head attends to.

    Args:
        block: DecoderBlockWithCrossAnalysis
        dec_x: decoder input (batch, dec_seq, d_model)
        enc_x: encoder input (batch, enc_seq, d_model)
        batch_idx: which batch element to analyze
    Returns:
        Dictionary with per-head statistics
    """
    _ = block(dec_x, enc_x)
    weights = block.cross_attn.attn_weights  # (batch, n_heads, dec_seq, enc_seq)

    n_heads = weights.shape[1]
    dec_seq = weights.shape[2]
    enc_seq = weights.shape[3]

    print(f"\nCross-Attention Analysis")
    print(f"  Batch index: {batch_idx}")
    print(f"  Heads: {n_heads}, Decoder seq: {dec_seq}, Encoder seq: {enc_seq}")
    print()

    # For each head, find which encoder positions get the most attention
    # averaged across all decoder positions
    head_stats = {}
    for head_idx in range(n_heads):
        # Average attention across decoder positions: (enc_seq,)
        avg_attn = weights[batch_idx, head_idx].mean(dim=0)  # mean over dec positions
        # Find position with max attention
        max_pos = avg_attn.argmax().item()
        max_val = avg_attn.max().item()
        # Entropy (higher = more distributed attention)
        entropy = -(avg_attn * torch.log(avg_attn + 1e-10)).sum().item()
        # Concentration (what fraction of attention goes to top-k positions)
        topk_vals, _ = avg_attn.topk(min(3, enc_seq))
        concentration = topk_vals.sum().item()

        head_stats[head_idx] = {
            'max_pos': max_pos,
            'max_val': max_val,
            'entropy': entropy,
            'concentration': concentration,
            'avg_attn': avg_attn
        }

        print(f"  Head {head_idx:2d}: max_pos={max_pos:3d} (val={max_val:.3f}), "
              f"entropy={entropy:.2f}, top-3 conc={concentration:.3f}")

    return head_stats


def visualize_attention_pattern(weights, dec_seq, enc_seq, head_idx=0, batch_idx=0):
    """
    Print a text visualization of attention pattern for a specific head.
    """
    attn = weights[batch_idx, head_idx]  # (dec_seq, enc_seq)
    print(f"\nAttention heatmap for head {head_idx} (batch {batch_idx}):")
    print(f"  Decoder seq (rows) x Encoder seq (cols)")
    print("  " + "".join(f"{j:3d}" for j in range(enc_seq)))
    for i in range(min(dec_seq, 10)):  # cap at 10 for readability
        row = attn[i]
        values = "".join(f"{v.item():3.1f}" for v in row[:min(enc_seq, 20)])
        print(f"  [{i:2d}] {values}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    torch.manual_seed(42)

    # Config
    batch, dec_seq, enc_seq = 2, 10, 8
    d_model, n_heads = 64, 4

    # Synthetic inputs with structure (simulating real data)
    dec_x = torch.randn(batch, dec_seq, d_model)
    enc_x = torch.randn(batch, enc_seq, d_model)

    # Decoder block with cross-attention analysis
    block = DecoderBlockWithCrossAnalysis(d_model, n_heads)
    block.eval()

    print("=" * 60)
    print("Cross-Attention Head Analysis")
    print("=" * 60)

    # Analyze heads
    head_stats = analyze_cross_attention_heads(block, dec_x, enc_x)

    # Visualize first head's attention
    weights = block.cross_attn.attn_weights
    visualize_attention_pattern(weights, dec_seq, enc_seq, head_idx=0)

    print("\n" + "=" * 60)
    print("Interpretation:")
    print("  - max_pos: which encoder token this head focuses on most")
    print("  - entropy: how distributed the attention is (higher = more spread)")
    print("  - top-3 conc: what fraction of attention goes to top 3 positions")
    print("=" * 60)
