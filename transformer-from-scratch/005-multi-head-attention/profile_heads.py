"""
Profile per-head attention patterns.

Shows which heads learn which kinds of patterns:
positional (diagonal), content-based (specific token pairs),
or uniform (no strong preference).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, bias=True):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.w_q = nn.Linear(d_model, d_model, bias=bias)
        self.w_k = nn.Linear(d_model, d_model, bias=bias)
        self.w_v = nn.Linear(d_model, d_model, bias=bias)
        self.w_o = nn.Linear(d_model, d_model, bias=bias)

    def forward(self, x):
        batch, seq_len, d_model = x.shape
        q = self.w_q(x)
        k = self.w_k(x)
        v = self.w_v(x)
        q = q.view(batch, seq_len, self.n_heads, self.d_head).transpose(1, 2)
        k = k.view(batch, seq_len, self.n_heads, self.d_head).transpose(1, 2)
        v = v.view(batch, seq_len, self.n_heads, self.d_head).transpose(1, 2)
        scale = math.sqrt(self.d_head)
        scores = q @ k.transpose(-2, -1) / scale
        attn_weights = torch.softmax(scores, dim=-1)
        attn = attn_weights @ v
        attn = attn.transpose(1, 2).contiguous()
        attn = attn.view(batch, seq_len, d_model)
        return self.w_o(attn)


def get_attn_weights(mha, x):
    """Extract per-head attention weights without running a full forward."""
    mha.eval()
    with torch.no_grad():
        q = mha.w_q(x)
        k = mha.w_k(x)
        v = mha.w_v(x)
        q = q.view(x.size(0), x.size(1), mha.n_heads, mha.d_head).transpose(1, 2)
        k = k.view(x.size(0), x.size(1), mha.n_heads, mha.d_head).transpose(1, 2)
        scale = math.sqrt(mha.d_head)
        scores = q @ k.transpose(-2, -1) / scale
        weights = torch.softmax(scores, dim=-1)
    return weights  # (batch, n_heads, seq, seq)


def classify_head(weights):
    """
    Classify what pattern an attention head is learning.

    Diagonal (positional): head attends strongly to nearby tokens.
    Column (content-based): head attends strongly to specific keys.
    Uniform: attention weights are roughly evenly spread.
    """
    seq_len = weights.shape[-1]

    # Diagonal score: mean weight along the diagonal (position i -> i).
    diag = torch.diagonal(weights, dim1=-2, dim2=-1)
    diag_score = diag.mean().item()

    # Off-diagonal spread: how peaked is the distribution?
    # Use a mask instead of fill_diagonal_ which requires 2D input.
    mask = ~torch.eye(seq_len, dtype=torch.bool, device=weights.device)
    off_diag = weights.masked_select(mask).max().item()

    # Uniform baseline: each position gets 1/seq_len.
    uniform = 1.0 / seq_len

    if diag_score > uniform * 2 and diag_score > off_diag:
        return "positional (diagonal)"
    elif off_diag > diag_score * 1.5:
        return "content-based (selective)"
    else:
        return "uniform / diffuse"


# ---------------------------------------------------------------------------
# Main demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    torch.manual_seed(123)

    batch, seq_len, d_model, n_heads = 2, 16, 64, 8
    x = torch.randn(batch, seq_len, d_model)

    mha = MultiHeadAttention(d_model, n_heads)
    weights = get_attn_weights(mha, x)

    print(f"Input: batch={batch}, seq_len={seq_len}, d_model={d_model}, n_heads={n_heads}")
    print(f"\nPer-head pattern classification:")
    print("-" * 45)

    for head in range(n_heads):
        # Average over batch to get a cleaner signal.
        w = weights[:, head, :, :].mean(0)  # (seq, seq)
        pattern = classify_head(w.unsqueeze(0))
        print(f"  Head {head:2d}: {pattern}")

        # Show the most-attended pairs (top-3 off-diagonal).
        # Zero out the diagonal before finding topk.
        w_no_diag = w.masked_fill(torch.eye(seq_len, dtype=torch.bool, device=w.device), 0)
        w_flat = w_no_diag.flatten()
        topk = w_flat.topk(3)
        for idx, val in zip(topk.indices, topk.values):
            i, j = idx.item() // seq_len, idx.item() % seq_len
            print(f"    pos {i} -> {j}: weight={val:.3f}")

    print("\nInterpretation:")
    print("  positional: head focuses on nearby tokens (diagonal pattern)")
    print("  content-based: head focuses on specific tokens regardless of position")
    print("  uniform: head distributes attention fairly evenly")
