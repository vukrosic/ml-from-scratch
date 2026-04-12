"""
Visualize attention weights as a heatmap.
Shows attention patterns for both standard and causal self-attention
on a random sequence, then saves the figure.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for saving files.

# ---------------------------------------------------------------------------
# From-scratch SelfAttention (same as attention.py)
# ---------------------------------------------------------------------------

class SelfAttention(nn.Module):
    def __init__(self, d_model, n_heads, bias=True):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.qkv_proj = nn.Linear(d_model, 3 * d_model, bias=bias)
        self.out_proj = nn.Linear(d_model, d_model, bias=bias)

    def forward(self, x, return_weights=False):
        batch, seq_len, d_model = x.shape
        qkv = self.qkv_proj(x)
        qkv = qkv.view(batch, seq_len, self.n_heads, 3 * self.d_head)
        qkv = qkv.transpose(1, 2)
        q, k, v = qkv.chunk(3, dim=-1)
        scale = math.sqrt(self.d_head)
        scores = q @ k.transpose(-2, -1) / scale
        attn_weights = F.softmax(scores, dim=-1)
        attn = attn_weights @ v
        attn = attn.transpose(1, 2).contiguous()
        attn = attn.view(batch, seq_len, d_model)
        out = self.out_proj(attn)
        if return_weights:
            return out, attn_weights
        return out


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model, n_heads, bias=True):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.qkv_proj = nn.Linear(d_model, 3 * d_model, bias=bias)
        self.out_proj = nn.Linear(d_model, d_model, bias=bias)

    def forward(self, x, return_weights=False):
        batch, seq_len, d_model = x.shape
        qkv = self.qkv_proj(x)
        qkv = qkv.view(batch, seq_len, self.n_heads, 3 * self.d_head)
        qkv = qkv.transpose(1, 2)
        q, k, v = qkv.chunk(3, dim=-1)
        scale = math.sqrt(self.d_head)
        scores = q @ k.transpose(-2, -1) / scale
        causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool))
        scores = scores.masked_fill(~causal_mask, float('-inf'))
        attn_weights = F.softmax(scores, dim=-1)
        attn = attn_weights @ v
        attn = attn.transpose(1, 2).contiguous()
        attn = attn.view(batch, seq_len, d_model)
        out = self.out_proj(attn)
        if return_weights:
            return out, attn_weights
        return out


# ---------------------------------------------------------------------------
# Attention visualization
# ---------------------------------------------------------------------------

def plot_attention_weights(weights, title, ax, seq_labels=None):
    """
    Render one attention weight matrix as a heatmap.
    weights shape: (seq_len, seq_len) or (n_heads, seq_len, seq_len).
    If n_heads > 1, shows the mean across heads.
    """
    # Average over heads for a single summary heatmap.
    if weights.ndim == 3:
        weights = weights.mean(dim=0)  # (seq_len, seq_len)

    im = ax.imshow(weights.cpu().numpy(), cmap="viridis", aspect="auto")
    ax.set_title(title)
    ax.set_xlabel("Key position")
    ax.set_ylabel("Query position")
    if seq_labels is not None:
        ax.set_xticks(range(len(seq_labels)))
        ax.set_yticks(range(len(seq_labels)))
        ax.set_xticklabels(seq_labels, rotation=45, fontsize=7)
        ax.set_yticklabels(seq_labels, fontsize=7)
    plt.colorbar(im, ax=ax, shrink=0.8)


def plot_all_heads(weights, n_heads, seq_labels=None):
    """
    Render attention weights as a grid of heatmaps, one per head.
    weights shape: (n_heads, seq_len, seq_len).
    Returns a matplotlib Figure.
    """
    seq_len = weights.shape[1]
    cols = min(n_heads, 4)
    rows = (n_heads + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3 * rows))
    axes = axes.flatten() if n_heads > 1 else [axes]

    for i in range(n_heads):
        im = axes[i].imshow(weights[i].cpu().numpy(), cmap="viridis", aspect="auto")
        axes[i].set_title(f"Head {i+1}")
        axes[i].set_xlabel("Key")
        axes[i].set_ylabel("Query")
        if seq_labels is not None:
            axes[i].set_xticks(range(len(seq_labels)))
            axes[i].set_yticks(range(len(seq_labels)))
            axes[i].set_xticklabels(seq_labels, rotation=45, fontsize=6)
            axes[i].set_yticklabels(seq_labels, fontsize=6)
        plt.colorbar(im, ax=axes[i], shrink=0.6)

    # Hide unused axes
    for j in range(n_heads, len(axes)):
        axes[j].axis("off")

    return fig


# ---------------------------------------------------------------------------
# Standalone demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    torch.manual_seed(123)

    # We use a short sequence of "words" as our tokens, for display labels.
    seq_len = 8
    d_model = 16
    n_heads = 4
    seq_labels = [f"tok_{i}" for i in range(seq_len)]

    # Random input (batch=1).
    x = torch.randn(1, seq_len, d_model)

    # ---- Standard self-attention ----
    attn = SelfAttention(d_model, n_heads)
    attn.eval()
    with torch.no_grad():
        _, weights_standard = attn(x, return_weights=True)
    # weights_standard shape: (batch=1, n_heads, seq_len, seq_len)
    weights_standard = weights_standard[0]  # (n_heads, seq_len, seq_len)

    # ---- Causal self-attention ----
    causal_attn = CausalSelfAttention(d_model, n_heads)
    causal_attn.eval()
    with torch.no_grad():
        _, weights_causal = causal_attn(x, return_weights=True)
    weights_causal = weights_causal[0]

    # ---- Main figure: mean attention weights ----
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Self-Attention Weights — Mean Over Heads", fontsize=14, fontweight="bold")

    plot_attention_weights(
        weights_standard.mean(dim=0, keepdim=True).squeeze(0),
        "Standard Self-Attention",
        axes[0],
        seq_labels=seq_labels,
    )
    plot_attention_weights(
        weights_causal.mean(dim=0, keepdim=True).squeeze(0),
        "Causal (Masked) Self-Attention",
        axes[1],
        seq_labels=seq_labels,
    )
    plt.tight_layout()
    out_path = "/root/ml-from-scratch/transformer-from-scratch/002-self-attention/attention_weights.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved mean-attention heatmap to: {out_path}")

    # ---- Separate per-head figures ----
    fig_standard = plot_all_heads(weights_standard, n_heads, seq_labels=seq_labels)
    fig_standard.suptitle("Standard Self-Attention — All Heads", fontsize=12)
    fig_standard.savefig(
        "/root/ml-from-scratch/transformer-from-scratch/002-self-attention/attention_weights_standard_heads.png",
        dpi=150, bbox_inches="tight"
    )
    print("Saved standard per-head figure to: attention_weights_standard_heads.png")

    fig_causal = plot_all_heads(weights_causal, n_heads, seq_labels=seq_labels)
    fig_causal.suptitle("Causal Self-Attention — All Heads", fontsize=12)
    fig_causal.savefig(
        "/root/ml-from-scratch/transformer-from-scratch/002-self-attention/attention_weights_causal_heads.png",
        dpi=150, bbox_inches="tight"
    )
    print("Saved causal per-head figure to: attention_weights_causal_heads.png")

    # Print the raw weights for the first head so the user can verify manually.
    print("\nStandard attention weights, head=0:")
    print(weights_standard[0].numpy().round(3))
    print("\nCausal attention weights, head=0:")
    print(weights_causal[0].numpy().round(3))
    print("\nNotice the causal matrix is lower-triangular — token i never attends to j > i.")
