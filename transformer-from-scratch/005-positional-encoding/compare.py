"""
Compare sinusoidal vs learned positional encoding visually.

Generates a heatmap of both PE schemes and shows the extrapolation problem
with learned PE: positions beyond training are meaningless.
"""

import math
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from sinusoidal import sinusoidal_pe_vectorized


# ---------------------------------------------------------------------------
# Learned PE class (reproduced so this file is standalone)
# ---------------------------------------------------------------------------

class LearnedPositionalEmbedding(nn.Module):
    def __init__(self, seq_len, d_model):
        super().__init__()
        self.seq_len = seq_len
        self.pe = nn.Embedding(seq_len, d_model)
        nn.init.normal_(self.pe.weight, mean=0.0, std=0.02)

    def forward(self, x):
        seq_len = x.shape[1]
        positions = torch.arange(seq_len, device=x.device)
        return x + self.pe(positions)


# ---------------------------------------------------------------------------
# PE visualisation
# ---------------------------------------------------------------------------

def plot_pe(pe_sin, pe_learn, save_path="pe_comparison.png"):
    """
    Plot sinusoidal and learned PE side by side as heatmaps.
    Also demonstrate extrapolation for learned PE.
    """
    seq_len, d_model = pe_sin.shape

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Sinusoidal PE
    axes[0].imshow(pe_sin.numpy(), cmap="RdBu", aspect="auto")
    axes[0].set_title("Sinusoidal PE\n(fixed, valid for any pos)")
    axes[0].set_xlabel("dimension")
    axes[0].set_ylabel("position")

    # Learned PE
    axes[1].imshow(pe_learn.numpy(), cmap="RdBu", aspect="auto")
    axes[1].set_title("Learned PE\n(trainable, finite positions)")
    axes[1].set_xlabel("dimension")
    axes[1].set_ylabel("position")

    # Extrapolation: learned PE at position 500 (never seen)
    pos_large = 500
    # Sinusoidal at position 500 (valid — formula works for any integer)
    sin_at_500 = torch.zeros(d_model)
    freqs_idx = torch.arange(0, d_model, 2).float()
    freqs = torch.exp(-math.log(10000.0) * freqs_idx / d_model)
    angles = pos_large * freqs
    sin_at_500[0::2] = torch.sin(angles)
    sin_at_500[1::2] = torch.cos(angles)

    # Learned PE: clamp to max position
    learned_at_500 = pe_learn[min(pos_large, seq_len - 1)]

    diff = (sin_at_500 - learned_at_500).abs().mean().item()
    axes[2].bar(["sin @ 500\n(valid formula)"],
                [sin_at_500.abs().mean().item()],
                color="steelblue", label="sinusoidal")
    axes[2].bar(["learned @ 500\n(clamped to 7)"],
                [learned_at_500.abs().mean().item()],
                color="tomato", label="learned")
    axes[2].set_title("PE value at position 500\n(never in training)")
    axes[2].set_ylabel("mean |value|")
    axes[2].legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved {save_path}")


# ---------------------------------------------------------------------------
# Show dot-product similarity between positions
# ---------------------------------------------------------------------------

def position_similarity(pe, title, save_path="pe_similarity.png"):
    """
    Compute cosine similarity between PE vectors at different positions.
    Shows how "similar" position i is to position j.
    """
    # Normalise rows to unit length
    norms = pe / pe.norm(dim=-1, keepdim=True)
    sim = norms @ norms.T  # (seq, seq)

    plt.figure(figsize=(5, 4))
    plt.imshow(sim.numpy(), cmap="viridis", vmin=0, vmax=1)
    plt.colorbar(label="cosine similarity")
    plt.title(title)
    plt.xlabel("position")
    plt.ylabel("position")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved {save_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    seq_len, d_model = 32, 64

    # Sinusoidal
    sin_pe = sinusoidal_pe_vectorized(seq_len, d_model)

    # Learned (randomly initialised — training would change this)
    learned_model = LearnedPositionalEmbedding(seq_len, d_model)
    torch.manual_seed(0)
    learned_pe = learned_model.pe.weight.detach()

    # Plot
    plot_pe(sin_pe, learned_pe)

    # Similarity matrices
    position_similarity(sin_pe, "Sinusoidal PE: position similarity")
    position_similarity(learned_pe, "Learned PE: position similarity (untrained)")

    print("\n--- Position dot-product similarity (sinusoidal) ---")
    norms = sin_pe / sin_pe.norm(dim=-1, keepdim=True)
    sim = norms @ norms.T
    print("Diagonal = 1.0 (each position perfectly similar to itself)")
    print("Nearby positions have higher similarity than distant ones (by design).")
    print("\nFirst 5x5 similarity block (sinusoidal):")
    print(torch.round(sim[:5, :5], decimals=3))

    print("\n--- Key insight ---")
    print("Learned PE for position > seq_len is undefined / clamped.")
    print("Sinusoidal PE produces a valid vector for ANY integer position.")
