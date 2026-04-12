"""
Learned Positional Embeddings.

Instead of using fixed sin/cos, we let the model learn position representations
via a trainable lookup table (nn.Embedding). The table maps each integer
position to a d_model-dimensional vector, and training adjusts those vectors.

Tradeoff: flexible but doesn't generalise to unseen sequence lengths.
"""

import torch
import torch.nn as nn


class LearnedPositionalEmbedding(nn.Module):
    """
    Learned positional encoding: a trainable (seq_len, d_model) matrix.
    During training the model learns the best representation of position.
    During inference it can only represent positions 0 .. seq_len-1.
    """
    def __init__(self, seq_len, d_model):
        super().__init__()
        self.seq_len = seq_len
        self.pe = nn.Embedding(seq_len, d_model)
        # Initialise sensibly: small values like token embeddings
        nn.init.normal_(self.pe.weight, mean=0.0, std=0.02)

    def forward(self, x):
        """
        Add positional encoding to token embeddings.
        x shape: (batch, seq_len, d_model)
        returns: (batch, seq_len, d_model)
        """
        seq_len = x.shape[1]
        positions = torch.arange(seq_len, device=x.device)
        return x + self.pe(positions)


# ---------------------------------------------------------------------------
# Demo: learned vs sinusoidal
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    batch, seq_len, d_model = 2, 8, 16

    # Simulate token embeddings (e.g. from a token embedding layer)
    torch.manual_seed(42)
    x = torch.randn(batch, seq_len, d_model)

    # Sinusoidal (fixed)
    from sinusoidal import sinusoidal_pe_vectorized
    sin_pe = sinusoidal_pe_vectorized(seq_len, d_model)

    # Learned
    learned_pe = LearnedPositionalEmbedding(seq_len, d_model)
    out_learned = learned_pe(x)

    # Verify shapes
    assert out_learned.shape == x.shape
    print(f"Input shape:  {x.shape}")
    print(f"Output shape: {out_learned.shape}")

    # Inspect the learned table
    print(f"\nLearned PE table shape: {learned_pe.pe.weight.shape}")
    print("Learned PE table (first 3 positions):")
    print(torch.round(learned_pe.pe.weight[:3], decimals=3))

    # Check: does the learned PE differ from sinusoidal?
    print(f"\n||learned_PE - sin_PE|| = "
          f"{(learned_pe.pe.weight - sin_pe).abs().mean():.4f}")
    print("Learned and sinusoidal PE are different — as expected.")

    # Show the extrapolation problem
    print("\n--- Extrapolation demo ---")
    # Try to get position 500 from a table trained only on 0..7
    small_table = nn.Embedding(8, d_model)
    pos_500_idx = torch.tensor([500])
    # This silently wraps around if we go out of bounds (or we can use clamp)
    # nn.Embedding by itself does not bounds-check — it just indexes.
    # The key issue is the model was never trained on position 500.
    print(f"Position 500 index (clamped to 7): {pos_500_idx.clamp(max=7).item()}")
    print("The model never saw position 500 during training — what it outputs is meaningless.")
