"""
FeedForward Network from scratch — Linear -> GELU -> Linear.

This is the feedforward sub-layer inside every transformer block.
It operates position-wise: each token embedding goes through the same
two linear transformations independently.

GELU(x) = x * Phi(x) where Phi is the standard normal CDF.
We use the approximation from the GELU paper:
    GELU(x) ≈ 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715*x^3)))
"""

import math
import torch
import torch.nn as nn


def gelu(x):
    """
    Gaussian Error Linear Unit activation.

    GELU(x) = x * Phi(x) where Phi is the CDF of the standard normal distribution.
    We use the proven approximation from the original paper.
    For positive x, GELU is close to x * sigmoid(x). For negative x, it suppresses
    the value smoothly rather than zeroing it outright like ReLU.
    """
    cdf = 0.5 * (1.0 + torch.tanh(
        math.sqrt(2.0 / math.pi) * (x + 0.044715 * x ** 3)
    ))
    return x * cdf


class FeedForward(nn.Module):
    """
    Standard transformer feedforward layer.

    Architecture: Linear(d_model -> d_ff) -> GELU -> Dropout -> Linear(d_ff -> d_model)

    The expansion from d_model to d_ff (typically 4x) is where most of the
    transformer's parameters live.
    """

    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Expand: d_model -> d_ff
        x = self.linear1(x)
        # Non-linear activation
        x = gelu(x)
        # Regularize
        x = self.dropout(x)
        # Compress: d_ff -> d_model
        x = self.linear2(x)
        return x


if __name__ == "__main__":
    d_model = 128
    d_ff = 4 * d_model  # 512, standard expansion factor

    # Create FFN layer
    ffn = FeedForward(d_model, d_ff, dropout=0.0)
    print("FFN architecture:")
    print(f"  input:  Linear({d_model} -> {d_ff})")
    print(f"  act:    GELU")
    print(f"  output: Linear({d_ff} -> {d_model})")
    print(f"  params: {sum(p.numel() for p in ffn.parameters()):,}")

    # Forward pass
    batch, seq_len = 2, 10
    x = torch.randn(batch, seq_len, d_model)
    output = ffn(x)
    print(f"\nInput shape:  {x.shape}")
    print(f"Output shape: {output.shape}")

    # Verify GELU behavior
    print("\nGELU properties:")
    print(f"  gelu(1.0)  = {gelu(torch.tensor(1.0)).item():.4f}  (positive: ~x*sigmoid(x))")
    print(f"  gelu(-1.0) = {gelu(torch.tensor(-1.0)).item():.4f}  (negative: suppressed, not zero)")
    print(f"  gelu(0.0)  = {gelu(torch.tensor(0.0)).item():.4f}  (at zero)")
