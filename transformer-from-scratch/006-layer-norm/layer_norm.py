"""
LayerNorm from scratch.

LayerNorm normalizes over the feature dimension (last axis), computing mean and
variance for each individual sample — not across a batch. This makes it
independent of batch size and ideal for sequential models where each token
needs its own normalization.

Formula: y = (x - mean) / sqrt(var + eps) * gamma + beta
  - gamma, beta are learnable scale and shift (initialized to 1 and 0)
  - eps prevents division by zero (usually 1e-5)
"""

import torch
import torch.nn as nn


class LayerNorm(nn.Module):
    """
    LayerNorm: normalize each feature vector independently.

    Unlike BatchNorm (which normalizes across a batch), LayerNorm computes
    mean and variance over the feature dimension only. This means:
    - No batch dimension dependency — works for any batch size
    - Each sample is normalized independently
    - Common in Transformers (BERT, GPT, etc.)
    """

    def __init__(self, d_model, eps=1e-5):
        super().__init__()
        # gamma (scale) starts at 1, beta (shift) starts at 0
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))
        self.eps = eps

    def forward(self, x):
        # x shape: (batch, ..., features) — feature dim is always last
        mean = x.mean(dim=-1, keepdim=True)      # mean over feature axis
        var = x.var(dim=-1, keepdim=True, unbiased=False)  # variance over features
        x_norm = (x - mean) / torch.sqrt(var + self.eps)   # normalize
        return self.gamma * x_norm + self.beta             # scale and shift


if __name__ == "__main__":
    # Example: 4 samples, each with 8 features
    batch, d_model = 4, 8
    x = torch.randn(batch, d_model)

    ln = LayerNorm(d_model)
    y = ln(x)

    print(f"Input shape:  {x.shape}")
    print(f"Output shape: {y.shape}")
    print(f"Output mean per sample (should be ~0): {y.mean(dim=-1)}")
    print(f"Output std  per sample (should be ~1): {y.std(dim=-1)}")
