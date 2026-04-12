"""
RMSNorm from scratch.

RMSNorm (Root Mean Square Normalization) is a stripped-down LayerNorm that
removes the mean-centring step. Instead of computing both mean and variance,
it only computes the root mean square (RMS) of the feature vector:

    y = x / RMS(x) * gamma + beta,  where RMS(x) = sqrt(mean(x^2))

This saves one reduction operation (no mean), which adds up when you have
dozens of RMSNorm layers in a large model. Llama, Mistral, and many modern
LLMs use RMSNorm instead of LayerNorm for this reason.
"""

import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    """
    RMSNorm: normalize by RMS only, no mean centering.

    Key insight: LayerNorm subtracts the mean primarily to make the output
    zero-centred. But this zero-centring doesn't meaningfully improve
    performance in most cases — what matters is the scaling (variance).
    By dropping the mean subtraction, RMSNorm saves one reduction pass and
    is slightly cheaper to compute.
    """

    def __init__(self, d_model, eps=1e-5):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.eps = eps

    def forward(self, x):
        # RMS = sqrt(mean(x^2)) over feature dimension
        rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        x_norm = x / rms
        return self.gamma * x_norm


if __name__ == "__main__":
    batch, d_model = 4, 8
    x = torch.randn(batch, d_model)

    rms = RMSNorm(d_model)
    y = rms(x)

    print(f"Input:  {x}")
    print(f"Output: {y}")
    print(f"Output RMS per sample (should be ~1): {y.pow(2).mean(dim=-1).sqrt()}")
    print(f"Note: output mean is NOT forced to 0 (that's RMSNorm, not LayerNorm)")
