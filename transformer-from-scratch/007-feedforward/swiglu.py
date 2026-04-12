"""
SwiGLU activation — used in Llama, PaLM, and many modern LLMs.

SwiGLU(x) = SiLU(x) * sigmoid(x)
          = x * sigmoid(x) * sigmoid(x)

The Swish activation was proposed in "Swish: a Self-Gated Activation Function" (Ramachandran et al., 2017).
The gated variant (SwiGLU) was introduced in the PaLM paper and is now standard in cutting-edge LLMs.

FFN-SwiGLU replaces the standard FFN:
    FFN(x) = Linear2(GELU(Linear1(x)))

With:
    FFN-SwiGLU(x) = Linear3(SiLU(Linear1(x)) * sigmoid(Linear2(x)))

The extra linear projection is the "gate" — it learns what to gate.
"""

import math
import torch
import torch.nn as nn


def silu(x):
    """SiLU (Swish-weighted): x * sigmoid(x)."""
    return x * torch.sigmoid(x)


def swiglu(x):
    """
    SwiGLU: SiLU(x) * sigmoid(x) — the gating mechanism.

    This is the activation used in Llama, PaLM, and many modern architectures.
    It has two learnable parameters (the W and V linear layers in FFN-SwiGLU)
    that control what information flows through the gate.
    """
    return silu(x) * torch.sigmoid(x)


class FFNSwiGLU(nn.Module):
    """
    FeedForward with SwiGLU activation.

    FFN-SwiGLU(x) = Linear3(silu(Linear1(x)) * sigmoid(Linear2(x)))

    Compared to standard FFN (Linear -> GELU -> Linear):
    - Uses SiLU * sigmoid gate instead of just GELU
    - Has an extra linear projection (the gate V in the paper notation)
    - Typically provides better performance at similar compute cost

    Used in: Llama 1/2/3, PaLM, Mistral, Gemma, and most modern LLMs.
    """

    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        # Gate projection (V in paper notation)
        self.w_gate = nn.Linear(d_model, d_ff)
        # Up projection (W in paper notation)
        self.w_up = nn.Linear(d_model, d_ff)
        # Down projection (output)
        self.w_down = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # w_up(x): main activation path
        # w_gate(x): gate signal
        up = silu(self.w_up(x))
        gate = torch.sigmoid(self.w_gate(x))
        h = up * gate
        h = self.dropout(h)
        return self.w_down(h)


class FFNReGLU(nn.Module):
    """
    FeedForward with ReGLU activation (from "GLU Variants Improve Transformer", 2020).

    ReGLU(x) = max(0, Linear1(x)) * Linear2(x)

    This was one of the first gated linear unit variants shown to improve transformers.
    The paper showed that GLU variants consistently outperform plain ReLU.
    """

    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.w_gate = nn.Linear(d_model, d_ff)
        self.w_up = nn.Linear(d_model, d_ff)
        self.w_down = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        gate = torch.relu(self.w_gate(x))
        up = self.w_up(x)
        h = gate * up
        h = self.dropout(h)
        return self.w_down(h)


class FFNGeGLU(nn.Module):
    """
    FeedForward with GeGLU activation — GELU gating instead of ReLU.

    GeGLU(x) = GELU(Linear1(x)) * Linear2(x)

    From "GLU Variants Improve Transformer" (Noam Shazeer, 2020).
    """

    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.w_gate = nn.Linear(d_model, d_ff)
        self.w_up = nn.Linear(d_model, d_ff)
        self.w_down = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        gate = nn.functional.gelu(self.w_gate(x))
        up = self.w_up(x)
        h = gate * up
        h = self.dropout(h)
        return self.w_down(h)


if __name__ == "__main__":
    d_model = 256
    d_ff = 4 * d_model

    batch, seq_len = 2, 16
    x = torch.randn(batch, seq_len, d_model)

    # Instantiate FFN variants
    ffn_swiglu = FFNSwiGLU(d_model, d_ff)
    ffn_reglu = FFNReGLU(d_model, d_ff)
    ffn_geglu = FFNGeGLU(d_model, d_ff)

    print("FFN-SwiGLU output shape:", ffn_swiglu(x).shape)
    print("FFN-ReGLU output shape:", ffn_reglu(x).shape)
    print("FFN-GeGLU output shape:", ffn_geglu(x).shape)

    print("\nSwiGLU activation properties:")
    print(f"  swiglu(1.0)  = {swiglu(torch.tensor(1.0)).item():.4f}")
    print(f"  swiglu(-1.0) = {swiglu(torch.tensor(-1.0)).item():.4f}")
    print(f"  swiglu(0.0)  = {swiglu(torch.tensor(0.0)).item():.4f}")

    # Note: FFNSwiGLU has 3 Linear layers vs standard FFN's 2
    from feedforward import FeedForward
    print(f"\nParameter count comparison:")
    print(f"  Standard FFN: {sum(p.numel() for p in FeedForward(d_model, d_ff).parameters()):,}")
    print(f"  FFN-SwiGLU:   {sum(p.numel() for p in FFNSwiGLU(d_model, d_ff).parameters()):,}")
    print(f"  (SwiGLU has 3 linear layers, hence ~50% more parameters)")
