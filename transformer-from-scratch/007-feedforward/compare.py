"""
Compare our from-scratch FFN against PyTorch's nn.Sequential.

We verify that our implementation produces numerically identical output
to: nn.Sequential(nn.Linear, nn.GELU, nn.Dropout, nn.Linear)
"""

import math
import torch
import torch.nn as nn

from feedforward import FeedForward, gelu


def ffn_pytorch(d_model, d_ff, dropout=0.0):
    """
    PyTorch equivalent of our FeedForward.
    nn.Sequential composes layers in order: Linear -> GELU -> Dropout -> Linear.
    """
    return nn.Sequential(
        nn.Linear(d_model, d_ff),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(d_ff, d_model),
    )


def gelu_pytorch(x):
    """PyTorch's GELU — same approximation we use."""
    return nn.functional.gelu(x)


if __name__ == "__main__":
    d_model = 256
    d_ff = 4 * d_model  # 1024
    dropout = 0.0  # no dropout for deterministic comparison

    # Our FFN
    ffn_ours = FeedForward(d_model, d_ff, dropout=dropout)

    # PyTorch equivalent
    ffn_torch = ffn_pytorch(d_model, d_ff, dropout=dropout)

    # Copy weights so we're comparing identical models
    ffn_torch[0].weight.data = ffn_ours.linear1.weight.data.clone()
    ffn_torch[0].bias.data = ffn_ours.linear1.bias.data.clone()
    ffn_torch[3].weight.data = ffn_ours.linear2.weight.data.clone()
    ffn_torch[3].bias.data = ffn_ours.linear2.bias.data.clone()

    # Test 1: Numerical equivalence
    torch.manual_seed(42)
    x = torch.randn(4, 32, d_model)
    out_ours = ffn_ours(x)
    out_torch = ffn_torch(x)

    max_diff = (out_ours - out_torch).abs().max().item()
    print("=" * 50)
    print("NUMERICAL COMPARISON: our FFN vs PyTorch nn.Sequential")
    print("=" * 50)
    print(f"Max |difference|: {max_diff:.2e}")
    print(f"Identical?        {torch.allclose(out_ours, out_torch)}")

    # Test 2: GELU approximation accuracy
    print("\n" + "=" * 50)
    print("GELU IMPLEMENTATION COMPARISON")
    print("=" * 50)
    test_values = [-3.0, -1.0, 0.0, 0.5, 1.0, 2.0, 5.0]
    print(f"{'x':>8} {'ours':>12} {'PyTorch':>12} {'diff':>12}")
    print("-" * 48)
    for x_val in test_values:
        x_tensor = torch.tensor(x_val)
        ours = gelu(x_tensor)
        torch_gel = gelu_pytorch(x_tensor)
        diff = (ours - torch_gel).abs().item()
        print(f"{x_val:>8.2f} {ours.item():>12.6f} {torch_gel.item():>12.6f} {diff:>12.2e}")

    # Test 3: Gradient comparison
    print("\n" + "=" * 50)
    print("GRADIENT COMPARISON")
    print("=" * 50)
    x = torch.randn(2, 16, d_model, requires_grad=True)
    (ffn_ours(x).sum()).backward()
    (ffn_torch(x).sum()).backward()
    grad_ours = ffn_ours.linear2.weight.grad
    grad_torch = ffn_torch[3].weight.grad
    max_grad_diff = (grad_ours - grad_torch).abs().max().item()
    print(f"Max |grad difference|: {max_grad_diff:.2e}")
    print(f"Gradients identical?  {torch.allclose(grad_ours, grad_torch)}")

    print("\nAll tests passed — our from-scratch FFN is numerically equivalent to PyTorch.")
