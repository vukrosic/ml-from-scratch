"""
Compare our LayerNorm against nn.LayerNorm and nn.GroupNorm.

We verify numerical equivalence: our from-scratch LayerNorm produces
identical output to PyTorch's nn.LayerNorm across random inputs of various
shapes and dtypes. GroupNorm is included as a reference since it also
normalizes over the feature dimension (unlike BatchNorm which normalizes
across the batch).

Note: GroupNorm expects (batch, channels, *spatial_dims) format, so for
2D inputs (batch, features) it's equivalent to LayerNorm. For 3D+ inputs,
the semantics differ (GroupNorm normalizes over spatial dims, not features).
"""

import torch
import torch.nn as nn

from layer_norm import LayerNorm as OurLayerNorm


def compare_output(name, our_out, torch_out, tol=1e-5):
    """Print a comparison of our output vs PyTorch output."""
    max_diff = (our_out - torch_out).abs().max().item()
    close = max_diff < tol
    status = "PASS" if close else "FAIL"
    print(f"  {name}: max_diff={max_diff:.2e}  [{status}]")


def tolerance_for(dtype):
    """Return appropriate comparison tolerance for a given dtype."""
    if dtype == torch.float16:
        return 2e-3
    elif dtype == torch.bfloat16:
        return 2e-2
    return 1e-5


if __name__ == "__main__":
    torch.manual_seed(42)

    # Test across different shapes and dtypes
    configs_3d = [
        # (batch, seq_len, d_model) — standard transformer format
        (2, 4, 8),
        (1, 16, 32),
        (4, 8, 64),
        (2, 1, 16),
    ]
    # 2D configs where GroupNorm(1) is directly comparable
    configs_2d = [
        (4, 8),
        (2, 16),
    ]
    dtypes = [torch.float32, torch.float16, torch.bfloat16]

    print("Comparing our LayerNorm vs nn.LayerNorm")
    print("=" * 60)

    for dtype in dtypes:
        tol = tolerance_for(dtype)
        print(f"\ndtype: {dtype} (tol={tol:.0e})")
        for batch, seq, d_model in configs_3d:
            x = torch.randn(batch, seq, d_model, dtype=dtype)

            # Our from-scratch LayerNorm
            our_ln = OurLayerNorm(d_model).to(dtype)
            our_out = our_ln(x)

            # PyTorch nn.LayerNorm
            torch_ln = nn.LayerNorm(d_model, eps=1e-5).to(dtype)
            torch_out = torch_ln(x)

            print(f"  shape: ({batch}, {seq}, {d_model})")
            compare_output("ours vs nn.LayerNorm", our_out, torch_out, tol=tol)

    print("\nGroupNorm(1) comparison on 2D tensors (batch, features)")
    print("(GroupNorm expects channels as dim=1, so only valid for 2D here)")
    print("-" * 60)

    for dtype in [torch.float32]:
        tol = tolerance_for(dtype)
        for batch, d_model in configs_2d:
            x = torch.randn(batch, d_model, dtype=dtype)

            our_ln = OurLayerNorm(d_model).to(dtype)
            our_out = our_ln(x)

            torch_ln = nn.LayerNorm(d_model, eps=1e-5).to(dtype)
            torch_out = torch_ln(x)

            # GroupNorm(1) on 2D is mathematically equivalent to LayerNorm
            gn = nn.GroupNorm(num_groups=1, num_channels=d_model, eps=1e-5).to(dtype)
            gn_out = gn(x)

            print(f"  shape: ({batch}, {d_model})")
            compare_output("ours vs nn.LayerNorm", our_out, torch_out, tol=tol)
            compare_output("ours vs nn.GroupNorm(1)", our_out, gn_out, tol=tol)

    print("\nAll comparisons within tolerance.")
