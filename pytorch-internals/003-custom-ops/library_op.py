"""
Registering custom ops with torch.library for torch.compile compatibility.

torch.compile traces through the autograd graph. By default, custom
autograd.Function nodes are traced as opaque black boxes. Registering
with torch.library makes the op transparent to compile, enabling:
- Graph fusion (your op fused with adjacent ops)
- Better kernel selection
- No graph breaks

Without registration, a custom op appears as a "fallback" op and
compile treats it as a graph break.
"""

import torch
from torch.library import Library


class ReLUFunction(torch.autograd.Function):
    """Custom ReLU with the same backward as forward mask."""

    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return x.clamp(min=0)

    @staticmethod
    def backward(ctx, grad_output):
        x, = ctx.saved_tensors
        return grad_output * (x > 0)


# --- torch.library registration ---
#
# torch.library works in two steps:
#   1. lib.define()  — declare the op schema (name, inputs, outputs)
#   2. lib.impl()     — provide the implementation for a specific "role"
#
# "Autograd" role means: use this function as the autograd kernel.
# PyTorch will internally call it from the autograd engine.

lib = Library("custom_ops", "DEF")

# Step 1: declare the op. "() -> ()" is a no-op schema; adjust for real ops.
lib.define("relu(Tensor x) -> Tensor")

# Step 2: register our autograd.Function as the Autograd role.
# Pass the autograd function directly.
lib.impl("relu", ReLUFunction.apply, "Autograd")


def relu(x):
    """Registered custom op — torch.compile can trace through this."""
    return torch.ops.custom_ops.relu(x)


# --- Unregistered autograd.Function (for comparison) ---

def custom_relu_unregistered(x):
    return ReLUFunction.apply(x)


# --- torch.compile test ---

if __name__ == "__main__":
    x = torch.randn(4, requires_grad=True)

    # Registered op — compile can trace through it.
    compiled_relu = torch.compile(relu)
    out = compiled_relu(x)
    print("Registered op compiled successfully:", out)

    # Unregistered Function — compile treats it as a graph break.
    compiled_custom = torch.compile(custom_relu_unregistered)
    out2 = compiled_custom(x)
    print("Unregistered op also works (but as a graph break):", out2)

    # Verify they produce the same result.
    x2 = torch.randn(4, requires_grad=True)
    print("\nRegistered:   ", relu(x2))
    print("Unregistered: ", custom_relu_unregistered(x2))
