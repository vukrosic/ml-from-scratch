"""
Custom ReLU with torch.autograd.Function.

torch.autograd.Function lets you define a custom forward/backward pair.
PyTorch's autograd engine uses the backward() to compute gradients automatically.

Use this when you need fine-grained control over the gradient computation,
e.g. gradient amplification, gradient clipping, or custom routing.
"""

import torch


class ReLUFunction(torch.autograd.Function):
    """
    Custom ReLU implemented as an autograd Function.

    forward(x): applies ReLU element-wise.
    backward(grad_output): returns the gradient propagated to the input.

    ctx.save_for_backward stores tensors needed in backward().
    """

    @staticmethod
    def forward(ctx, x):
        # Store the input for use in backward — needed to compute mask.
        ctx.save_for_backward(x)
        return x.clamp(min=0)

    @staticmethod
    def backward(ctx, grad_output):
        # Retrieve stored input and compute ReLU mask (1 where x > 0).
        x, = ctx.saved_tensors
        mask = x > 0
        # Amplify gradients where x > 0 by a factor of 2.
        # This is the "useful example" — you can modify gradients arbitrarily.
        amplified = grad_output * mask * 2.0
        return amplified


def custom_relu(x):
    """
    Apply the custom ReLU with gradient amplification.

    During forward: standard ReLU (zero out negatives).
    During backward: gradients for positive inputs are doubled.
    """
    return ReLUFunction.apply(x)


# --- Demo: gradient amplification in action ---

if __name__ == "__main__":
    x = torch.tensor([1.0, -2.0, 3.0, -4.0], requires_grad=True)

    # Custom ReLU — gradients for positive elements get amplified 2x.
    out = custom_relu(x)
    out.sum().backward()

    print("Input:              ", x.data)
    print("ReLU output:        ", out)
    print("Gradient (amplified):", x.grad)

    # Compare with standard ReLU — gradients are NOT amplified.
    x2 = torch.tensor([1.0, -2.0, 3.0, -4.0], requires_grad=True)
    torch.relu(x2).sum().backward()
    print("\nStandard ReLU gradient:", x2.grad)
