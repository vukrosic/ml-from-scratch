"""
grad_clip.py — Gradient clipping from scratch.

Gradient clipping stabilizes training by preventing gradients from growing
too large (the "exploding gradients" problem). When the gradient norm exceeds
a threshold, we rescale all gradients so their norm equals the threshold.

The rule (from Pascanu et al. 2012):
    If ||g|| > clip_threshold:
        g = g * (clip_threshold / ||g||)

This is equivalent to clamping the gradient norm to clip_threshold.
"""

import torch
from collections.abc import Iterable


class GradientClipper:
    """
    Clip gradients by global norm.

    Parameters
    ----------
    params : iterable of tensors
        Parameters whose gradients to clip.
    max_norm : float
        Maximum gradient norm. Gradients with norm exceeding this value
        are rescaled so their norm equals max_norm.
    norm_type : float, default 2.0
        The p-norm to compute. Uses the same norm type as torch.
    """

    def __init__(
        self,
        params: Iterable,
        max_norm: float,
        norm_type: float = 2.0,
    ):
        self.params = list(params)
        self.max_norm = max_norm
        self.norm_type = norm_type

    def clip(self) -> None:
        """
        Compute the global gradient norm and rescale gradients in-place
        if it exceeds max_norm.

        The clipping formula:
            clip_coef = max_norm / (total_norm + eps)
            for p in params:
                p.grad.data.mul_(clip_coef)

        where total_norm is the sqrt of sum(p_i.grad^2) for all params.
        """
        # Compute global gradient norm across all parameters
        total_norm = torch.norm(
            torch.stack([torch.norm(p.grad, self.norm_type) for p in self.params]),
            p=self.norm_type,
        )

        # Numerical stability epsilon
        clip_coef = self.max_norm / (total_norm + 1e-6)

        # Only clip if gradient norm exceeds the threshold
        if clip_coef < 1.0:
            for p in self.params:
                p.grad.data.mul_(clip_coef)


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    torch.manual_seed(42)

    # Simulate a gradient explosion scenario: very large gradients
    # that would cause divergence without clipping
    x = torch.tensor([1.0, 2.0], requires_grad=True)

    print("Demonstrating gradient clipping on an exploding gradient")
    print(f"Initial: x={x.data.tolist()}\n")

    clipper = GradientClipper([x], max_norm=1.0)

    for step in range(1, 11):
        # Clear gradients
        if x.grad is not None:
            x.grad.zero_()

        # Loss function that produces growing gradients
        # d/dx [x^8] = 8*x^7, which grows very fast
        loss = x[0] ** 8 + x[1] ** 8
        loss.backward()

        # Show pre-clip gradient norm
        pre_clip_norm = x.grad.norm().item()
        print(f"Step {step:2d}: pre-clip grad_norm={pre_clip_norm:.4e}", end="")

        # Apply clipping
        clipper.clip()

        post_clip_norm = x.grad.norm().item()
        print(f" | post-clip grad_norm={post_clip_norm:.4e}")

        # Manual gradient descent step (lr=0.01)
        with torch.no_grad():
            x -= x.grad * 0.01
        x.grad.zero_()

    print(f"\nFinal: x={x.data.tolist()}")
    print("Gradient clipping prevented explosion and kept training stable.")