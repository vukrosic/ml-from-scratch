"""
SGD with Momentum from Scratch

Momentum accumulates a velocity term that carries updates forward
in consistent directions, dampening oscillations in others.
"""

import torch


def sgd_momentum(params, grads, velocities, lr=1e-3, momentum=0.9):
    """
    SGD update with momentum.

    Args:
        params: List of parameter tensors
        grads: List of gradient tensors (same shape as params)
        velocities: List of velocity tensors (must be initialized to zeros)
        lr: Learning rate
        momentum: Momentum coefficient (typically 0.9)
    """
    for p, g, v in zip(params, grads, velocities):
        v.data = momentum * v + g          # accumulate velocity
        p.data = p - lr * v                # update with velocity


def demo():
    """Train a single weight toward a target using SGD with momentum."""
    target = 5.0
    w = torch.tensor([0.0], requires_grad=True)
    optimizer = torch.optim.SGD([w], lr=0.1, momentum=0.9)

    # Track velocity explicitly for visualization
    velocity = torch.zeros_like(w)

    losses = []
    for step in range(100):
        loss = (w - target) ** 2
        losses.append(loss.item())

        loss.backward()

        # Manual momentum update (equivalent to optimizer.step())
        with torch.no_grad():
            velocity = 0.9 * velocity + w.grad
            w.data -= 0.1 * velocity

        w.retain_grad()
        print(f"Step {step:3d}: w = {w.item():.4f}, loss = {loss.item():.4f}, velocity = {velocity.item():.4f}")

        optimizer.zero_grad()

    return losses


if __name__ == "__main__":
    demo()
