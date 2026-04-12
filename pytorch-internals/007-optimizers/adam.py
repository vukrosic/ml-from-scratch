"""
Adam Optimizer from Scratch

Adam maintains two running averages:
- First moment (m): direction of the gradient (like momentum)
- Second moment (v): scale of the gradient (like RMSProp)

The update divides the learning rate by the RMS of past gradients,
providing adaptive per-parameter learning rates.

Bias correction is applied because both moments are initialized to zero,
causing them to be biased toward zero early in training.
"""

import torch


def adam(params, grads, m, v, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8):
    """
    Adam update with bias correction.

    Args:
        params: List of parameter tensors
        grads: List of gradient tensors
        m: List of first moment tensors (initialized to zeros)
        v: List of second moment tensors (initialized to zeros)
        lr: Learning rate
        beta1: Exponential decay rate for first moment (default 0.9)
        beta2: Exponential decay rate for second moment (default 0.999)
        eps: Small constant to prevent division by zero (default 1e-8)
    """
    for t, (p, g) in enumerate(zip(params, grads), 1):
        m.data = beta1 * m + (1 - beta1) * g
        v.data = beta2 * v + (1 - beta2) * g * g

        # Bias correction: compensates for initialization bias
        m_hat = m / (1 - beta1 ** t)
        v_hat = v / (1 - beta2 ** t)

        p.data = p - lr * m_hat / (v_hat.sqrt() + eps)


def demo():
    """Train a single weight toward a target using Adam."""
    target = 5.0
    w = torch.tensor([0.0], requires_grad=True)

    # Adam state: first and second moments
    m = torch.zeros_like(w)
    v = torch.zeros_like(w)

    losses = []
    for step in range(100):
        loss = (w - target) ** 2
        losses.append(loss.item())

        loss.backward()
        g = w.grad

        # Adam update with bias correction
        m = 0.9 * m + 0.1 * g
        v = 0.999 * v + 0.001 * g * g
        m_hat = m / (1 - 0.9 ** (step + 1))
        v_hat = v / (1 - 0.999 ** (step + 1))

        w.data = w - 0.1 * m_hat / (v_hat.sqrt() + 1e-8)

        print(f"Step {step:3d}: w = {w.item():.4f}, loss = {loss.item():.4f}, "
              f"m_hat = {m_hat.item():.4f}, v_hat = {v_hat.item():.4f}")

        w.grad.zero_()

    return losses


if __name__ == "__main__":
    demo()
