"""
AdamW (Adam with Decoupled Weight Decay) from Scratch

AdamW decouples weight decay from gradient-based updates.
In Adam, L2 regularization gets scaled by the adaptive learning rate,
which means weight decay is stronger in low-gradient directions.
AdamW applies weight decay directly to the parameters, independent
of the gradient scaling.

The key difference from Adam:
    Adam:   p = p - lr * m_hat / (sqrt(v_hat) + eps)
    AdamW:  p = p - lr * m_hat / (sqrt(v_hat) + eps)
            p = p - lr * weight_decay * p    # decoupled step
"""

import torch


def adamw(params, grads, m, v, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=0.01):
    """
    AdamW update with decoupled weight decay.

    Args:
        params: List of parameter tensors
        grads: List of gradient tensors
        m: List of first moment tensors (initialized to zeros)
        v: List of second moment tensors (initialized to zeros)
        lr: Learning rate
        beta1: Exponential decay rate for first moment (default 0.9)
        beta2: Exponential decay rate for second moment (default 0.999)
        eps: Small constant to prevent division by zero (default 1e-8)
        weight_decay: Decay coefficient (default 0.01)
    """
    for t, (p, g) in enumerate(zip(params, grads), 1):
        m.data = beta1 * m + (1 - beta1) * g
        v.data = beta2 * v + (1 - beta2) * g * g

        # Bias correction
        m_hat = m / (1 - beta1 ** t)
        v_hat = v / (1 - beta2 ** t)

        # Adam update
        p.data = p - lr * m_hat / (v_hat.sqrt() + eps)

        # Decoupled weight decay: applied directly to parameters
        p.data = p - lr * weight_decay * p


def demo():
    """Train a single weight toward a target using AdamW."""
    target = 5.0
    w = torch.tensor([0.0], requires_grad=True)

    # AdamW state
    m = torch.zeros_like(w)
    v = torch.zeros_like(w)

    weight_decay = 0.01
    lr = 0.1

    losses = []
    for step in range(100):
        loss = (w - target) ** 2
        losses.append(loss.item())

        loss.backward()
        g = w.grad

        # AdamW update
        m = 0.9 * m + 0.1 * g
        v = 0.999 * v + 0.001 * g * g
        m_hat = m / (1 - 0.9 ** (step + 1))
        v_hat = v / (1 - 0.999 ** (step + 1))

        w.data = w - lr * m_hat / (v_hat.sqrt() + 1e-8)
        w.data = w - lr * weight_decay * w  # decoupled weight decay

        print(f"Step {step:3d}: w = {w.item():.4f}, loss = {loss.item():.4f}")

        w.grad.zero_()

    return losses


if __name__ == "__main__":
    demo()
