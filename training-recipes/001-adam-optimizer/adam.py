"""
Adam optimizer from scratch — implemented directly from the paper.

Adam: Adaptive Moment Estimation
Kingma & Ba, 2014 — https://arxiv.org/abs/1412.6980

The update rule is:
    update = lr * m_hat / (sqrt(v_hat) + eps)

where m_hat and v_hat are bias-corrected first and second moment estimates.
"""

import torch
from collections.abc import Iterable


class Adam:
    """
    Adam optimizer.

    Parameters
    ----------
    params : iterable of tensors
        Parameters to optimize.
    lr : float, default 1e-3
        Learning rate.
    beta1 : float, default 0.9
        Decay rate for first moment (momentum).
    beta2 : float, default 0.999
        Decay rate for second moment (RMSProp).
    eps : float, default 1e-8
        Numerical stability constant.
    """

    def __init__(
        self,
        params: Iterable,
        lr: float = 1e-3,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
    ):
        self.params = list(params)
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps

        # State: first moment (m), second moment (v), and step counter (t)
        self.state = {
            id(p): {"m": torch.zeros_like(p), "v": torch.zeros_like(p), "t": 0}
            for p in self.params
        }

    def step(self) -> None:
        """
        Perform one optimization step.

        From the paper (Algorithm 1):
            m_t = beta1 * m_{t-1} + (1 - beta1) * g_t          (1)
            v_t = beta2 * v_{t-1} + (1 - beta2) * g_t^2        (2)
            m_hat = m_t / (1 - beta1^t)                         (3)  <- bias correction
            v_hat = v_t / (1 - beta2^t)                         (4)  <- bias correction
            theta_{t} = theta_{t-1} - lr * m_hat / (sqrt(v_hat) + eps)  (5)
        """
        for p in self.params:
            grad = p.grad
            if grad is None:
                continue

            s = self.state[id(p)]
            s["t"] += 1
            t = s["t"]

            # (1) Update biased first moment estimate
            s["m"].mul_(self.beta1).add_(grad, alpha=1 - self.beta1)

            # (2) Update biased second raw moment estimate
            s["v"].mul_(self.beta2).addcmul_(grad, grad, value=1 - self.beta2)

            # (3) & (4) Bias correction
            m_hat = s["m"] / (1 - self.beta1 ** t)
            v_hat = s["v"] / (1 - self.beta2 ** t)

            # (5) Update parameters
            p.data.addcdiv_(m_hat, v_hat.sqrt().add_(self.eps), value=-self.lr)

    def zero_grad(self) -> None:
        for p in self.params:
            if p.grad is not None:
                p.grad.zero_()


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    torch.manual_seed(42)

    # Simple quadratic: f(x, y) = x^2 + 10*y^2
    # Minimum at (0, 0)
    x = torch.tensor([3.0, -4.0], requires_grad=True)

    optimizer = Adam([x], lr=0.1)

    print("Solving min f(x,y) = x^2 + 10*y^2 with custom Adam")
    print(f"Initial: x={x.data.tolist()}")

    for step in range(1, 51):
        optimizer.zero_grad()
        loss = x[0] ** 2 + 10 * x[1] ** 2
        loss.backward()
        optimizer.step()

        if step % 10 == 0:
            print(f"  step {step:3d} | loss={loss.item():.6f} | x={x.data.tolist()}")

    print("Done.")
