"""
adaptive_clip.py — Adaptive per-layer gradient clipping.

Instead of clipping all gradients by a single global norm, adaptive clipping
computes a clipping threshold per-layer based on running statistics (EMA) of
each layer's gradient magnitude. Layers with unusually large gradients get
clipped harder.

This is inspired by "Adaptive Gradient Clipping" (DadaptAdam, 2023) and
related techniques used in training large models.

The key idea: instead of a fixed threshold, use a moving average of past
gradient norms to set a data-driven threshold per layer.
"""

import torch
from collections.abc import Iterable


class AdaptiveClipper:
    """
    Per-layer adaptive gradient clipping based on running statistics.

    Each layer gets its own clipping threshold computed from an exponential
    moving average (EMA) of its historical gradient norms.

    Parameters
    ----------
    params : iterable of tensors
        Parameters to clip.
    clip_factor : float, default 2.0
        Multiplicative factor for the EMA-based threshold.
        Clips when layer_norm > clip_factor * ema_norm.
    ema_decay : float, default 0.99
        Decay rate for the EMA of gradient norms.
    norm_type : float, default 2.0
        The p-norm to use for gradient computation.
    """

    def __init__(
        self,
        params: Iterable,
        clip_factor: float = 2.0,
        ema_decay: float = 0.99,
        norm_type: float = 2.0,
    ):
        self.params = list(params)
        self.clip_factor = clip_factor
        self.ema_decay = ema_decay
        self.norm_type = norm_type

        # Per-layer EMA state: stores the EMA of gradient norms for each param
        self.ema_norms = {
            id(p): None for p in self.params
        }

    def clip(self) -> None:
        """
        Compute per-layer gradient norms, update their EMAs, and clip
        any layer whose current norm exceeds clip_factor * ema_norm.
        """
        for p in self.params:
            if p.grad is None:
                continue

            pid = id(p)
            grad_norm = p.grad.norm(self.norm_type)

            # Initialize EMA on first step
            if self.ema_norms[pid] is None:
                self.ema_norms[pid] = grad_norm.detach()
            else:
                # Update EMA: ema = decay * ema + (1 - decay) * new
                self.ema_norms[pid] = (
                    self.ema_decay * self.ema_norms[pid]
                    + (1 - self.ema_decay) * grad_norm.detach()
                )

            # Compute adaptive threshold
            threshold = self.clip_factor * self.ema_norms[pid]

            # Clip if norm exceeds threshold
            if grad_norm > threshold:
                clip_coef = threshold / (grad_norm + 1e-8)
                p.grad.data.mul_(clip_coef)

    def get_ema_norms(self):
        """Return the current EMA norms for inspection."""
        return {
            name: self.ema_norms[pid].item()
            for pid, (name, _) in enumerate(zip(self.params, range(len(self.params))))
        }


class StableAdamW(torch.optim.Optimizer):
    """
    AdamW with adaptive gradient clipping, based on the Decoupled Adam (DadaptAdam)
    approach. This combines weight decay with adaptive per-layer clipping.

    Reference: https://arxiv.org/abs/2309.05027
    """

    def __init__(self, params, lr=1e-3, weight_decay=0.1, beta1=0.9, beta2=0.999,
                 eps=1e-8, clip_factor=2.0, ema_decay=0.99):
        defaults = dict(lr=lr, weight_decay=weight_decay, beta1=beta1, beta2=beta2,
                       eps=eps, clip_factor=clip_factor, ema_decay=ema_decay)
        super().__init__(params, defaults)

    def __init__(self, params, lr=1e-3, weight_decay=0.1, beta1=0.9, beta2=0.999,
                 eps=1e-8, clip_factor=2.0, ema_decay=0.99):
        super().__init__(params, locals())

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            wd = group["weight_decay"]
            beta1 = group["beta1"]
            beta2 = group["beta2"]
            eps = group["eps"]
            clip_factor = group["clip_factor"]
            ema_decay = group["ema_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad
                state = self.state[p]

                # Initialize state
                if len(state) == 0:
                    state["exp_avg"] = torch.zeros_like(p)
                    state["exp_avg_sq"] = torch.zeros_like(p)
                    state["ema_grad_norm"] = None
                    state["t"] = 0

                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                t = state["t"]
                t += 1

                # Adaptive clipping
                grad_norm = grad.norm(2)
                if state["ema_grad_norm"] is None:
                    state["ema_grad_norm"] = grad_norm.detach()
                else:
                    state["ema_grad_norm"] = (
                        ema_decay * state["ema_grad_norm"]
                        + (1 - ema_decay) * grad_norm.detach()
                    )

                clip_thresh = clip_factor * state["ema_grad_norm"]
                if grad_norm > clip_thresh:
                    grad = grad * (clip_thresh / (grad_norm + eps))

                # Adam update
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                # Bias correction
                bias_correction1 = 1 - beta1 ** t
                bias_correction2 = 1 - beta2 ** t

                m_hat = exp_avg / bias_correction1
                v_hat = exp_avg_sq / bias_correction2

                # Decoupled weight decay
                p.data.mul_(1 - lr * wd)

                # Adam step
                p.data.addcdiv_(m_hat, v_hat.sqrt().add_(eps), value=-lr)

        return loss


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    torch.manual_seed(42)

    # Create a simple model
    model = torch.nn.Sequential(
        torch.nn.Linear(10, 32),
        torch.nn.ReLU(),
        torch.nn.Linear(32, 32),
        torch.nn.ReLU(),
        torch.nn.Linear(32, 2),
    )

    clipper = AdaptiveClipper(model.parameters(), clip_factor=2.0, ema_decay=0.99)

    print("Adaptive per-layer gradient clipping")
    print(f"Clip factor: 2.0, EMA decay: 0.99\n")

    for step in range(1, 21):
        # Forward pass with random data
        x = torch.randn(8, 10)
        y = torch.randn(8, 2)

        model.zero_grad()
        pred = model(x)
        loss = nn.functional.mse_loss(pred, y)
        loss.backward()

        # Show per-layer norms before clipping
        pre_clip = {
            name: p.grad.norm().item()
            for name, p in model.named_parameters() if p.grad is not None
        }

        clipper.clip()

        post_clip = {
            name: p.grad.norm().item()
            for name, p in model.named_parameters() if p.grad is not None
        }

        if step % 5 == 0:
            print(f"Step {step:2d}:")
            for name in pre_clip:
                clipped = " <- CLIPPED" if post_clip[name] < pre_clip[name] * 0.99 else ""
                print(f"  {name:20s}: {pre_clip[name]:.4e} -> {post_clip[name]:.4e}{clipped}")

        # Simple training step
        with torch.no_grad():
            for p in model.parameters():
                if p.grad is not None:
                    p -= p.grad * 0.01

    print("\nAdaptive clipping adjusts per-layer thresholds based on history.")