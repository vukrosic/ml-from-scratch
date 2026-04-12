"""
Warmup LayerNorm: tracks running mean and std over the first N steps.

In the first N training steps, we can accumulate statistics over a warmup
window to get a more stable initial normalization. After the warmup period,
the layer switches to using the accumulated statistics (like a running
BatchNorm would). This is mostly for experimentation — standard LayerNorm
does not use warmup tracking, but understanding it helps clarify the
difference between LayerNorm (sample-level stats) and BatchNorm (running stats).

This implementation allows you to:
1. Track running mean/std during a warmup window
2. Freeze (lock) the statistics after warmup
3. Compare locked vs per-sample normalization
"""

import torch
import torch.nn as nn


class WarmupLayerNorm(nn.Module):
    """
    LayerNorm with optional warmup statistics tracking.

    During warmup (steps 0 to warmup_steps-1), the layer accumulates running
    mean and variance. After warmup, it locks to the accumulated statistics
    and ignores the input's own statistics (frozen normalization).

    Use case: Some architectures benefit from stable normalization stats
    in early training before the model has seen enough data to compute
    good per-sample statistics.
    """

    def __init__(self, d_model, warmup_steps=100, eps=1e-5):
        super().__init__()
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        self.eps = eps
        self.step = 0

        # Learnable scale and shift
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))

        # Running statistics (accumulated during warmup)
        self.register_buffer('running_mean', torch.zeros(d_model))
        self.register_buffer('running_var', torch.ones(d_model))

    def forward(self, x, lock_stats=False):
        """
        x: (batch, ..., features)
        lock_stats: if True, always use running stats (frozen after warmup)
        """
        if lock_stats or self.step >= self.warmup_steps:
            # Use running (frozen) statistics
            # running_mean/var have shape (d_model,) → unsqueeze for broadcasting
            mean = self.running_mean.unsqueeze(0)   # (1, d_model)
            var = self.running_var.unsqueeze(0)     # (1, d_model)
        else:
            # Compute statistics from current input
            mean = x.mean(dim=-1, keepdim=True)      # (batch, 1)
            var = x.var(dim=-1, keepdim=True, unbiased=False)  # (batch, 1)

            # Accumulate running statistics during warmup
            with torch.no_grad():
                # Average per-feature means across the batch, then update running
                alpha = 0.1  # momentum for running stats
                batch_mean = mean.mean(dim=0).squeeze(-1)  # mean over batch → (d_model,)
                batch_var = var.mean(dim=0).squeeze(-1)     # mean over batch → (d_model,)
                self.running_mean = alpha * batch_mean + (1 - alpha) * self.running_mean
                self.running_var = alpha * batch_var + (1 - alpha) * self.running_var

        self.step += 1
        x_norm = (x - mean) / torch.sqrt(var + self.eps)
        return self.gamma * x_norm + self.beta

    def reset(self):
        """Reset running statistics and step counter."""
        self.step = 0
        self.running_mean.zero_()
        self.running_var.fill_(1.0)


if __name__ == "__main__":
    torch.manual_seed(42)
    batch, d_model, steps = 4, 8, 120
    x = torch.randn(steps, batch, d_model)

    ln_warmup = WarmupLayerNorm(d_model, warmup_steps=100)
    ln_standard = nn.LayerNorm(d_model, eps=1e-5)

    print("Step  | Warmup LN (locked stats) | Standard LN (per-sample)")
    print("-" * 70)
    for step in [0, 50, 99, 100, 119]:
        out_warmup = ln_warmup(x[step], lock_stats=(step >= 100))
        out_standard = ln_standard(x[step])

        w_mean = out_warmup.mean().item()
        w_std = out_warmup.std().item()
        s_mean = out_standard.mean().item()
        s_std = out_standard.std().item()

        mode = "locked" if step >= 100 else "per-sample"
        print(f"  {step:3d}  | warmup={w_mean:+.4f} ± {w_std:.4f} [{mode}]"
              f" | standard={s_mean:+.4f} ± {s_std:.4f}")

    print(f"\nFinal running mean norm: {ln_warmup.running_mean.norm().item():.4f}")
    print("After warmup, locked stats give consistent (but not identical) normalization.")
