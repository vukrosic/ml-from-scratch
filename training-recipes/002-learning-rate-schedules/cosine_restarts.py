"""
cosine_restarts.py — Cosine Annealing with Warm Restarts.

Standard cosine annealing decays the LR to near-zero at the end of training.
With warm restarts (Loshchilov & Hutter 2017), the LR periodically resets to
the initial value and decays again. This escapes local minima and saddle points
that a single annealing run would get stuck in.

The SGDR formula (paper Algorithm 1):
    lr(t) = eta_min + (base_lr - eta_min) * (1 + cos(pi * t_cur / T_i)) / 2

where t_cur is the number of steps since the last restart and T_i is the
period for restart i.

Matches: torch.optim.lr_scheduler.CosineAnnealingWarmRestarts
"""

import math
import matplotlib.pyplot as plt


class CosineAnnealingWarmRestarts:
    """
    Cosine Annealing with Warm Restarts.

    The learning rate resets to base_lr every T_i steps (or epochs) and
    decays along a cosine curve. Each restart period can be multiplied by
    a factor to make subsequent cycles longer.

    Parameters
    ----------
    optimizer : torch.optim.Optimizer
        Wrapped optimizer.
    T_0 : int
        Length of the first restart period (in steps/epochs).
    T_mult : int, default 1
        Multiplier for the period after each restart.
        T_i = T_0 * T_mult ** i
    eta_min : float, default 0.0
        Minimum learning rate.
    """

    def __init__(
        self,
        optimizer,
        T_0: int,
        T_mult: int = 1,
        eta_min: float = 0.0,
    ):
        self.optimizer = optimizer
        self.base_lr = optimizer.param_groups[0]["lr"]
        self.T_0 = T_0
        self.T_mult = T_mult
        self.eta_min = eta_min
        self.T_i = T_0
        self.t_cur = 0
        self.last_epoch = -1

    def get_lr(self):
        """Compute LR at current position within the restart cycle."""
        return [
            self.eta_min
            + (self.base_lr - self.eta_min)
            * (1 + math.cos(math.pi * self.t_cur / self.T_i))
            / 2
            for _ in self.optimizer.param_groups
        ]

    def step(self):
        """
        Advance the scheduler by one step/epoch.

        When t_cur reaches T_i, we restart: reset t_cur=0 and extend
        the period by multiplying by T_mult.
        """
        self.last_epoch += 1
        self.t_cur += 1

        # If we've completed a full cycle, restart
        if self.t_cur >= self.T_i:
            self.t_cur = 0
            self.T_i = int(self.T_i * self.T_mult)

        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group["lr"] = lr


def plot_restarts_curve(T_0=10, T_mult=2, total_steps=80, eta_min=0.0, base_lr=1.0):
    """Plot the LR curve over time for a given restart configuration."""
    import torch

    model = torch.nn.Linear(10, 2)
    optimizer = torch.optim.SGD([{"params": [model.weight], "lr": base_lr}])
    scheduler = CosineAnnealingWarmRestarts(
        optimizer, T_0=T_0, T_mult=T_mult, eta_min=eta_min
    )

    lrs = []
    for _ in range(total_steps):
        scheduler.step()
        lrs.append(optimizer.param_groups[0]["lr"])

    plt.figure(figsize=(10, 4))
    plt.plot(range(total_steps), lrs, linewidth=2, color="#2196F3")
    plt.xlabel("Step / Epoch")
    plt.ylabel("Learning Rate")
    plt.title(
        f"CosineAnnealingWarmRestarts (T_0={T_0}, T_mult={T_mult}, eta_min={eta_min})"
    )
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("cosine_restarts.png", dpi=150)
    print("Saved plot to cosine_restarts.png")

    # Annotate restart points
    for i, lr in enumerate(lrs):
        if i > 0 and lrs[i - 1] < lr:
            plt.axvline(x=i, color="#FF5722", linestyle="--", alpha=0.5)
    plt.savefig("cosine_restarts.png", dpi=150)
    plt.show()


if __name__ == "__main__":
    import torch

    model = torch.nn.Linear(10, 2)
    optimizer = torch.optim.SGD([{"params": [model.weight], "lr": 1.0}])
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)

    print(f"{'Step':>5} | {'LR':>10} | Period (T_i)")
    print("-" * 35)

    periods = [scheduler.T_i]
    for epoch in range(1, 61):
        scheduler.step()
        lr = optimizer.param_groups[0]["lr"]
        if scheduler.t_cur == 0 and epoch > 1:
            periods.append(scheduler.T_i)
        if epoch % 10 == 0 or scheduler.t_cur == 0:
            print(
                f"{epoch:>5} | {lr:>10.6f} | {scheduler.T_i} "
                + (" <-- restart" if scheduler.t_cur == 0 and epoch > 1 else "")
            )

    print("\nT_i sequence (period doubles after each restart):", periods)

    # Plot the curve
    plot_restarts_curve()
