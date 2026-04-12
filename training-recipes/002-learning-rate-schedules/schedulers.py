"""
schedulers.py — Learning rate schedules from scratch.

We implement four schedulers:
    1. Constant      — fixed LR throughout training.
    2. Step Decay    — drop LR by a factor every N epochs.
    3. Cosine Annealing — decay LR along a half-cosine curve.
    4. Linear Warmup + Decay — ramp up LR for the first N steps, then decay.

These match the behavior of:
    - torch.optim.lr_scheduler.StepLR
    - torch.optim.lr_scheduler.CosineAnnealingLR
    - torch.optim.lr_scheduler.LinearLR (with warmup)

Each scheduler is a simple class that holds state and exposes get_lr().
"""

import math
from typing import Optional


class ConstantLR:
    """
    Constant learning rate — the baseline.

    lr(t) = initial_lr  for all t
    """

    def __init__(self, optimizer, last_epoch: int = -1):
        self.optimizer = optimizer
        self.initial_lr = optimizer.param_groups[0]["lr"]
        self.last_epoch = last_epoch

    def get_lr(self):
        return [self.initial_lr for _ in self.optimizer.param_groups]

    def step(self):
        self.last_epoch += 1
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group["lr"] = lr


class StepDecay:
    """
    Decay the learning rate by `gamma` every `step_size` epochs.

    lr(t) = initial_lr * gamma ** floor(t / step_size)

    Example: step_size=10, gamma=0.5 → LR halves every 10 epochs.
    Matches: torch.optim.lr_scheduler.StepLR
    """

    def __init__(
        self,
        optimizer,
        step_size: int = 10,
        gamma: float = 0.1,
        last_epoch: int = -1,
    ):
        self.optimizer = optimizer
        self.step_size = step_size
        self.gamma = gamma
        self.initial_lr = optimizer.param_groups[0]["lr"]
        self.last_epoch = last_epoch

    def get_lr(self):
        factor = self.gamma ** (self.last_epoch // self.step_size)
        return [self.initial_lr * factor for _ in self.optimizer.param_groups]

    def step(self):
        self.last_epoch += 1
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group["lr"] = lr


class CosineAnnealing:
    """
    Decay learning rate along a cosine curve from initial_lr to 0.

    lr(t) = initial_lr * (1 + cos(pi * t / T)) / 2

    where T = total_steps (or epochs). The LR reaches near-zero at the end.

    Matches: torch.optim.lr_scheduler.CosineAnnealingLR
    """

    def __init__(
        self,
        optimizer,
        total_steps: int,
        eta_min: float = 0.0,
        last_epoch: int = -1,
    ):
        self.optimizer = optimizer
        self.total_steps = total_steps
        self.eta_min = eta_min
        self.initial_lr = optimizer.param_groups[0]["lr"]
        self.last_epoch = last_epoch

    def get_lr(self):
        if self.last_epoch >= self.total_steps:
            return [self.eta_min for _ in self.optimizer.param_groups]
        progress = self.last_epoch / self.total_steps
        cosine = (1 + math.cos(math.pi * progress)) / 2
        return [
            self.eta_min + (self.initial_lr - self.eta_min) * cosine
            for _ in self.optimizer.param_groups
        ]

    def step(self):
        self.last_epoch += 1
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group["lr"] = lr


class LinearWarmupCosineDecay:
    """
    Warmup linearly from lr=0 to initial_lr over warmup_steps,
    then decay along a cosine curve to eta_min.

    Phase 1 (warmup):   lr(t) = initial_lr * t / warmup_steps
    Phase 2 (decay):    lr(t) = eta_min + (initial_lr - eta_min)
                        * (1 + cos(pi * (t - warmup_steps) / (total_steps - warmup_steps))) / 2

    Common in transformers (BERT, GPT). Matches parts of:
        - torch.optim.lr_scheduler.LinearLR (warmup phase)
        - torch.optim.lr_scheduler.CosineAnnealingLR (decay phase)
    """

    def __init__(
        self,
        optimizer,
        warmup_steps: int,
        total_steps: int,
        eta_min: float = 0.0,
        last_epoch: int = -1,
    ):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.eta_min = eta_min
        self.initial_lr = optimizer.param_groups[0]["lr"]
        self.last_epoch = last_epoch

    def get_lr(self):
        t = self.last_epoch
        if t < self.warmup_steps:
            # Linear warmup
            return [
                self.initial_lr * t / self.warmup_steps
                for _ in self.optimizer.param_groups
            ]
        elif t < self.total_steps:
            # Cosine decay
            progress = (t - self.warmup_steps) / (self.total_steps - self.warmup_steps)
            cosine = (1 + math.cos(math.pi * progress)) / 2
            return [
                self.eta_min + (self.initial_lr - self.eta_min) * cosine
                for _ in self.optimizer.param_groups
            ]
        else:
            # Done — stay at eta_min
            return [self.eta_min for _ in self.optimizer.param_groups]

    def step(self):
        self.last_epoch += 1
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group["lr"] = lr


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import torch

    model = torch.nn.Linear(10, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=1.0)

    epochs = 50
    warmup_epochs = 5
    scheduler = LinearWarmupCosineDecay(
        optimizer, warmup_steps=warmup_epochs, total_steps=epochs
    )

    print(f"{'Epoch':>6} | {'LR':>12}")
    print("-" * 22)
    for epoch in range(epochs):
        scheduler.step()
        lr = optimizer.param_groups[0]["lr"]
        marker = " <-- warmup" if epoch < warmup_epochs else ""
        if epoch % 5 == 0 or epoch < warmup_epochs:
            print(f"{epoch:>6} | {lr:>12.6f}{marker}")

    print("\nSchedulers implemented: ConstantLR, StepDecay, CosineAnnealing, LinearWarmupCosineDecay")
