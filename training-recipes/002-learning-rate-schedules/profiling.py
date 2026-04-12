"""
profile.py — Profile the overhead of each scheduler vs. the full training step.

We break down the cost of each component in a training step:
    1. Forward pass
    2. Backward pass
    3. Optimizer step
    4. Scheduler step

The scheduler is typically O(1) per step (just a few arithmetic ops per
parameter group), so its contribution is negligible vs. the backward pass.
"""

import time
import torch
import torch.nn as nn
from schedulers import (
    ConstantLR,
    StepDecay,
    CosineAnnealing,
    LinearWarmupCosineDecay,
)


class MediumMLP(nn.Module):
    """Medium-sized MLP for realistic profiling."""

    def __init__(self, d_model: int = 256, d_hidden: int = 1024, n_layers: int = 4):
        super().__init__()
        layers = []
        in_dim = d_model
        for _ in range(n_layers):
            layers.extend([nn.Linear(in_dim, d_hidden), nn.GELU()])
            in_dim = d_hidden
        layers.append(nn.Linear(in_dim, d_model))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def profile_step_components(
    model, optimizer, scheduler, X, y, device, n_steps: int = 100
):
    """
    Profile the time spent in each component of a training step.

    Returns a dict with timing breakdown.
    """
    model.train()
    timings = {
        "forward": 0.0,
        "backward": 0.0,
        "optimizer": 0.0,
        "scheduler": 0.0,
    }

    # Warmup
    for _ in range(5):
        optimizer.zero_grad()
        out = model(X)
        loss = nn.functional.cross_entropy(out, y)
        loss.backward()
        optimizer.step()
        scheduler.step()

    # Profile
    for _ in range(n_steps):
        # Forward
        t0 = time.perf_counter()
        optimizer.zero_grad()
        out = model(X)
        loss = nn.functional.cross_entropy(out, y)
        timings["forward"] += time.perf_counter() - t0

        # Backward
        t0 = time.perf_counter()
        loss.backward()
        timings["backward"] += time.perf_counter() - t0

        # Optimizer
        t0 = time.perf_counter()
        optimizer.step()
        timings["optimizer"] += time.perf_counter() - t0

        # Scheduler
        t0 = time.perf_counter()
        scheduler.step()
        timings["scheduler"] += time.perf_counter() - t0

    # Average
    for k in timings:
        timings[k] /= n_steps
        timings[k] *= 1e6  # convert to microseconds

    return timings


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    batch_size = 128
    d_model = 256
    d_hidden = 1024
    n_layers = 4
    n_steps = 100

    model = MediumMLP(d_model, d_hidden, n_layers).to(device)
    X = torch.randn(batch_size, d_model, device=device)
    y = torch.randint(0, d_model, (batch_size,), device=device)

    schedulers = {
        "ConstantLR": ConstantLR,
        "StepDecay": lambda opt: StepDecay(opt, step_size=10, gamma=0.5),
        "CosineAnnealing": lambda opt: CosineAnnealing(opt, total_steps=1000),
        "LinearWarmupCosineDecay": lambda opt: LinearWarmupCosineDecay(
            opt, warmup_steps=100, total_steps=1000
        ),
    }

    print(
        f"\nModel: {sum(p.numel() for p in model.parameters()):,} parameters | "
        f"Batch size: {batch_size}"
    )
    print(f"\n{'Scheduler':<25} {'Forward (us)':<15} {'Backward (us)':<15} "
          f"{'Optimizer (us)':<17} {'Scheduler (us)':<15}")
    print("-" * 90)

    results = {}
    for name, SchedulerClass in schedulers.items():
        torch.manual_seed(42)
        model = MediumMLP(d_model, d_hidden, n_layers).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        scheduler = SchedulerClass(optimizer)

        timings = profile_step_components(
            model, optimizer, scheduler, X, y, device, n_steps
        )
        results[name] = timings

        print(
            f"{name:<25} "
            f"{timings['forward']:>12.1f}  "
            f"{timings['backward']:>12.1f}  "
            f"{timings['optimizer']:>14.1f}  "
            f"{timings['scheduler']:>12.1f}"
        )

    print("\n--- Key observations ---")
    for name, timings in results.items():
        total = sum(timings.values())
        sched_pct = 100 * timings["scheduler"] / total
        print(
            f"{name}: scheduler overhead = {sched_pct:.3f}% of total step time "
            f"({timings['scheduler']:.1f}us / {total:.1f}us)"
        )


if __name__ == "__main__":
    main()
