"""
benchmark.py — Benchmark throughput (samples/sec) for each scheduler.

We measure how many samples per second each scheduler can process during
training. The scheduler itself has minimal overhead, but this lets us
profile the full training step cost with different LR schedules.
"""

import time
import torch
import torch.nn as nn
from schedulers import ConstantLR, StepDecay, CosineAnnealing, LinearWarmupCosineDecay


class SimpleMLP(nn.Module):
    def __init__(self, input_dim: int = 128, hidden: int = 512, output_dim: int = 10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, output_dim),
        )

    def forward(self, x):
        return self.net(x)


def get_scheduler(name: str, optimizer, total_steps: int):
    if name == "constant":
        return ConstantLR(optimizer)
    elif name == "step":
        return StepDecay(optimizer, step_size=10, gamma=0.5)
    elif name == "cosine":
        return CosineAnnealing(optimizer, total_steps=total_steps)
    elif name == "warmup_cosine":
        return LinearWarmupCosineDecay(
            optimizer, warmup_steps=int(0.1 * total_steps), total_steps=total_steps
        )
    else:
        raise ValueError(f"Unknown scheduler: {name}")


def benchmark(scheduler_name: str, device, epochs: int = 10, batch_size: int = 256):
    """Train one epoch and measure throughput (samples/sec)."""
    torch.manual_seed(42)
    input_dim, hidden, output_dim = 128, 512, 10

    model = SimpleMLP(input_dim, hidden, output_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = get_scheduler(scheduler_name, optimizer, total_steps=epochs)

    X = torch.randn(4096, input_dim, device=device)
    y = torch.randint(0, output_dim, (4096,), device=device)
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(X, y),
        batch_size=batch_size,
        shuffle=True,
    )

    model.train()
    total_tokens = 0
    start = time.perf_counter()

    for batch_x, batch_y in loader:
        optimizer.zero_grad()
        logits = model(batch_x)
        loss = nn.functional.cross_entropy(logits, batch_y)
        loss.backward()
        optimizer.step()
        scheduler.step()
        total_tokens += batch_x.size(0)

    elapsed = time.perf_counter() - start
    throughput = total_tokens / elapsed
    return throughput, elapsed


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    schedulers = ["constant", "step", "cosine", "warmup_cosine"]
    names = {
        "constant": "ConstantLR",
        "step": "StepDecay",
        "cosine": "CosineAnnealing",
        "warmup_cosine": "LinearWarmupCosineDecay",
    }

    print(f"\n{'Scheduler':<25} {'Throughput (samples/s)':<25} {'Time (s)':<10}")
    print("-" * 60)

    for s in schedulers:
        throughput, elapsed = benchmark(s, device, epochs=10, batch_size=256)
        print(f"{names[s]:<25} {throughput:>15,.0f}       {elapsed:>8.3f}")

    print("\nNote: Scheduler overhead is negligible vs. backward pass.")


if __name__ == "__main__":
    main()
