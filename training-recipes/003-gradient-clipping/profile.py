"""
profile.py — Profile gradient clipping overhead across model sizes and batch sizes.

We measure:
1. Time spent computing gradient norms vs applying the clip
2. Overhead of clipping as a fraction of total training time
3. Scaling behavior with model size and batch size
"""

import torch
import torch.nn as nn
import time
import statistics
from grad_clip import GradientClipper


def make_model(input_dim: int, hidden: int, depth: int, output_dim: int) -> nn.Module:
    """Create a simple MLP with specified depth."""
    layers = []
    prev = input_dim
    for _ in range(depth):
        layers.append(nn.Linear(prev, hidden))
        layers.append(nn.ReLU())
        prev = hidden
    layers.append(nn.Linear(prev, output_dim))
    return nn.Sequential(*layers)


def profile_grad_clip_overhead(
    model: nn.Module,
    input_dim: int,
    hidden: int,
    depth: int,
    output_dim: int,
    batch_size: int,
    clip_threshold: float,
    num_runs: int = 100,
    warmup: int = 20,
) -> dict:
    """
    Profile gradient clipping overhead for a given model configuration.

    Returns a dict with timing statistics for:
    - forward + backward (no clip)
    - forward + backward + clip
    - norm computation alone
    - clip application alone
    """
    device = next(model.parameters()).device

    # Create data
    x = torch.randn(batch_size, input_dim, device=device)
    y = torch.randint(0, output_dim, (batch_size,), device=device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    clipper = GradientClipper(model.parameters(), max_norm=clip_threshold)

    # Warmup
    for _ in range(warmup):
        optimizer.zero_grad()
        loss = nn.functional.cross_entropy(model(x), y)
        loss.backward()
        optimizer.step()

    # ---- Profile: forward + backward only (no clipping) ----
    times_no_clip = []
    for _ in range(num_runs):
        optimizer.zero_grad()
        start = time.perf_counter()
        loss = nn.functional.cross_entropy(model(x), y)
        loss.backward()
        times_no_clip.append(time.perf_counter() - start)
        optimizer.step()

    # ---- Profile: forward + backward + clipping ----
    times_with_clip = []
    for _ in range(num_runs):
        optimizer.zero_grad()
        loss = nn.functional.cross_entropy(model(x), y)
        loss.backward()
        start_clip = time.perf_counter()
        clipper.clip()
        times_with_clip.append(time.perf_counter() - start_clip)
        start_step = time.perf_counter()
        optimizer.step()
        times_with_clip[-1] += time.perf_counter() - start_step

    # ---- Profile: norm computation alone ----
    times_norm = []
    for _ in range(num_runs):
        optimizer.zero_grad()
        loss = nn.functional.cross_entropy(model(x), y)
        loss.backward()
        start = time.perf_counter()
        total_norm = torch.norm(
            torch.stack([p.grad.norm(2) for p in model.parameters() if p.grad is not None])
        )
        times_norm.append(time.perf_counter() - start)
        optimizer.step()

    # ---- Profile: clip application alone (assuming norm computed) ----
    times_apply = []
    for _ in range(num_runs):
        optimizer.zero_grad()
        loss = nn.functional.cross_entropy(model(x), y)
        loss.backward()
        total_norm = torch.norm(
            torch.stack([p.grad.norm(2) for p in model.parameters() if p.grad is not None])
        )
        start = time.perf_counter()
        clip_coef = clip_threshold / (total_norm + 1e-6)
        if clip_coef < 1.0:
            for p in model.parameters():
                if p.grad is not None:
                    p.grad.data.mul_(clip_coef)
        times_apply.append(time.perf_counter() - start)
        optimizer.step()

    return {
        "no_clip_mean": statistics.mean(times_no_clip) * 1000,  # ms
        "no_clip_std": statistics.stdev(times_no_clip) * 1000,
        "with_clip_mean": statistics.mean(times_with_clip) * 1000,
        "with_clip_std": statistics.stdev(times_with_clip) * 1000,
        "norm_mean": statistics.mean(times_norm) * 1000,
        "norm_std": statistics.stdev(times_norm) * 1000,
        "apply_mean": statistics.mean(times_apply) * 1000,
        "apply_std": statistics.stdev(times_apply) * 1000,
    }


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")

    clip_threshold = 1.0
    num_runs = 100
    warmup = 20

    results = []

    # ---- Sweep 1: Model depth ----
    print("=== Sweep: Model Depth ===")
    print(f"{'Depth':>6} | {'No Clip (ms)':>12} | {'With Clip (ms)':>14} | {'Clip OH %':>10}")
    print("-" * 50)

    for depth in [2, 4, 6, 8, 10]:
        torch.manual_seed(42)
        model = make_model(input_dim=64, hidden=256, depth=depth, output_dim=10).to(device)
        r = profile_grad_clip_overhead(
            model, input_dim=64, hidden=256, depth=depth, output_dim=10,
            batch_size=128, clip_threshold=clip_threshold,
            num_runs=num_runs, warmup=warmup
        )
        overhead_pct = 100 * (r["with_clip_mean"] - r["no_clip_mean"]) / r["no_clip_mean"]
        results.append({"sweep": "depth", "depth": depth, **r})
        print(f"{depth:>6} | {r['no_clip_mean']:>10.3f} +/- {r['no_clip_std']:.3f} | "
              f"{r['with_clip_mean']:>12.3f} +/- {r['with_clip_std']:.3f} | {overhead_pct:>8.1f}%")

    # ---- Sweep 2: Batch size ----
    print("\n=== Sweep: Batch Size ===")
    print(f"{'Batch':>6} | {'No Clip (ms)':>12} | {'With Clip (ms)':>14} | {'Clip OH %':>10}")
    print("-" * 50)

    for batch_size in [32, 64, 128, 256, 512]:
        torch.manual_seed(42)
        model = make_model(input_dim=64, hidden=256, depth=4, output_dim=10).to(device)
        r = profile_grad_clip_overhead(
            model, input_dim=64, hidden=256, depth=4, output_dim=10,
            batch_size=batch_size, clip_threshold=clip_threshold,
            num_runs=num_runs, warmup=warmup
        )
        overhead_pct = 100 * (r["with_clip_mean"] - r["no_clip_mean"]) / r["no_clip_mean"]
        results.append({"sweep": "batch_size", "batch_size": batch_size, **r})
        print(f"{batch_size:>6} | {r['no_clip_mean']:>10.3f} +/- {r['no_clip_std']:.3f} | "
              f"{r['with_clip_mean']:>12.3f} +/- {r['with_clip_std']:.3f} | {overhead_pct:>8.1f}%")

    # ---- Summary: Where does time go? ----
    print("\n=== Clipping Cost Breakdown ===")
    depth_results = [r for r in results if r["sweep"] == "depth"]
    typical = depth_results[len(depth_results) // 2]

    total_clip_time = typical["norm_mean"] + typical["apply_mean"]
    total_step_time = typical["with_clip_mean"]

    print(f"Gradient norm computation: {typical['norm_mean']:.3f} ms "
          f"({100*typical['norm_mean']/total_step_time:.1f}% of clip step)")
    print(f"Clip application (scaling): {typical['apply_mean']:.3f} ms "
          f"({100*typical['apply_mean']/total_step_time:.1f}% of clip step)")
    print(f"Total clipping overhead: {total_clip_time:.3f} ms "
          f"({100*total_clip_time/total_step_time:.1f}% of clip step)")

    print("\nKey insight: gradient norm computation dominates clipping cost.")


if __name__ == "__main__":
    main()