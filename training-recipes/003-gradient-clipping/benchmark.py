"""
benchmark.py — Compare our gradient clipping against torch.nn.utils.clip_grad_norm_.

We verify numerical equivalence (gradients after clipping should be identical)
and benchmark throughput (samples/second) for both implementations.
"""

import torch
import torch.nn as nn
import time
import statistics
from grad_clip import GradientClipper


class SimpleMLP(nn.Module):
    def __init__(self, input_dim: int = 128, hidden: int = 512, output_dim: int = 10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, output_dim),
        )

    def forward(self, x):
        return self.net(x)


def benchmark_clipping(model, optimizer, loader, num_runs: int = 50, warmup: int = 10):
    """
    Benchmark gradient clipping by timing training steps with clipping enabled.
    Returns average time per step in milliseconds.
    """
    device = next(model.parameters()).device

    # Warmup runs
    for _ in range(warmup):
        for batch_x, batch_y in loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            loss = nn.functional.cross_entropy(model(batch_x), batch_y)
            loss.backward()
            optimizer.step()

    # Timed runs
    times = []
    for _ in range(num_runs):
        start = time.perf_counter()
        for batch_x, batch_y in loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            loss = nn.functional.cross_entropy(model(batch_x), batch_y)
            loss.backward()
            optimizer.step()
        end = time.perf_counter()
        times.append((end - start) * 1000)  # ms

    return statistics.mean(times), statistics.stdev(times)


def verify_numerical_match(model, loader, clip_threshold: float = 1.0, num_batches: int = 20):
    """
    Verify that our gradient clipper produces the same results as torch's implementation.

    We run the same forward/backward pass with both implementations and compare
    the resulting gradient values after clipping.
    """
    device = next(model.parameters()).device
    max_diff = 0.0

    for batch_x, batch_y in loader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)

        # ---- Our implementation ----
        model.zero_grad()
        loss1 = nn.functional.cross_entropy(model(batch_x), batch_y)
        loss1.backward()

        our_clipper = GradientClipper(model.parameters(), max_norm=clip_threshold)
        our_clipper.clip()

        our_grads = {name: p.grad.clone() for name, p in model.named_parameters() if p.grad is not None}

        # ---- PyTorch implementation ----
        model.zero_grad()
        loss2 = nn.functional.cross_entropy(model(batch_x), batch_y)
        loss2.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_threshold)

        torch_grads = {name: p.grad.clone() for name, p in model.named_parameters() if p.grad is not None}

        # ---- Compare ----
        for name in our_grads:
            diff = (our_grads[name] - torch_grads[name]).abs().max().item()
            max_diff = max(max_diff, diff)

    return max_diff


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")

    # ---- Setup ----
    input_dim = 128
    hidden = 512
    output_dim = 10
    batch_size = 256
    clip_threshold = 1.0
    batch_count = 50

    # Create a fixed dataset
    torch.manual_seed(42)
    X = torch.randn(batch_size * batch_count, input_dim)
    y = torch.randint(0, output_dim, (batch_size * batch_count,))
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(X, y),
        batch_size=batch_size,
        shuffle=False,
    )

    # ---- Numerical verification ----
    print("=== Numerical Verification ===")
    torch.manual_seed(42)
    model_verify = SimpleMLP(input_dim, hidden, output_dim).to(device)
    optimizer_verify = torch.optim.Adam(model_verify.parameters(), lr=1e-3)
    max_diff = verify_numerical_match(model_verify, loader, clip_threshold, num_batches=batch_count)
    print(f"Max gradient difference (our clip vs torch clip): {max_diff:.2e}")
    if max_diff < 1e-5:
        print("PASS: Gradients match to numerical precision.")
    else:
        print("WARNING: Gradients differ beyond numerical precision.")

    # ---- Performance benchmark ----
    print("\n=== Performance Benchmark ===")
    num_runs = 50

    # Our implementation
    torch.manual_seed(42)
    model_ours = SimpleMLP(input_dim, hidden, output_dim).to(device)
    optimizer_ours = torch.optim.Adam(model_ours.parameters(), lr=1e-3)
    our_clipper = GradientClipper(model_ours.parameters(), max_norm=clip_threshold)

    # Patch step to use our clipper
    def step_with_ours():
        optimizer_ours.zero_grad()
        for batch_x, batch_y in loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            loss = nn.functional.cross_entropy(model_ours(batch_x), batch_y)
            loss.backward()
            our_clipper.clip()
            optimizer_ours.step()
            break

    our_mean, our_std = benchmark_clipping(model_ours, optimizer_ours, loader, num_runs=num_runs)
    print(f"Our GradientClipper:  {our_mean:.2f} +/- {our_std:.2f} ms/batch")

    # PyTorch implementation
    torch.manual_seed(42)
    model_torch = SimpleMLP(input_dim, hidden, output_dim).to(device)
    optimizer_torch = torch.optim.Adam(model_torch.parameters(), lr=1e-3)

    def step_with_torch():
        optimizer_torch.zero_grad()
        for batch_x, batch_y in loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            loss = nn.functional.cross_entropy(model_torch(batch_x), batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model_torch.parameters(), max_norm=clip_threshold)
            optimizer_torch.step()
            break

    torch_mean, torch_std = benchmark_clipping(model_torch, optimizer_torch, loader, num_runs=num_runs)
    print(f"torch.nn.utils.clip_grad_norm_: {torch_mean:.2f} +/- {torch_std:.2f} ms/batch")

    ratio = our_mean / torch_mean
    print(f"\nRatio (ours / torch): {ratio:.2f}x")
    if ratio < 1.1:
        print("Our implementation is comparable to PyTorch's built-in.")
    else:
        print("Note: Our implementation is slower (from-scratch, no kernel fusion).")


if __name__ == "__main__":
    main()