"""
Measure what GradScaler recovers vs raw FP16.

Creates a simple model, computes gradients in three modes:
1. FP32 (ground truth)
2. FP16 without GradScaler (underflow)
3. FP16 with GradScaler

Reports gradient recovery rate for each parameter.
Run: python grad_scaler.py [--hidden-size N] [--steps N]
"""

import argparse
import torch
import torch.nn as nn
import torch.cuda.amp as amp


class SimpleModel(nn.Module):
    def __init__(self, hidden_size=256):
        super().__init__()
        self.linear1 = nn.Linear(784, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, 10)

    def forward(self, x):
        x = torch.relu(self.linear1(x))
        x = torch.relu(self.linear2(x))
        return self.linear3(x)


def compute_grads_fp32(model, inputs, targets, loss_fn):
    """Compute gradients in FP32 (ground truth)."""
    model.zero_grad()
    output = model(inputs)
    loss = loss_fn(output, targets)
    loss.backward()
    grads = {n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None}
    return grads


def compute_grads_fp16_raw(model, inputs, targets, loss_fn):
    """Compute gradients in FP16 without GradScaler (will underflow)."""
    model.zero_grad()
    with amp.autocast(dtype=torch.float16):
        output = model(inputs)
        loss = loss_fn(output, targets)
    loss.backward()
    grads = {n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None}
    return grads


def compute_grads_fp16_scaled(model, inputs, targets, loss_fn, scale_factor=128.0):
    """Compute gradients in FP16 with manual loss scaling."""
    model.zero_grad()
    with amp.autocast(dtype=torch.float16):
        output = model(inputs)
        loss = loss_fn(output, targets)
    scaled_loss = loss * scale_factor
    scaled_loss.backward()
    grads = {n: p.grad.clone() / scale_factor for n, p in model.named_parameters() if p.grad is not None}
    return grads


def recovery_rate(approx_grads, true_grads):
    """Fraction of true gradient magnitude recovered."""
    total_true = sum(g.abs().sum().item() for g in true_grads.values())
    total_approx = sum(g.abs().sum().item() for g in approx_grads.values())
    if total_true == 0:
        return 0.0
    return total_approx / total_true


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hidden-size", type=int, default=256)
    parser.add_argument("--steps", type=int, default=100)
    args = parser.parse_args()

    device = torch.device("cuda")
    model = SimpleModel(hidden_size=args.hidden_size).to(device)
    loss_fn = nn.CrossEntropyLoss()

    # Use fixed seeds for reproducibility
    torch.manual_seed(42)
    inputs = torch.randn(128, 784, device=device)
    targets = torch.randint(0, 10, (128,), device=device)

    print(f"Configuration: hidden={args.hidden_size}, steps={args.steps}")
    print()

    # Run multiple steps and average
    recovery_fp16_raw = []
    recovery_fp16_scaled = []

    for step in range(args.steps):
        # Regenerate data each step to get varying gradient magnitudes
        inputs = torch.randn(128, 784, device=device)
        targets = torch.randint(0, 10, (128,), device=device)

        grads_fp32 = compute_grads_fp32(model, inputs, targets, loss_fn)
        grads_fp16_raw = compute_grads_fp16_raw(model, inputs, targets, loss_fn)
        grads_fp16_scaled = compute_grads_fp16_scaled(model, inputs, targets, loss_fn)

        recovery_fp16_raw.append(recovery_rate(grads_fp16_raw, grads_fp32))
        recovery_fp16_scaled.append(recovery_rate(grads_fp16_scaled, grads_fp32))

        # Zero gradients for next iteration
        model.zero_grad()

    avg_raw = sum(recovery_fp16_raw) / len(recovery_fp16_raw)
    avg_scaled = sum(recovery_fp16_scaled) / len(recovery_fp16_scaled)

    print("Gradient Recovery Rates:")
    print(f"  FP16 without GradScaler: {avg_raw*100:.1f}%")
    print(f"  FP16 with GradScaler:    {avg_scaled*100:.1f}%")
    print()
    print(f"GradScaler recovers {avg_scaled/avg_raw:.1f}x more gradient magnitude")


if __name__ == "__main__":
    main()
