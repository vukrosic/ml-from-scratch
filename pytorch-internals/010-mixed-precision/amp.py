"""
FP32 vs FP16 vs BF16 training comparison.

Measures throughput (steps/sec) and memory usage for each precision mode.
Run: python amp.py [--hidden-size N] [--batch-size N] [--steps N]
"""

import argparse
import time
import torch
import torch.nn as nn
import torch.cuda.amp as amp


class MLP(nn.Module):
    def __init__(self, hidden_size=512, num_layers=4):
        super().__init__()
        layers = []
        in_dim = 784
        for _ in range(num_layers):
            layers.append(nn.Linear(in_dim, hidden_size))
            layers.append(nn.ReLU())
            in_dim = hidden_size
        layers.append(nn.Linear(hidden_size, 10))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def benchmark_precision(precision, hidden_size, batch_size, steps, warmup):
    """Train in given precision mode and return throughput and memory."""
    device = torch.device("cuda")
    model = MLP(hidden_size=hidden_size).to(device)
    optimizer = torch.optim.Adam(model.parameters())
    loss_fn = nn.CrossEntropyLoss()

    # Fake data
    dummy_input = torch.randn(batch_size, 784, device=device)
    dummy_target = torch.randint(0, 10, (batch_size,), device=device)

    if precision == "fp16":
        scaler = amp.GradScaler()
    else:
        scaler = None

    # Warmup
    for _ in range(warmup):
        optimizer.zero_grad()
        with amp.autocast(dtype=torch.bfloat16 if precision == "bf16" else torch.float16):
            output = model(dummy_input)
            loss = loss_fn(output, dummy_target)
        if scaler:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

    torch.cuda.synchronize()

    # Timing
    t0 = time.perf_counter()
    for _ in range(steps):
        optimizer.zero_grad()
        with amp.autocast(dtype=torch.bfloat16 if precision == "bf16" else torch.float16):
            output = model(dummy_input)
            loss = loss_fn(output, dummy_target)
        if scaler:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    mem_used = torch.cuda.max_memory_allocated() / 1e9  # GB

    return elapsed / steps, mem_used


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hidden-size", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--warmup", type=int, default=50)
    args = parser.parse_args()

    print(f"Configuration: hidden={args.hidden_size}, batch={args.batch_size}, steps={args.steps}")
    print()

    results = {}
    for precision in ["fp32", "fp16", "bf16"]:
        print(f"Running {precision}...")
        throughput, mem = benchmark_precision(
            precision, args.hidden_size, args.batch_size, args.steps, args.warmup
        )
        results[precision] = (throughput, mem)
        print(f"  Throughput: {throughput*1000:.2f} ms/step")
        print(f"  Memory:     {mem:.2f} GB")

    print()
    print("Summary (speedup vs FP32):")
    fp32_throughput = results["fp32"][0]
    for precision in ["fp16", "bf16"]:
        t, m = results[precision]
        print(f"  {precision}: {fp32_throughput/t:.2f}x faster, {m:.2f} GB")


if __name__ == "__main__":
    main()
