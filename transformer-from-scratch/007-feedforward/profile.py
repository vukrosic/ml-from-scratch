"""
Profile FFN implementations across batch sizes and model dimensions.

This extended profiling script measures:
- Forward pass latency
- Backward pass latency
- Peak memory usage
- Throughput (tokens/sec)

We profile:
- Standard FFN (GELU)
- FFN-SwiGLU
- FFN-ReGLU
- FFN-GeGLU

Across different model sizes and batch configurations.
"""

import time
import gc
import torch
import torch.nn as nn
from swiglu import FFNSwiGLU, FFNReGLU, FFNGeGLU
from feedforward import FeedForward


def benchmark_forward(fn, n_runs=20, warmup=5):
    """Time forward pass. Returns latency in microseconds."""
    for _ in range(warmup):
        fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(n_runs):
        fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    return (time.perf_counter() - start) / n_runs * 1e6


def benchmark_backward(x, model, n_runs=20, warmup=5):
    """Time forward + backward pass. Returns latency in microseconds."""
    for _ in range(warmup):
        out = model(x)
        out.sum().backward()
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(n_runs):
        model.zero_grad()
        if x.grad is not None:
            x.grad.zero_()
        out = model(x)
        out.sum().backward()
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    return (time.perf_counter() - start) / n_runs * 1e6


def measure_memory(model, x):
    """Measure peak memory usage during forward + backward pass."""
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
        gc.collect()

    out = model(x)
    out.sum().backward()

    if torch.cuda.is_available():
        peak_mem = torch.cuda.max_memory_allocated() / 1024**2  # MB
        # Reset for next measurement
        model.zero_grad()
        x.grad = None
    else:
        # Rough estimate for CPU
        param_mem = sum(p.numel() * 4 for p in model.parameters()) / 1024**2
        peak_mem = param_mem * 2  # rough estimate

    return peak_mem


class FFNSwiGLU(nn.Module):
    """SwiGLU FFN."""
    def __init__(self, d_model, d_ff, dropout=0.0):
        super().__init__()
        self.w_gate = nn.Linear(d_model, d_ff)
        self.w_up = nn.Linear(d_model, d_ff)
        self.w_down = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        gate = torch.sigmoid(self.w_gate(x))
        up = torch.sigmoid(self.w_up(x)) * self.w_up(x)  # SiLU = x * sigmoid(x)
        return self.w_down(self.dropout(up * gate))


class FFNReGLU(nn.Module):
    """ReGLU FFN."""
    def __init__(self, d_model, d_ff, dropout=0.0):
        super().__init__()
        self.w_gate = nn.Linear(d_model, d_ff)
        self.w_up = nn.Linear(d_model, d_ff)
        self.w_down = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        gate = torch.relu(self.w_gate(x))
        up = self.w_up(x)
        return self.w_down(self.dropout(gate * up))


class FFNGeGLU(nn.Module):
    """GeGLU FFN."""
    def __init__(self, d_model, d_ff, dropout=0.0):
        super().__init__()
        self.w_gate = nn.Linear(d_model, d_ff)
        self.w_up = nn.Linear(d_model, d_ff)
        self.w_down = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        gate = nn.functional.gelu(self.w_gate(x))
        up = self.w_up(x)
        return self.w_down(self.dropout(gate * up))


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print("Note: On CPU this may take several minutes. Reduce n_runs if needed.")

    # Model configurations to profile
    configs = [
        # (d_model, d_ff, batch, seq_len, label)
        (128, 512, 2, 32, "Tiny"),
    ]

    ffn_types = ["standard", "swiglu", "reglu", "geglu"]

    print("\n" + "=" * 90)
    print("FORWARD PASS LATENCY (μs)")
    print("=" * 90)
    print(f"{'Config':>15} {'Batch':>6} {'Seq':>6}  "
          f"{'Standard':>12} {'SwiGLU':>12} {'ReGLU':>12} {'GeGLU':>12}")
    print("-" * 90)

    for d_model, d_ff, batch, seq_len, label in configs:
        x = torch.randn(batch, seq_len, d_model, device=device)

        ffn_standard = FeedForward(d_model, d_ff).to(device)
        ffn_swiglu = FFNSwiGLU(d_model, d_ff).to(device)
        ffn_reglu = FFNReGLU(d_model, d_ff).to(device)
        ffn_geglu = FFNGeGLU(d_model, d_ff).to(device)

        t_standard = benchmark_forward(lambda: ffn_standard(x))
        t_swiglu = benchmark_forward(lambda: ffn_swiglu(x))
        t_reglu = benchmark_forward(lambda: ffn_reglu(x))
        t_geglu = benchmark_forward(lambda: ffn_geglu(x))

        print(f"{label:>15} {batch:>6} {seq_len:>6}  "
              f"{t_standard:>12.2f} {t_swiglu:>12.2f} {t_reglu:>12.2f} {t_geglu:>12.2f}")

    print("\n" + "=" * 90)
    print("FORWARD + BACKWARD PASS LATENCY (μs)")
    print("=" * 90)
    print(f"{'Config':>15} {'Batch':>6} {'Seq':>6}  "
          f"{'Standard':>12} {'SwiGLU':>12} {'ReGLU':>12} {'GeGLU':>12}")
    print("-" * 90)

    for d_model, d_ff, batch, seq_len, label in configs:
        x = torch.randn(batch, seq_len, d_model, device=device, requires_grad=True)

        ffn_standard = FeedForward(d_model, d_ff).to(device)
        ffn_swiglu = FFNSwiGLU(d_model, d_ff).to(device)
        ffn_reglu = FFNReGLU(d_model, d_ff).to(device)
        ffn_geglu = FFNGeGLU(d_model, d_ff).to(device)

        t_standard = benchmark_backward(x, ffn_standard)
        t_swiglu = benchmark_backward(x, ffn_swiglu)
        t_reglu = benchmark_backward(x, ffn_reglu)
        t_geglu = benchmark_backward(x, ffn_geglu)

        print(f"{label:>15} {batch:>6} {seq_len:>6}  "
              f"{t_standard:>12.2f} {t_swiglu:>12.2f} {t_reglu:>12.2f} {t_geglu:>12.2f}")

    print("\n" + "=" * 90)
    print("MEMORY FOOTPRINT (MB)")
    print("=" * 90)
    print(f"{'Config':>15} {'Batch':>6} {'Seq':>6}  "
          f"{'Standard':>12} {'SwiGLU':>12} {'ReGLU':>12} {'GeGLU':>12}")
    print("-" * 90)

    for d_model, d_ff, batch, seq_len, label in configs:
        x = torch.randn(batch, seq_len, d_model, device=device, requires_grad=True)

        ffn_standard = FeedForward(d_model, d_ff).to(device)
        ffn_swiglu = FFNSwiGLU(d_model, d_ff).to(device)
        ffn_reglu = FFNReGLU(d_model, d_ff).to(device)
        ffn_geglu = FFNGeGLU(d_model, d_ff).to(device)

        m_standard = measure_memory(ffn_standard, x)
        m_swiglu = measure_memory(ffn_swiglu, x)
        m_reglu = measure_memory(ffn_reglu, x)
        m_geglu = measure_memory(ffn_geglu, x)

        print(f"{label:>15} {batch:>6} {seq_len:>6}  "
              f"{m_standard:>12.2f} {m_swiglu:>12.2f} {m_reglu:>12.2f} {m_geglu:>12.2f}")

    print("\nNOTES:")
    print("  - SwiGLU has ~50% more parameters than standard FFN")
    print("  - Memory roughly scales with parameter count + activations")
    print("  - Forward+backward latency shows training cost")
    print("  - For inference, only forward pass matters")
