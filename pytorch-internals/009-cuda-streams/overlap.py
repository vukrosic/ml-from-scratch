"""overlap.py — simulate overlapping data transfer and computation."""

import torch
import time


def overlapped_compute_vs_transfer(batch_size=2048, hidden=512, iterations=20):
    """
    Compare sequential vs overlapped data transfer and computation.

    Sequential: transfer -> compute -> transfer -> compute
    Overlapped: transfer (async) + compute (async) interleaved
    """
    device = 'cuda'

    # Initial batch on GPU
    current = torch.randn(batch_size, hidden, device=device)
    weight = torch.randn(hidden, hidden, device=device)

    compute_stream = torch.cuda.Stream()
    transfer_stream = torch.cuda.Stream()

    # --- Sequential baseline ---
    torch.cuda.synchronize()
    t0 = time.perf_counter()

    for _ in range(iterations):
        # Simulate transfer (CPU->GPU copy)
        next_batch = torch.randn(batch_size, hidden, device=device)

        # Compute on current batch
        result = current @ weight

        current = next_batch

    torch.cuda.synchronize()
    sequential_time = time.perf_counter() - t0

    # --- Overlapped version ---
    torch.cuda.synchronize()
    t0 = time.perf_counter()

    # Pre-transfer first extra batch
    next_batch = torch.randn(batch_size, hidden, device=device)

    for _ in range(iterations):
        # Launch compute on current batch using compute_stream
        with torch.cuda.stream(compute_stream):
            result = current @ weight

        # Launch transfer of next batch using transfer_stream
        with torch.cuda.stream(transfer_stream):
            following_batch = torch.randn(batch_size, hidden, device=device)

        # Wait for compute to finish before advancing
        compute_stream.synchronize()

        current = next_batch
        next_batch = following_batch

    torch.cuda.synchronize()
    overlapped_time = time.perf_counter() - t0

    print(f"Sequential:  {sequential_time*1000:.2f} ms for {iterations} iterations")
    print(f"Overlapped:  {overlapped_time*1000:.2f} ms for {iterations} iterations")
    print(f"Speedup:     {sequential_time/overlapped_time:.2f}x")

    return sequential_time, overlapped_time


def overlapped_pipeline_realistic(batch_size=1024, hidden=256, num_batches=50):
    """
    Realistic pipeline: matmul + relu + matmul compute per batch.
    Overlap transfer of batch N+1 with compute of batch N.
    """
    device = 'cuda'
    compute_stream = torch.cuda.Stream()
    transfer_stream = torch.cuda.Stream()

    # Model weights
    w1 = torch.randn(hidden * 4, hidden, device=device)
    w2 = torch.randn(hidden, hidden * 4, device=device)

    def compute_fn(x, w1, w2, stream):
        with torch.cuda.stream(stream):
            h = x @ w1
            h = torch.relu(h)
            out = h @ w2
            return out

    # Initialize
    current = torch.randn(batch_size, hidden, device=device)
    next_batch = torch.randn(batch_size, hidden, device=device)

    torch.cuda.synchronize()
    t0 = time.perf_counter()

    for i in range(num_batches):
        # Launch compute on current batch
        future_result = compute_fn(current, w1, w2, compute_stream)

        # Launch transfer of next batch
        if i < num_batches - 1:
            with torch.cuda.stream(transfer_stream):
                next_batch = torch.randn(batch_size, hidden, device=device)

        # Wait for compute to finish before advancing
        compute_stream.synchronize()

        current = next_batch

    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    print(f"Pipeline ({num_batches} batches, {batch_size}x{hidden}): {elapsed*1000:.2f} ms total")
    print(f"Per batch (overlapped): {elapsed/num_batches*1000:.3f} ms")

    return elapsed


def main():
    print("=== Data Transfer vs Compute Overlap ===\n")
    overlapped_compute_vs_transfer(batch_size=2048, hidden=512, iterations=20)

    print("\n=== Realistic Pipeline ===\n")
    overlapped_pipeline_realistic(batch_size=1024, hidden=256, num_batches=50)


if __name__ == "__main__":
    main()
