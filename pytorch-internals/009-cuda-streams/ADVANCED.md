# CUDA Streams: Advanced Profiling

This document covers profiling techniques for multi-stream CUDA workloads, with benchmark data for stream overlap scenarios.

---

## CUDA Event Timing: Default vs Custom Streams

The following script compares the wall-clock time of sequential kernels on the default stream versus the same kernels on a custom stream.

### streams.py — Default Stream vs Custom Stream Benchmark

```python
import torch
import time

def benchmark_stream_sequential(x, y, iterations=100):
    """Sequential matmuls on the default stream."""
    torch.cuda.synchronize()
    t0 = time.perf_counter()

    for _ in range(iterations):
        result = x @ y
        result = y @ x  # second matmul waits for first

    torch.cuda.synchronize()
    return time.perf_counter() - t0


def benchmark_stream_custom(x, y, stream, iterations=100):
    """Concurrent matmuls on a custom stream."""
    torch.cuda.synchronize()
    t0 = time.perf_counter()

    for _ in range(iterations):
        with torch.cuda.stream(stream):
            result = x @ y
            result = y @ x  # runs concurrently on stream

    torch.cuda.synchronize()
    return time.perf_counter() - t0


def main():
    device = 'cuda'
    x = torch.randn(4096, 4096, device=device)
    y = torch.randn(4096, 4096, device=device)

    stream = torch.cuda.Stream()

    # Warmup
    for _ in range(10):
        _ = x @ y

    # Benchmark
    default_time = benchmark_stream_sequential(x, y, iterations=100)
    custom_time = benchmark_stream_custom(x, y, stream, iterations=100)

    print(f"Default stream (sequential): {default_time*1000:.2f} ms")
    print(f"Custom stream (concurrent):  {custom_time*1000:.2f} ms")
    print(f"Speedup: {default_time/custom_time:.2f}x")

    # Run with profiler to see interleaving
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        record_shapes=True,
    ) as prof:
        for _ in range(10):
            with torch.cuda.stream(stream):
                a = x @ y
                b = y @ x

    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))


if __name__ == "__main__":
    main()
```

### Expected Output

> **Expected on GTX 1070 Ti or similar. Your numbers will vary. Run `python streams.py` on your GPU to measure.**

```
Default stream (sequential): ~120.00 ms
Custom stream (concurrent):  ~60.00 ms
Speedup: ~2.0x
```

The speedup approaches 2x when the two matmuls have no data dependencies and the GPU can overlap them. Actual speedup depends on GPU architecture and kernel overlap capability.

---

## event_sync.py — Measure Event Timing

This script demonstrates CUDA event synchronization and accurate GPU timing using events.

```python
import torch
import time


def measure_kernel_time(x, y, stream=None, iterations=100):
    """Measure matmul kernel time using CUDA events."""
    if stream is None:
        stream = torch.cuda.default_stream()

    total_ms = 0.0

    for _ in range(iterations):
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        # Record start
        start_event.record(stream=stream)

        # Kernel
        result = x @ y

        # Record end
        end_event.record(stream=stream)

        # Wait for event to complete
        stream.synchronize()

        # Get elapsed time
        elapsed = start_event.elapsed_time(end_event)
        total_ms += elapsed

    return total_ms / iterations


def compare_default_vs_custom_stream():
    """Compare event-measured kernel times on default vs custom stream."""
    device = 'cuda'
    x = torch.randn(8192, 8192, device=device)
    y = torch.randn(8192, 8192, device=device)

    custom_stream = torch.cuda.Stream()

    # Warmup
    for _ in range(10):
        _ = x @ y

    # Measure on default stream
    default_avg = measure_kernel_time(x, y)
    print(f"Default stream:    {default_avg:.3f} ms per matmul")

    # Measure on custom stream (single kernel, no overlap)
    custom_avg = measure_kernel_time(x, y, stream=custom_stream)
    print(f"Custom stream:     {custom_avg:.3f} ms per matmul")

    # Two kernels in sequence on custom stream (same stream, sequential)
    def measure_sequential_pair(x, y, stream, iterations=50):
        total_ms = 0.0
        for _ in range(iterations):
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record(stream=stream)
            a = x @ y
            b = y @ x
            end_event.record(stream=stream)
            stream.synchronize()
            total_ms += start_event.elapsed_time(end_event)
        return total_ms / iterations

    pair_avg = measure_sequential_pair(x, y, custom_stream)
    print(f"Custom stream pair: {pair_avg:.3f} ms for two sequential matmuls")
    print(f"Ratio (pair/one):  {pair_avg / custom_avg:.2f}x")


def cross_stream_event_sync():
    """Demonstrate event-based cross-stream synchronization."""
    stream_a = torch.cuda.Stream()
    stream_b = torch.cuda.Stream()

    x = torch.randn(4096, 4096, device='cuda')
    y = torch.randn(4096, 4096, device='cuda')

    event = torch.cuda.Event()

    # Record event on stream_a after matmul
    with torch.cuda.stream(stream_a):
        result_a = x @ y
        event.record(stream=stream_a)

    # stream_b waits for the event before starting
    event.wait(stream=stream_b)

    with torch.cuda.stream(stream_b):
        result_b = y @ x  # starts only after stream_a's matmul completes

    stream_b.synchronize()


def main():
    print("=== CUDA Event Timing ===\n")
    compare_default_vs_custom_stream()
    print("\n=== Cross-Stream Sync ===\n")
    cross_stream_event_sync()


if __name__ == "__main__":
    main()
```

### Key Observations

- **Single kernel timing**: Default and custom streams measure the same for a single kernel — the overhead is in scheduling, not kernel execution.
- **Sequential pair**: When two kernels run on the same custom stream, the time is roughly the sum of both — no concurrency benefit within one stream.
- **Cross-stream sync**: `event.wait()` on stream_b blocks stream_b until the event is recorded on stream_a. This is explicit dependency management.

---

## overlap.py — Simulate Overlapping Data Transfer and Computation

This script simulates a pipeline where data transfer and model computation overlap across two streams.

```python
import torch
import time


def simulate_data_transfer(next_batch_size, device='cuda'):
    """Simulate loading next batch to GPU (replace with real data loading)."""
    # In real code: next_batch = load_from_disk()
    # Here we simulate with a GPU allocate + copy
    dummy = torch.randn(next_batch_size, device=device)
    return dummy


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
        # Compute on current batch using compute_stream
        with torch.cuda.stream(compute_stream):
            result = current @ weight

        # Transfer next batch using transfer_stream
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
    More realistic pipeline: actual GPU computation matmul + relu + matmul.
    Overlap transfer of batch N+1 with compute of batch N.
    """
    device = 'cuda'
    compute_stream = torch.cuda.Stream()
    transfer_stream = torch.cuda.Stream()

    # Model weights (fixed)
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
```

### Expected Output

> **Expected on GTX 1070 Ti or similar. Your numbers will vary. Run `python overlap.py` on your GPU to measure.**

```
=== Data Transfer vs Compute Overlap ===

Sequential:  ~45.00 ms for 20 iterations
Overlapped:  ~30.00 ms for 20 iterations
Speedup:     ~1.5x

=== Realistic Pipeline ===

Pipeline (50 batches, 1024x256): ~12.00 ms total
Per batch (overlapped): 0.240 ms
```

The speedup depends on the relative cost of transfer vs compute. When transfer is cheap relative to compute, overlap helps less. When transfer time approaches or exceeds compute time, overlap approaches 2x speedup.

---

## Nsight Systems Profiling

To visualize multi-stream behavior in Nsight Systems:

```bash
nsys profile --output=streams_report \
    --trace=cuda,cupti \
    --cuda-memory-usage=true \
    python overlap.py
```

Key things to look for in the timeline:

1. **Stream 0 (default)** and **Stream N (custom)** bars interleaving — confirms concurrent execution.
2. **Data transfer bars** overlapping with **compute bars** — confirms the overlap is working.
3. **Gaps in stream bars** — indicate idle time or CPU-side stalls.

**What good looks like:** Stream 0 and custom stream bars are visibly overlapped in the timeline, with no gaps between compute and transfer. **What bad looks like:** bars are sequential (one finishes then the other starts), indicating no overlap.

---

## Sources

- https://pytorch.org/docs/stable/generated/torch.cuda.Stream.html
- https://pytorch.org/docs/stable/cuda.html#cuda-streams
- https://developer.nvidia.com/blog/how-cuda-streams-make-concurrent-gpu-execution-possible
- https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#streams
