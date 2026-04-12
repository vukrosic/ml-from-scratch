"""event_sync.py — measure event timing across streams."""

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

        start_event.record(stream=stream)
        result = x @ y
        end_event.record(stream=stream)
        stream.synchronize()

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

    # Measure on custom stream
    custom_avg = measure_kernel_time(x, y, stream=custom_stream)
    print(f"Custom stream:     {custom_avg:.3f} ms per matmul")

    # Two kernels in sequence on custom stream
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
