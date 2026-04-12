"""streams.py — default stream vs custom stream benchmark."""

import torch
import time


def benchmark_stream_sequential(x, y, iterations=100):
    """Sequential matmuls on the default stream."""
    torch.cuda.synchronize()
    t0 = time.perf_counter()

    for _ in range(iterations):
        result = x @ y
        result = y @ x  # second matmul waits for first on default stream

    torch.cuda.synchronize()
    return time.perf_counter() - t0


def benchmark_stream_custom(x, y, stream, iterations=100):
    """Concurrent matmuls on a custom stream."""
    torch.cuda.synchronize()
    t0 = time.perf_counter()

    for _ in range(iterations):
        with torch.cuda.stream(stream):
            result = x @ y
            result = y @ x  # runs concurrently on custom stream

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


if __name__ == "__main__":
    main()
