# CUDA Streams From Scratch

Every CUDA kernel you launch goes onto a stream. By default, everything runs on the default stream — and that stream blocks. Custom streams let you run independent operations in parallel, overlap data transfers with computation, and squeeze more utilization out of your GPU. This is what CUDA streams actually do.

---

## What Problem Streams Solve

### The Problem: The Default Stream Blocks

When you run this:

```python
x = torch.randn(1024, 1024, device='cuda')
y = torch.randn(1024, 1024, device='cuda')

# All on the default stream — these run sequentially
result1 = x @ y      # waits for previous ops to finish
result2 = y @ x      # waits for result1 to finish
```

The GPU executes each operation in order on a single stream. Operation 2 cannot start until operation 1 completes — even if the GPU has free resources that could handle both.

```
Default stream:
| matmul | waited | matmul |
|<- A ->|  gap    |<- B  ->|
```

### The Fix: Custom Streams Run Independently

A custom stream is a separate GPU work queue. Operations on different streams interleave freely — the GPU schedules them based on hardware availability.

```python
stream = torch.cuda.Stream()

with torch.cuda.stream(stream):
    result2 = y @ x    # runs on stream, does not wait for result1
```

```
Stream 0 (default):  | matmul A |
Stream 1 (custom):   | matmul B |
                      ↑ runs concurrently, no waiting
```

Both kernels launch without waiting for each other. The GPU schedules them as hardware allows.

---

## How Streams Work

### The Default Stream

Every GPU has one default stream that all PyTorch operations use if you do not specify otherwise. It is synchronous by default — each operation waits for the previous one to finish.

### Creating a Custom Stream

```python
# Create a new stream with default priority
stream = torch.cuda.Stream()

# Create a stream with a specific priority
# Lower number = higher priority
stream = torch.cuda.Stream(priority=0)
```

### The stream() Context Manager

The `torch.cuda.stream()` context manager routes all CUDA operations inside the block to the specified stream.

```python
stream = torch.cuda.Stream()

with torch.cuda.stream(stream):
    # All operations here run on `stream`
    a = x @ y
    b = y @ z
    c = a @ b
```

Outside the block, operations return to the default stream.

---

## Stream Priorities and CUDA Work Queues

### How Priority Interacts with the GPU Scheduler

CUDA assigns each stream a priority when you create it. The GPU scheduler prefers higher-priority (lower-numbered) streams when scheduling work.

```
Priority 0 (high):   |--- kernel A ---|
Priority 1 (medium):|-- kernel B --|    |--- kernel C ---|
Priority 2 (low):    |------ kernel D ------|
```

High-priority streams do not preempt running kernels, but they get dispatched first when the GPU is free.

### Check Stream Priority

```python
# Get the priority range for this device
low, high = torch.cuda.Stream.priorities()
print(f"Priority range: {high} (highest) to {low} (lowest)")
```

Typical output on NVIDIA GPUs:

```
Priority range: 0 (highest) to -2 (lowest)
```

### When Priority Matters

Priority becomes relevant when multiple streams have work queued. If you have a latency-sensitive operation (say, a small critical section) and a throughput operation running simultaneously, giving the critical stream higher priority ensures it gets scheduled first.

---

## Synchronization Primitives

### wait() — Make One Stream Wait for Another

```python
stream = torch.cuda.Stream()
event = torch.cuda.Event()

with torch.cuda.stream(stream):
    result = x @ y

# Insert event into stream
event.record()

# Default stream waits for the event
event.wait()

# Default stream continues after event is recorded
final = result + bias
```

This is how you synchronize across streams without blocking the whole GPU.

### synchronize() — Block Until a Stream Finishes

```python
stream = torch.cuda.Stream()

with torch.cuda.stream(stream):
    result = x @ y

# This blocks the calling CPU thread until stream completes
stream.synchronize()

# Safe to use `result` now — stream is done
print(result.sum())
```

### Events — Measure Time and Sync Points

CUDA events record a point in a stream. You can measure elapsed time between two events, or use them as explicit synchronization points.

```python
start_event = torch.cuda.Event(enable_timing=True)
end_event = torch.cuda.Event(enable_timing=True)

start_event.record()

with torch.cuda.stream(stream):
    result = x @ y

end_event.record()
stream.synchronize()

elapsed_ms = start_event.elapsed_time(end_event)
print(f"Kernel took {elapsed_ms:.2f} ms")
```

---

## Practical Use Case: Overlapping Data Transfer and Computation

The classic use case for streams is hiding data transfer time by overlapping it with computation.

### The Naive Sequential Version

```python
# Sequential: transfer, then compute, then transfer
data = load_data_from_disk()           # slow CPU-side read
data = data.to('cuda')                 # GPU transfer — GPU idle
result = model(data)                   # compute — GPU busy, PCIe idle
output = result.to('cpu')              # transfer back — GPU idle
```

The GPU sits idle during data transfer. The PCIe bus sits idle during computation.

### The Overlapped Version

```python
# Transfer part 1 to GPU
input_tensor = data.to('cuda')

# While GPU computes on current batch, transfer next batch
compute_stream = torch.cuda.Stream()
transfer_stream = torch.cuda.Stream()

with torch.cuda.stream(compute_stream):
    result = model(input_tensor)

# Transfer next batch while result processes on CPU
with torch.cuda.stream(transfer_stream):
    next_data = load_next_batch()
    next_input = next_data.to('cuda')

# Synchronize before using next_input
next_input.synchronize()
```

```
Time:        |--- compute on batch N ---|--- compute on batch N+1 ---|
Compute:     |████ model forward ████|
Transfer:              |--- load next batch -->|--- transfer to GPU -->|
             |<-- batch N idle -->|
```

The compute and transfer streams run concurrently. The CPU prepares the next batch while the GPU processes the current one.

### The Template

```python
def overlapped_pipeline(data_loader, model):
    transfer_stream = torch.cuda.Stream()

    # Pre-transfer first batch
    batch = next(iter(data_loader))
    input_batch = batch['input'].to('cuda')

    for batch in data_loader:
        compute_stream = torch.cuda.Stream()

        # Compute on current batch
        with torch.cuda.stream(compute_stream):
            result = model(input_batch)

        # Transfer next batch on transfer stream
        with torch.cuda.stream(transfer_stream):
            next_input = batch['input'].to('cuda')

        # Synchronize before using next batch
        result.synchronize()
        yield result

        # Advance
        input_batch = next_input
```

---

## How Async Execution Affects Profiling

### The Core Problem: Async by Default

CUDA operations return control to Python immediately. The GPU is still working. If you measure time with `time.perf_counter()` without synchronizing, you measure Python overhead — not GPU time.

```python
start = time.perf_counter()
x = x @ y          # launches kernel, returns immediately
end = time.perf_counter()
print(end - start)  # ~0.0001 ms — Python overhead, not GPU time
```

### The Fix: Synchronize Before and After

```python
torch.cuda.synchronize()   # ensure GPU is idle before we start
start = time.perf_counter()

x = x @ y                  # launches kernel
torch.cuda.synchronize()   # wait for kernel to finish

end = time.perf_counter()
print(end - start)        # actual GPU time
```

### Profiling with Events

The most accurate way to measure GPU kernel time:

```python
start_event = torch.cuda.Event(enable_timing=True)
end_event = torch.cuda.Event(enable_timing=True)

start_event.record()
x = x @ y
end_event.record()
torch.cuda.synchronize()

elapsed = start_event.elapsed_time(end_event)
```

Events use GPU hardware timers — they measure what the GPU actually experienced, not CPU time.

### Profiling Custom Streams

When using multiple streams, each stream's work is async relative to the CPU and to other streams. Use `torch.cuda.synchronize()` to get a consistent baseline:

```python
torch.cuda.synchronize()   # reset: all streams idle

stream = torch.cuda.Stream()
with torch.cuda.stream(stream):
    x = x @ y

stream.synchronize()       # wait for stream to finish
```

---

## Minimal Stream Example

### Piece 1: Default stream behavior

```python
import torch

x = torch.randn(4096, 4096, device='cuda')
y = torch.randn(4096, 4096, device='cuda')

# All on default stream — sequential
result1 = x @ y
result2 = y @ x
result3 = x @ y
```

### Piece 2: Custom stream for independent operation

```python
stream = torch.cuda.Stream()

with torch.cuda.stream(stream):
    # This runs on `stream` — does not block default stream
    result2 = y @ x

# Default stream continues immediately
# result2 is not ready yet — it is still running on stream
final = result1 + result2  # would need synchronize first
```

### Piece 3: Proper synchronization

```python
stream = torch.cuda.Stream()

with torch.cuda.stream(stream):
    result2 = y @ x

# Wait for stream to finish before using result2
stream.synchronize()

# Now safe to use result2
final = result1 + result2
```

### Piece 4: Overlapping data transfer

```python
transfer_stream = torch.cuda.Stream()
compute_stream = torch.cuda.Stream()

# Start transfer on transfer stream
data = torch.randn(1024, 1024, device='cuda')
with torch.cuda.stream(transfer_stream):
    next_data = torch.randn(1024, 1024, device='cuda')

# Compute on compute stream while transfer runs
with torch.cuda.stream(compute_stream):
    result = data @ data

# Wait for compute to finish
compute_stream.synchronize()
```

---

## Recap

- **Default stream blocks.** Every operation waits for the previous one to finish.
- **Custom streams run independently.** Operations on different streams interleave freely.
- **Use `torch.cuda.stream(stream)`** context manager to route ops to a stream.
- **Stream priorities** affect scheduling order when multiple streams have queued work.
- **Synchronize before measuring.** GPU calls are asynchronous — always call `torch.cuda.synchronize()` before timing.
- **Events measure GPU time accurately.** Use `torch.cuda.Event(enable_timing=True)` for precise profiling.
- **Overlap transfers and compute** by running them on different streams simultaneously.
- **The CPU never waits for the GPU.** GPU operations return immediately — synchronization is explicit.

---

## Going Further

For CUDA event timing comparisons across different stream configurations, benchmark scripts for PCIe transfer vs compute overlap, Nsight Systems profiling examples for multi-stream workloads, and the full event synchronization timing data — see [ADVANCED.md](./ADVANCED.md).

Sources:
- https://pytorch.org/docs/stable/generated/torch.cuda.Stream.html
- https://pytorch.org/docs/stable/cuda.html#cuda-streams
- https://developer.nvidia.com/blog/how-cuda-streams-make-concurrent-gpu-execution-possible
