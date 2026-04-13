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

**Timeline with annotated timing:**

```
Default stream timeline (all on Stream 0):
|............ matmul A ............|.5ms gap.|............ matmul B ............|
|<--------- 5 ms --------->|        |        |<--------- 5 ms --------->|
                               ↑                  ↑
                          GPU gap:            matmul B
                          GPU idle            cannot start until
                                             matmul A finishes

Total wall-clock time: 5ms + 0.5ms + 5ms = 10.5 ms
```

The 0.5ms gap is the hardware context-switch time as the GPU switches between kernels. Even though matmul B is completely independent of matmul A, it must wait.

### The Fix: Custom Streams Run Independently

A custom stream is a separate GPU work queue. Operations on different streams interleave freely — the GPU schedules them based on hardware availability.

```python
stream = torch.cuda.Stream()

with torch.cuda.stream(stream):
    result2 = y @ x    # runs on stream, does not wait for result1
```

**Same two matmuls on different streams with overlapping timeline:**

```
Stream 0 (default):  |............ matmul A ............|
Stream 1 (custom):   |..... matmul B (starts immediately) .....|

Timeline:
|<----------------- 5 ms total (overlapped) ----------------->|

matmul A:  ████████████████████████████████████████  (5 ms)
matmul B:  ████████████████████████████████  (5 ms)
           |--- starts at 0 ms (no waiting) ---|

Total wall-clock time: ~5 ms (vs 10.5 ms sequential)
Speedup: ~2x
```

Both kernels launch without waiting for each other. The GPU schedules them as hardware allows — the second matmul starts immediately even though the first one is still running.

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

**Step-by-step what `record()` and `wait()` do:**

```
Step 1: event.record() is called inside stream
        - Captures stream's current position counter
        - event.timestamp = stream_position_counter (e.g., 42)
        - event is now "attached" to stream at this position

        Stream (custom): |...... matmul ......| event_record@42 |
                                                      ^
                                                 event stored here

Step 2: event.wait() is called from default stream
        - default stream notes: "before doing my next op,
          wait until stream reaches position 42"
        - default stream does NOT block here, it continues

        Default stream: | previous ops | event_wait() | final = result + bias |
                                                      ↑
                                              inserts dependency:
                                              "don't proceed past here
                                              until event@42 is reached"

Step 3: GPU scheduler
        - custom stream executes matmul, reaches position 42
        - marks event as COMPLETED
        - default stream is now unblocked and can proceed

Result: final = result + bias only runs AFTER matmul on stream is done
        But default stream could run OTHER ops before reaching this point
```

This is how you synchronize across streams without blocking the whole GPU. The default stream can still do independent work — it only blocks at the specific `event.wait()` point.

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

### stream.synchronize() vs event.wait(): The Key Difference

These two primitives have fundamentally different effects on the CPU and GPU:

```
stream.synchronize() — BLOCKS THE CPU THREAD

    CPU thread:  |........ do stuff .......|**** blocked ****|.. continue ..|
                                       ↑
                              CPU waits here until
                              the ENTIRE stream finishes

    All streams continue running during this time — the GPU is not
    paused. Only this CPU thread is blocked, waiting for the stream
    to complete.

event.wait() — BLOCKS A SPECIFIC STREAM (not the CPU)

    Default stream:  |.. ops ..| event_wait() | subsequent ops |
                                          ↑
                                  This stream pauses here.
                                  Other ops on this stream that come
                                  AFTER event_wait() must wait.

    Custom stream:    |.. matmul ..| [event recorded here]

    Key difference: The CPU thread continues immediately after calling
    event.wait(). Only the default stream's execution is deferred at
    the event.wait() point. Other CPU work can proceed.

Summary:
    stream.synchronize()  → CPU thread waits. All GPU streams keep going.
    event.wait()          → One specific stream waits. CPU is not blocked.
```

### Events — Measure Time and Sync Points

CUDA events record a point in a stream. You can measure elapsed time between two events, or use them as explicit synchronization points.

```python
# Create two events: start and end markers
start_event = torch.cuda.Event(enable_timing=True)
end_event = torch.cuda.Event(enable_timing=True)

# Record start event in the stream (captures current stream position)
start_event.record()

with torch.cuda.stream(stream):
    result = x @ y

# Record end event after the kernel
end_event.record()

# Block CPU until the kernel completes so we get accurate timing
stream.synchronize()

# elapsed_time uses GPU hardware timers — accurate GPU time, not CPU time
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

**Timeline with annotated timing:**

```
Sequential (naive) — NO overlap:
Time (ms):    |0........10........20........30........40|
Transfer:     |--- transfer --|-- gap --|-- transfer --|
Compute:                           |--- compute ---|
              ↑ total: ~40 ms (transfer + compute sequential)

Overlapped (two streams):
Time (ms):    |0........10........20........30|
Compute:      |████ matmul+relu+matmul ████|
Transfer:              |--- next batch transfer -->|
                                   ↑         ↑
                               transfer   compute
                               overlaps   on next
                               compute   batch starts
                               here      here

              |<---- ~25 ms total ---->|
              Savings: ~15 ms (37.5% reduction)
```

The key insight: while the GPU is running matmul+relu+matmul on batch N (compute stream), the transfer stream is simultaneously moving batch N+1 from CPU to GPU. The CPU-side preparation of batch N+1 also happens concurrently with GPU compute. The compute and transfer streams run concurrently. The CPU prepares the next batch while the GPU processes the current one.

### The Template

```python
def overlapped_pipeline(data_loader, model):
    # Dedicated stream for data transfer (separate from compute)
    transfer_stream = torch.cuda.Stream()

    # Pre-transfer first batch so compute stream has immediate work
    batch = next(iter(data_loader))
    input_batch = batch['input'].to('cuda')

    for batch in data_loader:
        # New compute stream each iteration — avoids stream ordering issues
        compute_stream = torch.cuda.Stream()

        # Launch compute on current batch (runs on compute_stream)
        with torch.cuda.stream(compute_stream):
            result = model(input_batch)

        # Launch transfer of next batch (runs on transfer_stream concurrently)
        with torch.cuda.stream(transfer_stream):
            next_input = batch['input'].to('cuda')

        # Wait for compute to finish before yielding result
        result.synchronize()
        yield result

        # Advance: next batch becomes current batch
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

> **Key rule: CUDA operations are asynchronous. Always call `torch.cuda.synchronize()` before timing or the measurement includes Python overhead, not GPU time.**

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

The full runnable version of this example is in `streams.py`. Run it with `python streams.py`.

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

## Which File Demonstrates What

| File | What It Demonstrates |
|------|----------------------|
| `streams.py` | Default stream vs custom stream benchmark — sequential vs concurrent matmuls, profiler output |
| `event_sync.py` | CUDA event timing, cross-stream synchronization with `event.wait()` |
| `overlap.py` | Overlapping data transfer and computation across two streams, realistic pipeline |

---

## Going Further

For CUDA event timing comparisons across different stream configurations, benchmark scripts for PCIe transfer vs compute overlap, Nsight Systems profiling examples for multi-stream workloads, and the full event synchronization timing data — see [ADVANCED.md](./ADVANCED.md).

Sources:
- https://pytorch.org/docs/stable/generated/torch.cuda.Stream.html
- https://pytorch.org/docs/stable/cuda.html#cuda-streams
- https://developer.nvidia.com/blog/how-cuda-streams-make-concurrent-gpu-execution-possible

---

Get the video walkthrough of multi-stream benchmarks, Nsight Systems profiling, and PCIe transfer overlap: [OpenSuperintelligenceLab on Skool](https://www.skool.com/opensuperintelligencelab)
