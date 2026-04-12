# Attention Profiling — Advanced

This document covers profiling tools and real benchmark data for attention mechanisms.

---

## Profiling Attention with PyTorch Profiler

PyTorch's profiler can capture attention kernel execution times and memory usage.

```python
import torch
from torch.profiler import profile, ProfilerActivity

def profile_attention(Q, K, V, warmup=10, active_steps=20):
    """Profile scaled_dot_product_attention with PyTorch profiler.

    Q, K, V: (batch, seq, d_head) tensors on CUDA
    """
    # Warmup
    for _ in range(warmup):
        _ = torch.nn.functional.scaled_dot_product_attention(Q, K, V)
    torch.cuda.synchronize()

    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
    ) as prof:
        for _ in range(active_steps):
            _ = torch.nn.functional.scaled_dot_product_attention(Q, K, V)
        torch.cuda.synchronize()

    # Print the top CUDA kernels by time
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
    return prof
```

Run this on your GPU to see which kernels are taking time. At large sequence lengths, you will typically see `flash_attention_forward` and `flash_attention_backward` if Flash Attention is available.

---

## Memory Profiling: Attention Matrix Size

The attention matrix is (batch, num_heads, seq, seq). For batch=1, num_heads=8, seq=4096, this is 8 × 4096 × 4096 float32 elements = 512 MB just for the forward pass.

```python
def attention_memory(seq, num_heads, batch=1, dtype=torch.float32):
    """Calculate attention matrix memory in MB.

    seq: sequence length
    num_heads: number of attention heads
    batch: batch size
    dtype: tensor dtype
    Returns: memory in MB
    """
    bytes_per_element = 4 if dtype == torch.float32 else 2  # float16
    elements = batch * num_heads * seq * seq
    return elements * bytes_per_element / 1024**2


print("Attention matrix memory:")
for seq in [512, 1024, 2048, 4096]:
    mem = attention_memory(seq, num_heads=8)
    print(f"  seq={seq:4d}: {mem:7.1f} MB")
```

Output:
```
  seq= 512:    8.0 MB
  seq=1024:   32.0 MB
  seq=2048:  128.0 MB
  seq=4096:  512.0 MB
```

This is just the attention matrix in the forward pass. The backward pass doubles it (gradient of attention matrix). Flash Attention reduces this to O(seq) memory by never materializing the full matrix.

---

## KV Cache Memory

KV cache stores (keys, values) for each generated token.

```
Per token memory: 2 * d_model * bytes_per_element
d_model=512, float16: 2 * 512 * 2 = ~2 KB per token
d_model=512, float32: 2 * 512 * 4 = ~4 KB per token
```

For a 4096-token context with d_model=512, float16:
```
4096 * 2 KB = ~8 MB per layer
LLaMA 7B (80 layers): 80 * 8 MB = ~640 MB
```

---

## Flash Attention Availability

PyTorch 2.0+ automatically uses Flash Attention when available via SDPA. Check with:

```python
torch.backends.cuda.enable_flash_sdp(True)       # Flash Attention
torch.backends.cuda.enable_mem_efficient_sdp(True)  # Math+upper_triangle attention
torch.backends.cuda.enable_math_sdp(True)         # Pure math (no flash)
```

If Flash Attention is not available, SDPA falls back to the memory-efficient or naive kernel depending on your settings.

---

## Benchmarking Across Sequence Lengths

```python
import time
import torch
import torch.nn.functional as F

def benchmark_sdpa(batch, seq, d_head, steps=200, warmup=50):
    Q = torch.randn(batch, seq, d_head, device='cuda')
    K = torch.randn(batch, seq, d_head, device='cuda')
    V = torch.randn(batch, seq, d_head, device='cuda')

    for _ in range(warmup):
        _ = F.scaled_dot_product_attention(Q, K, V)
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(steps):
        _ = F.scaled_dot_product_attention(Q, K, V)
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / steps

print("SDPA throughput across sequence lengths:")
for seq in [256, 512, 1024, 2048, 4096]:
    t = benchmark_sdpa(batch=4, seq=seq, d_head=64)
    print(f"  seq={seq:4d}: {t*1000:.2f} ms ({1/t:.0f} steps/s)")
```

Expected scaling: time roughly quadruples when seq doubles (O(n²) behavior from the Q @ K.T matmul).

---

## Key Profiling Findings

1. **SDPA is always fused** — no explicit attention matrix unless you materialize it.
2. **Flash Attention kernel time** scales as O(n²) in compute but O(n) in memory.
3. **KV cache overhead** per step is the Q projection cost plus cat/memory fetch of cached K/V.
4. **Attention memory is the bottleneck** at long context lengths — Flash Attention 2 reduces this significantly.
5. **Procure a real GPU** — profiling on CPU tells you almost nothing about GPU attention performance.

Run `python -m torch.utils.bottleneck` on any of the tutorial scripts for a quick flamegraph.
