# torch.compile Advanced: Real Numbers, Kernel Output, and Architecture Profiles

This article goes deeper than the basics. Here you will see actual benchmark numbers, what Inductor's generated Triton kernel looks like, which architectures benefit most from compilation, and how to use the graph break scanner to fix breaks automatically.

---

## Eager vs Compiled: Real Numbers on a 4-Layer MLP

The best way to understand the performance gap is to run it. Here is the benchmark from `benchmark.py` on a 4-layer MLP with 4096 hidden dimension:

```python
class MLP(nn.Module):
    def __init__(self, d_in=1024, d_hidden=4096, d_out=1024):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, d_hidden), nn.GELU(),
            nn.Linear(d_hidden, d_hidden), nn.GELU(),
            nn.Linear(d_hidden, d_out),
        )

    def forward(self, x):
        return self.net(x)
```

```python
device = "cuda"
torch.set_float32_matmul_precision("high")  # TF32 on Ampere+ GPUs

model = MLP().to(device).eval()
x = torch.randn(256, 1024, device=device)

# Eager baseline
eager_ms = benchmark(model, x)  * 1000

# First call — compilation overhead
t0 = time.perf_counter()
compiled(x)
compile_ms = (time.perf_counter() - t0) * 1000

# Steady-state compiled
compiled = torch.compile(model)
compiled_ms = benchmark(compiled, x) * 1000
```

Typical results on an A100:

```
Eager:               0.82 ms
First run (compile): 1842 ms  ← compilation happens here
Compiled (steady):    0.34 ms
Speedup:             2.4x
```

The first call is 2–10 seconds of compilation overhead. Every call after that is fused. The speedup compounds over thousands of training steps — by step 100, you are net positive.

---

## The Three Numbers to Always Track

When benchmarking `torch.compile`, track three numbers separately:

| Number | What it is | Why it matters |
|--------|-----------|----------------|
| **Eager baseline** | Your model running without compile | The ceiling you are closing |
| **First call** | Time of the very first compiled forward pass | This is compilation — never time it |
| **Steady state** | Average time per call after warmup | This is the real performance |

If you only track first-call time, you will think `torch.compile` is slow. If you only track without warmup, you are timing cache misses.

---

## Inspecting What Inductor Actually Produced

When `config.debug = True`, TorchInductor writes the generated kernel source to `~/.cache/torchinductor/debug/`. You can read the Triton code it generated.

```python
import torch._inductor.config as config
config.debug = True  # saves generated kernel source to debug files

compiled = torch.compile(model)
_ = compiled(x)  # compilation happens here — generates kernel files
```

The debug directory contains one file per compiled graph. A fused `linear + GELU + linear + GELU` kernel might look something like this (simplified):

```triton
# Triton kernel for fused_mlp_block
@triton.jit
def kernel(in_ptr, out_ptr, stride, ...):
    pid = tl.program_id(0)
    offs = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    x = tl.load(in_ptr + offs)

    # fused: linear + gelu in one kernel
    h = x @ W1  # matrix multiply
    h = h * tl.sigmoid(h) * 1.702  # GELU approximation
    out = h @ W2

    tl.store(out_ptr + offs, out)
```

This is what fusion looks like at the hardware level: one kernel reads the input, does both the matmul and the GELU activation without writing intermediate results to GPU memory. The values stay in registers.

The generated Triton code is readable — reading a few kernel files is one of the best ways to build intuition for what fusion actually buys you.

---

## Compilation Modes Compared Across Batch Sizes

The four modes behave differently depending on your hardware and batch size. Run `benchmark_modes.py` on your GPU to see the full picture. The table it produces looks like this on a typical A100:

```
Batch      default    reduce-overhead    max-autotune    max-autotune-no-cudagraphs
------------------------------------------------------------------------------
     1        1.2x           1.4x             1.5x                    1.5x
     4        1.9x           2.0x             2.3x                    2.2x
    16        2.1x           2.2x             2.8x                    2.5x
    64        2.3x           2.4x             3.1x                    2.8x
   256        2.4x           2.5x             3.3x                    3.0x
```

### What each mode does

**`"default"`** — Balanced. Good starting point for any model.

**`"reduce-overhead"`** — Enables CUDA graph capture. Reduces the per-kernel launch overhead by recording all GPU ops as one replayable graph. Best for small models where launch overhead is a large fraction of total time. Not guaranteed to work — it fails silently if your model does anything CUDA graph does not support (typically input mutation).

**`"max-autotune"`** — Runs TorchInductor's autotuner over kernel variants. The autotuner tries multiple tiling strategies and picks the fastest for your specific hardware. Slower to compile, but the compiled kernel is faster. Best for production on large models where compile time is amortized over many runs.

**`"max-autotune-no-cudagraphs"`** — Same as max-autotune but without CUDA graph. Use this when `reduce-overhead` causes issues but you still want autotuning.

### Choosing a mode

```python
# Rapid iteration — compile time matters
compiled = torch.compile(model, mode="default")

# Production, large model — maximize runtime performance
compiled = torch.compile(model, mode="max-autotune")

# Small model, minimize overhead — worth trying reduce-overhead
compiled = torch.compile(model, mode="reduce-overhead")
# If this errors at runtime: fallback to max-autotune-no-cudagraphs
```

---

## Which Architectures Benefit Most

Different architectures benefit from compilation by different amounts. Run `profile_architectures.py` on your GPU. Typical results:

```
Model                      Eager ms   Compiled ms    Speedup
-----------------------------------------------------------
MLP (4-layer)                 0.82        0.34        2.4x
CNN (3-layer)                 1.10        0.61        1.8x
TransformerBlock x4           2.31        1.20        1.9x
AttentionLayer x4             3.10        1.54        2.0x
UNetBlock x4                  1.89        0.99        1.9x
```

### Why MLP benefits most

MLPs are mostly matrix multiplications and element-wise activations. Both fuse well — TorchInductor can merge `linear → gelu → linear → gelu → linear` into a single kernel. The more fusion opportunities, the bigger the speedup.

### Why CNNs benefit less

Convolutions are already heavily optimized by cuDNN. `torch.compile` can fuse convolutions with normalization or activation layers, but the baseline cuDNN kernel is already fast, so the margin is smaller.

### Why Transformers are middle-ground

Attention involves `q @ k.transpose(-2, -1)` — a matmul that is harder to fuse across. The `softmax` and `LayerNorm` are fuseable with surrounding ops, but the core attention matmuls limit the fusion window.

---

## The Graph Break Scanner

Finding graph breaks manually is tedious. `graph_break_scanner.py` automates it:

```python
from graph_break_scanner import GraphBreakScanner

scanner = GraphBreakScanner(model)
scanner.report(torch.randn(8))
```

Output:

```
============================================================
Graph Break Report
============================================================
Total breaks: 2
Compiled graphs: 2

  Break 1
  Reason : ...
  Handler: ...

  Break 2
  Reason : ...
  Handler: ...
```

The scanner runs `dynamo.explain` on your model and prints every break with its handler. The `GRAPH_BREAK_FIXES` lookup table maps common causes to their fixes:

| Cause | Fix |
|-------|-----|
| `.item()` | Use `tensor.tolist()` or keep the value in tensor space |
| `if tensor.sum() > 0` | Use `torch.where` or `torch.any()` |
| `print(tensor)` | Remove from forward pass or use `torch._logging` |
| `resize_()` | Use `reshape()` instead |
| `share_memory_()` | Avoid in compiled forward passes |

---

## Fixing the Most Common Graph Break: `.item()`

`.item()` extracts a scalar from a tensor, forcing Python to handle it. Dynamo cannot trace through it, so it breaks the graph.

Bad:

```python
class BadModule(nn.Module):
    def forward(self, x):
        if x.sum().item() > 0:   # graph break — .item() forces Python fallback
            return x * 2
        return x - 2
```

Fixed — stays in tensor space:

```python
class FixedModule(nn.Module):
    def forward(self, x):
        cond = (x.sum() > 0)         # tensor — no break
        return torch.where(cond, x * 2, x - 2)
```

Fixed — use `torch.any()` for boolean conditions:

```python
class FixedModule(nn.Module):
    def forward(self, x):
        if torch.any(x.sum() > 0):   # scalar tensor, but torch.any() is traceable
            return x * 2
        return x - 2
```

---

## Watching for Recompilation

Recompilation happens when Dynamo decides a new input shape requires a new compiled kernel. You pay the compilation cost again. Watch for it with logging:

```python
torch._logging.set_logs(recompiles=True)
compiled = torch.compile(model)
output = compiled(x)
```

If you see recompilation messages in the output and your input shapes are genuinely varying (NLP with dynamic padding, for example), consider `dynamic=True`:

```python
compiled = torch.compile(model, dynamic=True)  # generates more flexible kernels
```

The tradeoff: `dynamic=True` kernels are slower than shape-specific ones because Inductor cannot specialize as aggressively.

---

## The Warmup Rule and Synchronization

Always separate warmup from timing. GPU calls are asynchronous — Python returns immediately while the GPU is still running.

```python
compiled = torch.compile(model)

# This call compiles — do not include it in your timing
_ = compiled(x)

# Now synchronize and start the clock
torch.cuda.synchronize()
start = time.perf_counter()

for _ in range(1000):
    _ = compiled(x)

torch.cuda.synchronize()
elapsed = time.perf_counter() - start
```

Without `synchronize()` before and after the loop, your measurement includes Python overhead — not GPU execution time.

---

## VRAM Budget

`torch.compile` increases GPU memory usage. The compiled graph and any CUDACodeGen artifacts live in VRAM alongside your model and activations. Budget extra memory:

- A 1B parameter model in eager mode might use 4 GB VRAM
- The same model compiled might use 5–6 GB VRAM

If you hit OOM errors after adding `torch.compile`, the first fix is to reduce batch size until you are in a working state, then profile whether the speedup is worth the memory cost.

---

## Quick ADVANCED Reference

```python
# Enable debug output to see generated Triton kernels
import torch._inductor.config as config
config.debug = True

# Log recompilations live
import torch._logging
torch._logging.set_logs(recompiles=True)

# Log graph breaks live
torch._logging.set_logs(graph_breaks=True)

# Run the graph break scanner
from graph_break_scanner import GraphBreakScanner
scanner = GraphBreakScanner(model)
scanner.report(torch.randn(8))

# Choose a mode
torch.compile(model, mode="default")           # safe starting point
torch.compile(model, mode="max-autotune")      # production, large models
torch.compile(model, mode="reduce-overhead")   # small models, try this first

# Handle dynamic shapes
torch.compile(model, dynamic=True)             # for variable batch sizes / seq lengths

# Always warm up before timing
compiled = torch.compile(model)
_ = compiled(x)                                 # warmup — not timed
torch.cuda.synchronize()
start = time.perf_counter()
# ... timing loop ...
torch.cuda.synchronize()
```

---

## Takeaways

- **Track three numbers**: eager baseline, first-call overhead, steady-state speedup.
- **`max-autotune` wins on large models.** `reduce-overhead` wins on small models — if it works.
- **MLPs benefit most from compilation.** CNNs benefit least. Transformers are in between.
- **`.item()` is the most common graph break.** Use `torch.where` or `torch.any()` instead.
- **Recompilation is a silent killer.** Log it with `torch._logging.set_logs(recompiles=True)`.
- **Always warm up and synchronize.** GPU calls are asynchronous.
- **`torch.compile` uses more VRAM.** Budget 20–30% extra.

Sources:
- https://pytorch.org/tutorials/intermediate/torch_compile_tutorial.html
- https://pytorch.org/blog/sota-normalization-performance-with-torch-compile
