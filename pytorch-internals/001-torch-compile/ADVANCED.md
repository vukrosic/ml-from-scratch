# torch.compile Advanced

> **Before reading this, read [README.md](./README.md) first.** This article assumes you understand the three-stage pipeline (TorchDynamo, AOT Autograd, TorchInductor), what graph breaks are, and how `torch.compile(model)` works. ADVANCED goes deeper — benchmarks, kernel output, architecture profiles, and the graph break scanner.
>
> **All numbers in this article are placeholders.** Run the Python files in this directory on your GPU to get your actual numbers.

---

## Which File to Run For What

Each Python file in this directory demonstrates one concept:

| File | What it demonstrates |
|------|---------------------|
| `benchmark.py` | Eager vs compiled on a 4-layer MLP — produces the three numbers |
| `benchmark_modes.py` | All four modes across five batch sizes |
| `profile_architectures.py` | Five architectures benchmarked: MLP, CNN, Transformer, Attention, UNet |
| `graph_breaks.py` | BadModule vs FixedModule — shows the `.item()` break and its fix |
| `graph_break_scanner.py` | Automated scanner that finds breaks and suggests fixes |
| `tracer.py` | Toy tracer — builds a graph from operations |
| `fusion.py` | Toy fusion pass — merges adjacent matmul+relu |
| `codegen.py` | Generates Python from a graph |

---

## Eager vs Compiled: Real Numbers on a 4-Layer MLP

The best way to understand the performance gap is to run it yourself. Run `benchmark.py` on your GPU.

Here is the model. It is a 4-layer MLP with 4096 hidden dimension — a typical transformer building block:

```python
# 4-layer MLP: 1024 -> 4096 -> 4096 -> 4096 -> 1024
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

Move it to GPU and create a random input:

```python
device = "cuda"
torch.set_float32_matmul_precision("high")  # enables TF32 on Ampere+ GPUs

model = MLP().to(device).eval()   # eval mode — no gradients needed
x = torch.randn(256, 1024, device=device)  # batch=256, features=1024
```

`torch.set_float32_matmul_precision("high")` enables TF32 matmuls on Ampere and newer GPUs. This gives you the speed of FP16 with FP32 accumulation — a fair comparison between eager and compiled.

Run `benchmark.py` on your GPU. You will see three numbers:

```
Eager:               ~X ms     ← your model running without compile
First run (compile): ~Y ms     ← the very first call (compilation happens here)
Compiled (steady):   ~Z ms     ← every call after warmup
Speedup:             ~Nx       ← compiled / eager
```

The first call is always slow — that is TorchDynamo tracing, AOT Autograd capturing backward, and TorchInductor compiling the Triton kernel. Calls 2 onward are fused and fast. The speedup compounds over thousands of training steps.

Here is the benchmark function that produces those three numbers. This is what `benchmark.py` runs:

```python
def benchmark(fn, x, steps=200, warmup=50):
    # Warmup runs the model without timing
    # This establishes the cached compiled graph
    for _ in range(warmup):
        fn(x)
    torch.cuda.synchronize()

    # Now time the steady-state calls
    t0 = time.perf_counter()
    for _ in range(steps):
        fn(x)
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / steps
```

What that looks like on a timeline:

```
Eager mode:
|---call---||---call---||---call---|    each call takes ~X ms

torch.compile mode:
|____compilation____|  run   run   run   ← first call: ~Y ms (compile), then ~Z ms each
```

The compilation cost is paid once. After that, every call is fast.

---

## The Three Numbers to Always Track

When benchmarking `torch.compile`, track three numbers separately:

| Number | What it is | Why it matters |
|--------|-----------|----------------|
| **Eager baseline** | Your model running without compile | The ceiling you are closing |
| **First call** | Time of the very first compiled forward pass | This is compilation — never time it |
| **Steady state** | Average time per call after warmup | This is the real performance |

This is what the timeline looks like:

```
| compile | warmup | warmup | warmup | [====timed loop====] |
         ^        ^       ^                 ^
         |        |       |                 |
       ignore   ignore  ignore             time this
```

Never include the first call (compilation) or the warmup calls in your timing. If you do, your measurement will be wildly wrong.

---

## Inspecting What Inductor Actually Produced

TorchInductor can write the generated kernel source to disk. Enable this with `config.debug = True`:

```python
import torch._inductor.config as config
config.debug = True  # saves generated kernel source to debug files

compiled = torch.compile(model)
_ = compiled(x)  # compilation happens here — generates kernel files
```

After running, look in the debug directory:

```
~/.cache/torchinductor/debug/
├── c0_0000.cu          ← compiled CUDA source for graph 0
├── c0_0000.triton.py   ← generated Triton kernel source for graph 0
├── c0_0001.cu          ← compiled CUDA source for graph 1
└── c0_0001.triton.py   ← generated Triton kernel source for graph 1
```

Each compiled graph gets its own pair of files. Open a `.triton.py` file to see what Inductor generated.

Here is what a fused MLP kernel looks like — this is a simplified version of what you will see in your file. Real Triton kernels include additional tiling, masking, and configuration code, but the core idea is the same: all operations fused into one kernel with no memory round-trips:

```triton
@triton.jit
def kernel(in_ptr, out_ptr, W1, W2, ...):
    pid = tl.program_id(0)
    offs = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    x = tl.load(in_ptr + offs)

    # All four ops fused into one kernel:
    h = x @ W1                          # matmul 1
    h = h * tl.sigmoid(h) * 1.702       # GELU 1
    h = h @ W2                          # matmul 2
    h = h * tl.sigmoid(h) * 1.702       # GELU 2

    tl.store(out_ptr + offs, h)
```

Without fusion, this would be four separate kernel launches. The fused version keeps values in registers between ops — no memory round-trips. Reading a few of these files is the fastest way to understand what fusion actually buys you.

---

## Compilation Modes Compared Across Batch Sizes

`torch.compile` has four modes. They behave differently depending on your hardware and batch size. Run `benchmark_modes.py` on your GPU to see your numbers.

The table looks like this. Each cell is speedup over eager (higher is better):

```
Batch      default    reduce-overhead    max-autotune    max-autotune-no-cudagraphs
------------------------------------------------------------------------------
     1        ~Nx          ~Nx              ~Nx                  ~Nx
     4        ~Nx          ~Nx              ~Nx                  ~Nx
    16       ~Nx          ~Nx              ~Nx                  ~Nx
    64       ~Nx          ~Nx              ~Nx                  ~Nx
   256       ~Nx          ~Nx              ~Nx                  ~Nx
```

The pattern to watch for: batch size 1 barely benefits. The kernel is so small that GPU launch overhead dominates. Larger batches show the real gains.

Here is what each mode does:

- `"default"` — Balanced. Good starting point for any model.
- `"reduce-overhead"` — Enables CUDA graph capture. Records all GPU ops as one replayable unit. Best for small models. Not guaranteed to work (fails on input mutation or custom ops).
- `"max-autotune"` — Runs Inductor's autotuner over tiling strategies. Slower to compile, faster to run. Best for production on large models.
- `"max-autotune-no-cudagraphs"` — Same as max-autotune but without CUDA graph. Use this when reduce-overhead fails.

Choosing a mode — start with this:

```python
# Rapid iteration — compile time matters
compiled = torch.compile(model, mode="default")

# Production, large model — maximize runtime performance
compiled = torch.compile(model, mode="max-autotune")

# Small model — try reduce-overhead first
compiled = torch.compile(model, mode="reduce-overhead")
# If it errors at runtime (CUDA graph incompatibility):
compiled = torch.compile(model, mode="max-autotune-no-cudagraphs")
```

---

## Which Architectures Benefit Most

Different architectures benefit from compilation by different amounts. Run `profile_architectures.py` on your GPU.

The table looks like this. Each row is a different architecture type:

```
Model                      Eager ms   Compiled ms    Speedup
-----------------------------------------------------------
MLP (4-layer)                 ~X ms       ~Y ms        ~Nx
CNN (3-layer)                 ~X ms       ~Y ms        ~Nx
TransformerBlock x4            ~X ms       ~Y ms        ~Nx
AttentionLayer x4             ~X ms       ~Y ms        ~Nx
UNetBlock x4                  ~X ms       ~Y ms        ~Nx
```

As a bar chart (speedup, higher is better):

```
MLP (4-layer)         ████████████████████████  ~Nx
CNN (3-layer)         ██████████████            ~Nx
TransformerBlock x4   ██████████████            ~Nx
AttentionLayer x4     ██████████████            ~Nx
UNetBlock x4          ██████████████            ~Nx
                      |-------|---|----|
                     1.0x    2.0x   3.0x
```

Here is why different architectures get different speedups:

**MLP benefits most.** MLPs are matrix multiplications and element-wise activations. Both fuse extremely well:

```
MLP fusion:
Before:  [linear] → [GELU] → [linear] → [GELU] → [linear]
After:   [======== one fused kernel =========]

Each → is a separate kernel launch without fusion.
The fused kernel does all four ops in one trip.
```

**CNN benefits less.** Convolutions are already heavily optimized by cuDNN. `torch.compile` can fuse across the boundary (conv → norm → activation), but the baseline is already fast:

```
CNN fusion (limited):
Before:  [Conv2d] → [GroupNorm] → [SiLU] → [Conv2d] → ...
After:   [=== fused: Conv+Norm+Act ===] → [=== fused: Conv+Norm+Act ===]
         ← cuDNN handles Conv internals →
```

**Transformers are middle-ground.** Attention involves `q @ k.transpose(-2, -1)` — a matmul that is harder to fuse across. The softmax and LayerNorm fuse, but the core attention matmuls limit the window:

```
Transformer fusion:
Before:  [LayerNorm] → [q @ k.transpose] → [softmax] → [scale] → [attn @ v] → [proj]
After:   [===== LayerNorm + QKT + Softmax + Scale =====]  |  [attn @ v + proj]
         ←          mostly fused               ←]        |  ← partially fused
```

---

## The Graph Break Scanner

Finding graph breaks manually is tedious. `graph_break_scanner.py` automates it.

Use it on any model. Pass the model and a sample input:

```python
from graph_break_scanner import GraphBreakScanner

model = BuggyModel()  # a model with .item() and print() in forward
scanner = GraphBreakScanner(model)
scanner.report(torch.randn(8))
```

The report shows every break:

```
============================================================
Graph Break Report
============================================================
Total breaks : 2
Compiled graphs: 2
============================================================

  Break 1
  Reason : ...
  Handler: ...

  Break 2
  Reason : ...
  Handler: ...
```

`Total breaks` is how many breaks were found. `Compiled graphs` is how many separate compiled graphs Dynamo produced — ideally this is 1. Each break lists its reason (what Dynamo hit) and handler (what Dynamo did about it — typically a Python fallback).

You can also look up a fix directly from the reason string that appears in the report. Here is how to use `suggest_fix()` with a reason from the break report:

```python
from graph_break_scanner import suggest_fix

# The reason field comes from the scanner report
# e.g. reason might be "torch.ops.aten.item.default"
reason = "torch.ops.aten.item.default"
fix = suggest_fix(reason)
print(fix)
# → "Replace tensor.item() with tensor.tolist() or keep the value in tensor space."
```

The fix lookup table covers the most common causes:

| Cause | Fix |
|-------|-----|
| `.item()` | Use `tensor.tolist()` or keep the value in tensor space |
| `if tensor.sum() > 0` | Use `torch.where` or `torch.any()` |
| `print(tensor)` | Remove from forward pass or use `torch._logging` |
| `resize_()` | Use `reshape()` instead |
| `share_memory_()` | Avoid in compiled forward passes |
| `data_ptr()` | Avoid in compiled forward passes |

---

## Fixing the Most Common Graph Break: `.item()`

`.item()` extracts a scalar from a tensor. Dynamo cannot trace through it, so it breaks the graph.

This module causes a graph break:

```python
class BadModule(nn.Module):
    def forward(self, x):
        if x.sum().item() > 0:   # .item() forces Python fallback — graph break
            return x * 2
        return x - 2
```

Run `dynamo.explain` to see it:

```
Graph breaks: 1
Graphs:       2    ← two separate compiled graphs
```

Two graphs means two separate GPU kernel launches instead of one fused kernel.

**Fix 1 — stay in tensor space with `torch.where`:**

```python
class FixedModule(nn.Module):
    def forward(self, x):
        cond = (x.sum() > 0)         # tensor — Dynamo can trace this
        return torch.where(cond, x * 2, x - 2)
```

Running `dynamo.explain` on the fixed module:

```
Graph breaks: 0
Graphs:       1    ← one graph, fully fused
```

**Fix 2 — use `torch.any()` for boolean conditions:**

```python
class FixedModule(nn.Module):
    def forward(self, x):
        if torch.any(x.sum() > 0):   # torch.any() is traceable
            return x * 2
        return x - 2
```

**Fix 3 — use `tolist()` if you need a Python scalar:**

```python
class FixedModule(nn.Module):
    def forward(self, x):
        threshold = x.sum().tolist()  # converts outside the traced path
        if threshold > 0:
            return x * 2
        return x - 2
```

---

## Watching for Recompilation

Recompilation happens when Dynamo encounters a new input shape and has no cached kernel for it. You pay the full compilation cost again. Watch for it with logging:

```python
torch._logging.set_logs(recompiles=True)
compiled = torch.compile(model)
output = compiled(x)
```

When Dynamo recompiles, you will see messages containing "Compiling graph" and "Graph recompiled" in your output. Each recompilation costs the same as the first compilation — several seconds of stall in your training loop.

This happens when your input shapes change. For example, if batch size switches from 32 to 64 and Dynamo has no cached kernel for shape `(64, 512)`, it recompiles:

```
Batch size 32:
[torch._dynamo] Compiling graph 0 ...     ← first compilation

Batch size 64 (new shape):
[torch._dynamo] Compiling graph 0 ...     ← recompilation — costs ~Y seconds again
```

Every recompilation is a stall in your training loop. If you see constant recompilations and your shapes genuinely vary, use `dynamic=True`:

```python
# Handles varying batch sizes without recompiling
compiled = torch.compile(model, dynamic=True)
```

The tradeoff: `dynamic=True` kernels are slower than shape-specific ones because Inductor cannot specialize as aggressively.

---

## The Warmup Rule and Synchronization

GPU calls are asynchronous. When Python calls a CUDA kernel, it returns immediately — the GPU is still running. You must synchronize before and after timing to get a real measurement.

```python
compiled = torch.compile(model)

_ = compiled(x)    # warmup — compilation happens here, do not time this
_ = compiled(x)    # warmup — establish the cached graph

torch.cuda.synchronize()  # wait for GPU to finish warmup
start = time.perf_counter()

for _ in range(1000):
    _ = compiled(x)

torch.cuda.synchronize()  # wait for GPU to finish the loop
elapsed = time.perf_counter() - start
```

This is the timeline:

```
| compile | warmup | warmup | warmup | [====timed loop====] |
         ^        ^       ^       ^                 ^
       ignore   ignore  ignore  ignore             time this
```

Without synchronize, you are timing Python overhead — not GPU execution time.

---

## VRAM Budget

`torch.compile` uses more GPU memory. The compiled graph artifacts live in VRAM alongside your model and activations.

This is roughly how memory is laid out:

```
Eager mode:
|----- model ----|----- activations ----|----- gradients ----|
      ~X GB              ~Y GB                 ~Y GB

Compiled mode:
|----- model ----|----- activations ----|----- gradients ----|----- compiled graph -----|
      ~X GB              ~Y GB                 ~Y GB                ~Z GB extra
```

The extra ~Z GB comes from TorchInductor's compiled graph cache and any CUDA graph state. For a 1B parameter model, total VRAM is roughly ~(X + 2Y + Z) GB in compiled mode versus ~(X + 2Y) GB in eager mode.

If you hit OOM after adding `torch.compile`, reduce batch size until it fits, then decide if the speedup is worth the memory cost.

---

## Quick Reference

```python
# ============================================================
# Debugging
# ============================================================

# Enable debug output — saves generated Triton kernels
# → run: benchmark.py (any model with config.debug=True set)
import torch._inductor.config as config
config.debug = True

# Log recompilations live
# → run: any compiled model
import torch._logging
torch._logging.set_logs(recompiles=True)

# Log graph breaks live
# → run: graph_breaks.py
torch._logging.set_logs(graph_breaks=True)

# Run the graph break scanner
# → run: graph_break_scanner.py
from graph_break_scanner import GraphBreakScanner
scanner = GraphBreakScanner(model)
scanner.report(torch.randn(8))

# Look up a fix from a break reason
# → run: graph_break_scanner.py
from graph_break_scanner import suggest_fix
fix = suggest_fix("torch.ops.aten.item.default")

# ============================================================
# Compilation modes
# ============================================================
# → run: benchmark_modes.py to compare all four

torch.compile(model, mode="default")                          # safe starting point
torch.compile(model, mode="max-autotune")                     # production, large models
torch.compile(model, mode="reduce-overhead")                    # small models, try first
torch.compile(model, mode="max-autotune-no-cudagraphs")       # fallback if reduce-overhead fails

# ============================================================
# Dynamic shapes
# ============================================================

torch.compile(model, dynamic=True)  # for variable batch sizes or seq lengths

# ============================================================
# Benchmarking
# ============================================================
# → run: benchmark.py for the three numbers

compiled = torch.compile(model)
_ = compiled(x)                    # warmup — do not time this
torch.cuda.synchronize()
start = time.perf_counter()
for _ in range(1000):
    _ = compiled(x)
torch.cuda.synchronize()
elapsed = time.perf_counter() - start
```

---

## Takeaways

- **Track three numbers**: eager baseline, first-call overhead, steady-state speedup.
- **`max-autotune` wins on large models.** `reduce-overhead` wins on small models — if it works.
- **MLPs benefit most from compilation.** CNNs benefit least. Transformers are in between.
- **`.item()` is the most common graph break.** Use `torch.where` or `torch.any()` instead.
- **Recompilation is a silent killer.** Log it with `torch._logging.set_logs(recompiles=True)`.
- **Always warm up and synchronize.** GPU calls are asynchronous.
- **`torch.compile` uses more VRAM.** Budget extra memory.

Sources:
- https://pytorch.org/tutorials/intermediate/torch_compile_tutorial.html
- https://pytorch.org/blog/sota-normalization-performance-with-torch-compile
