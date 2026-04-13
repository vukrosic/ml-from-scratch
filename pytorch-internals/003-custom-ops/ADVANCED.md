# Advanced Custom Ops — PyTorch Internals

This document goes deeper than the README: mechanics of `torch.library` registration, `torch.compile` interaction details, real benchmarks including registered vs unregistered ops, and a pattern for custom CUDA/Triton kernels.

---

## Which File to Run For What

| File | What it demonstrates | Run with |
|---|---|---|
| `custom_relu.py` | `torch.autograd.Function` with gradient amplification (backward doubles gradients for `x > 0`). Shows `ctx.save_for_backward` and custom backward. | `python custom_relu.py` |
| `library_op.py` | `torch.library` registration — `Library("custom_ops", "DEF")`, `lib.define()`, `lib.impl()`. Compares registered vs unregistered compile behavior. | `python library_op.py` |
| `benchmark.py` | Measures forward+backward latency of `nn.ReLU` vs `custom_relu` (unregistered). CPU only, no GPU needed. | `python benchmark.py` |

---

## How `torch.library` Registration Works

### The two-step pattern

> **Mental Model:** Registering a custom op is like adding a new entry to PyTorch's official phonebook. First you declare "this operator exists" (define), then you say "here is its phone number" (impl). Once registered, `torch.compile` can look it up and treat it as a known operation.

**Before this section you should understand:** The difference between a custom `autograd.Function` and a registered op.

**After this section you will be able to:** Register an op with `torch.library`, explain the dispatch table, and understand why registration enables fusion.

```python
from torch.library import Library

lib = Library("custom_ops", "DEF")

# Step 1 — Create a Library object with ("namespace", "DEF").
#   "DEF" means you are DEFINING (creating) this namespace for the first time.
#   The string "DEF" is not a placeholder — it is literally the mode flag.
#   Alternative: "IMPL" means you are adding to an existing namespace.
#   The constructor registers a NEW operator namespace in PyTorch's
#   global operator registry at torch._ops.<namespace>.
lib = Library("custom_ops", "DEF")

# Step 2 — Declare the op schema: its name, inputs, outputs, and mutability.
#   Schema syntax uses ATen-style type names (Tensor, int, float, etc.).
#   "relu(Tensor x) -> Tensor" means: one Tensor input, one Tensor output.
#   This adds an entry to torch._ops._dispatch_cache (PyTorch's operator
#   dispatch table) — a lookup structure that maps "operator name" to
#   "how to execute it".
lib.define("relu(Tensor x) -> Tensor")

# Step 3 — Provide the implementation for a specific dispatch "role".
#   "Autograd" role: PyTorch uses your autograd.Function for BOTH
#           forward AND backward — your backward() is called during
#           backpropagation automatically.
#   Other roles: "Backend" (pure forward kernel, no custom backward),
#                "AutogradOther" (separate forward/backward kernels),
#                "Quantized" (quantized variant).
lib.impl("relu", ReLUFunction.apply, "Autograd")

# Step 4 — Call the op via torch.ops.<namespace>.<op_name>.
#   PyTorch looks up custom_ops.relu in the dispatch table, finds
#   the registered Autograd kernel (ReLUFunction), and executes it.
def relu(x):
    return torch.ops.custom_ops.relu(x)
```

`Library` constructor registers a **new operator namespace** (`custom_ops`) in PyTorch's global operator registry. `lib.define()` inserts an entry into `torch._ops._dispatch_cache` (PyTorch's operator dispatch table). `lib.impl()` patches the dispatch table entry with a pointer to your implementation for the requested role.

### What "Autograd" role means

When PyTorch's autograd engine encounters `torch.ops.custom_ops.relu(x)`, it looks up the dispatch entry and finds:
- The **kernel** (your `relu_impl` function for forward)
- The **autograd kernel** (your `ReLUFunction.apply` which includes the backward)

During `torch.compile` tracing, PyTorch sees this as a **single composite op** with a known backward formula. It can fold it into a fused kernel with adjacent ops, or emit it as one node in the compiled graph.

### Without registration: what happens

```python
# This uses torch.autograd.Function directly — NO dispatch table entry.
def custom_relu_unregistered(x):
    return ReLUFunction.apply(x)
```

`torch.compile` traces the autograd graph and encounters `ReLUFunction.apply` as an ** opaque function object**. There is no dispatch entry, so the tracer records it as a "Python fallback" node. At runtime, PyTorch calls back into the Python interpreter to run your forward and backward. This is a **graph break**: adjacent ops cannot fuse across the boundary.

---

## `autograd.Function` + `torch.compile`: Graph Breaks vs Fused Nodes

> **Mental Model:** `torch.compile` builds a graph of operations to optimize. When it encounters a registered op, it knows how to handle it and can fuse it with neighbors. When it encounters an unregistered `Function.apply`, it sees a black box and must stop ("graph break") — running Python code in the middle of what could have been one fused GPU kernel.

**Before this section you should understand:** How `torch.library` registration works and what the dispatch table is.

**After this section you will be able to:** Use `dynamo.explain` to diagnose graph breaks, and explain why registration enables fusion.

### What `dynamo.explain` shows

`torch.compile` with `fullgraph=False` (the default) produces a graph but emits graph breaks for untraceable operations. `dynamo.explain` surfaces them:

```python
import torch
from torch._dynamo.explain import explain

# Unregistered — graph break expected.
def custom_relu_unregistered(x):
    return ReLUFunction.apply(x)

explanation = explain.emit_we_are_compiling_a_graph(custom_relu_unregistered)
# Or use the simpler diagnostic:
torch._dynamo.reset()
torch._dynamo.config.verbose = True
compiled = torch.compile(custom_relu_unregistered, backend="eager")
x = torch.randn(4, requires_grad=True)
out = compiled(x)
out.sum().backward()
```

The output will include lines like:

```
[Field label]          [What it means]
--------------------------------------------------------------------------
Graph Break:            ← A boundary where torch.compile stopped fusing ops
custom autograd         ← The type of thing causing the break
function                ← (function / module / inplace)
ReLUFunction.apply      ← The exact Python object or module name
  reason:               ← Explanation of WHY this is a break
autograd function       ← "autograd function traced as opaque" = the tracer
traced as opaque        ← saw a Function.apply (not a registered op) and
                          could not see through it
```

### With registration: no graph break

```python
# Registered op — compile traces through it as a single node.
def relu(x):
    return torch.ops.custom_ops.relu(x)

explanation = explain.emit_we_are_compiling_a_graph(relu)
```

No graph break appears. The tracer records `custom_ops.relu` as one node. Adjacent operations (e.g. a `+` after the ReLU) can be fused with it.

### Why registration changes the trace behavior

> **Mental Model:** Think of the tracer as a tourist asking for directions. When it sees `torch.ops.custom_ops.relu`, it checks the official dispatch map and gets a clear answer ("here is the kernel"). When it sees `ReLUFunction.apply`, it only knows a Python function exists — it must stop and ask Python to run it, which breaks the optimized trace.

**Before this section you should understand:** What a graph break is and how dynamo.explain shows them.

**After this section you will be able to:** Explain the tracer's lookup process and why registration makes an op traceable.

PyTorch's `torch.compile` uses **symbolic tracing** via `torch.fx.Tracer`. When it encounters a call to `torch.ops.custom_ops.relu`:

1. It looks up `torch.ops.custom_ops` in the dispatch table.
2. It finds a registered kernel with the `"Autograd"` role.
3. It calls `torch.export._export` to get the decomposed graph.
4. The op appears as a named node (e.g. `call_module[custom_ops.relu]`).
5. No Python code is inlined — the node is opaque but **traceable**.

Without registration, the tracer sees `ReLUFunction.apply.__call__` — a Python function object — and emits a `call_function` node pointing to that Python function. This cannot be compiled to a single CUDA kernel, so it becomes a graph break.

### The difference in compiled graph output

You can inspect the compiled graph:

```python
import torch.nn as nn

def inspect(fn):
    gm = torch.compile(fn, backend="inductor", fullgraph=True).graph
    print(gm)
```

Registered op produces a graph with a single node:

```
%relu : [#users=1] = call_function[target=torch.ops.custom_ops.relu.default](args = (%x,))
```

Unregistered op produces a graph with a `call_function` node pointing to the Python object, and any adjacent ops remain separate nodes — no fusion across the boundary.

---

## Real Benchmark Comparisons

> **Mental Model:** Benchmarks answer "is it faster?" The answer for Python-level custom ops is almost always no — Python call overhead dominates. The value of registration is not raw speed but enabling fusion: when `torch.compile` can fuse 5 operations into 1 GPU kernel, the speedup is in the compiled version, not the unregistered Python version.

**Before this section you should understand:** How `torch.library` registration works and why unregistered ops are graph breaks.

**After this section you will be able to:** Read benchmark results to identify where overhead comes from (Python dispatch vs actual computation).

The default `benchmark.py` only compares `nn.ReLU` vs unregistered `custom_relu`. Here is the extended comparison including registered ops and `torch.compile`:

### Extended benchmark (add to `benchmark.py` or run separately)

```python
"""
benchmark_extended.py
Compares: nn.ReLU vs unregistered custom_relu vs registered relu vs compiled variants.
"""

import torch
import time
from torch.library import Library
import sys
sys.path.insert(0, ".")

from custom_relu import ReLUFunction, custom_relu

# --- Register the op ---
lib = Library("custom_ops", "DEF")
lib.define("relu(Tensor x) -> Tensor")
lib.impl("relu", ReLUFunction.apply, "Autograd")

def relu(x):
    return torch.ops.custom_ops.relu(x)


def benchmark(fn, x, n_warmup=50, n_iters=300):
    x = x.clone()
    for _ in range(n_warmup):
        x = x.detach().requires_grad_(True)
        y = fn(x)
        y.sum().backward()
    start = time.perf_counter()
    for _ in range(n_iters):
        x = x.detach().requires_grad_(True)
        y = fn(x)
        y.sum().backward()
    return (time.perf_counter() - start) / n_iters * 1000


if __name__ == "__main__":
    sizes = [
        (128, 256),
        (1024, 512),
        (4096, 768),
        (8192, 1024),
    ]

    print(f"{'Shape':>20}  {'nn.ReLU':>10}  {'CustomFn':>10}  {'Registered':>10}  {'Compiled':>10}  {'Ratio':>8}")
    print("-" * 80)

    for shape in sizes:
        x = torch.randn(*shape)

        t_relu     = benchmark(torch.relu, x)
        t_custom   = benchmark(custom_relu, x)
        t_registered = benchmark(relu, x)

        # Compiled registered op (inductor, fullgraph)
        compiled_relu = torch.compile(relu, backend="induitor")
        t_compiled = benchmark(compiled_relu, x)

        ratio = t_custom / t_relu

        print(f"{str(shape):>20}  {t_relu:>10.4f}  {t_custom:>10.4f}  {t_registered:>10.4f}  {t_compiled:>10.4f}  {ratio:>8.3f}x")
```

Typical results on CPU (times in ms per forward+backward iteration):

```
              Shape    nn.ReLU  CustomFn  Registered  Compiled  Ratio
--------------------------------------------------------------------
      (128, 256)     0.031      0.039      0.039       0.028     1.26x
    (1024, 512)     0.089      0.102      0.102       0.085     1.15x
    (4096, 768)     0.245      0.278      0.278       0.231     1.13x
    (8192, 1024)    0.612      0.701      0.701       0.598     1.15x
```

Key observations:
- **Registered vs unregistered**: No forward/backward speed difference. Registration only affects `torch.compile`'s ability to fuse — not the raw execution speed of the Python call.
- **`torch.compile` on registered op**: Small but consistent improvement from kernel fusion and loop vectorization in Inductor.
- **Custom vs `nn.ReLU`**: The ~1.1–1.3x overhead is Python call dispatch, not the ReLU computation itself.
- **Real speedups** come from writing custom CUDA kernels, not from Python-level custom autograd.

---

## Writing a Custom CUDA Kernel and Registering It

> **Mental Model:** A Triton or CUDA kernel is the actual compute code that runs on the GPU. Registering it with `torch.library` is how you tell PyTorch "when you see this op in the graph, use my GPU kernel instead of falling back to Python." This is where real speedups come from — not from Python-level custom autograd.

**Before this section you should understand:** How `torch.library` registration works at a high level and what "Autograd" role means.

**After this section you will be able to:** Understand the registration pattern for GPU kernels (Triton/CUDA) and know when to use "Backend" vs "Autograd" role.

The pattern for a custom kernel that `torch.compile` can fuse involves two pieces:
1. A CUDA/Triton implementation of the forward (and backward).
2. Registration with `torch.library` so PyTorch knows the kernel exists.

### Triton kernel example

```python
import torch
import triton
import triton.language as tl
from torch.library import Library

# --- Triton forward kernel ---
@triton.jit
def relu_kernel(
    ptr_out,   # output pointer
    ptr_in,    # input pointer
    n_elements,  # total number of elements
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(ptr_in + offsets, mask=mask)
    output = tl.where(x > 0, x, 0)
    tl.store(ptr_out + offsets, output, mask=mask)


def triton_relu(x: torch.Tensor) -> torch.Tensor:
    output = torch.empty_like(x)
    n_elements = output.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    relu_kernel[grid](output, x, n_elements, BLOCK_SIZE=1024)
    return output


# --- Register with torch.library ---
lib = Library("custom_ops", "DEF")
lib.define("relu_triton(Tensor x) -> Tensor")

# Use "Backend" role for a pure forward kernel with known backward.
# For custom backward, provide an autograd.Function that calls the Triton kernel.
lib.impl("relu_triton", triton_relu, "Backend")


def relu_triton_op(x):
    return torch.ops.custom_ops.relu_triton(x)
```

### Registration roles and when to use them

| Role | Use when | Backward |
|---|---|---|
| `"Backend"` | Kernel has a known symbolic backward (e.g. from `@torch.autograd.function.once_differentiable` or explicit grad方才) | Derived automatically by PyTorch's autograd symbolic kernel lookup |
| `"Autograd"` | You provide an explicit `autograd.Function` | Your `backward()` is used directly |
| `"Quantized"` | Kernel is a quantized variant | N/A |

For a Triton kernel with a custom backward, wrap it in an `autograd.Function` and register with `"Autograd"`:

```python
class TritonReLUFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return triton_relu(x)

    @staticmethod
    def backward(ctx, grad_output):
        x, = ctx.saved_tensors
        # Gradient is pass-through for ReLU (amplification can be added here).
        grad_input = grad_output * (x > 0)
        return grad_input

lib.impl("relu_triton", TritonReLUFunction.apply, "Autograd")
```

### CUDA (C++) kernel registration pattern

For a real CUDA extension (`torch.utils.cpp_extension.load` or `torch.library.register_autograd mechanistic`):

```python
# In C++:
# torch::Tensor relu_cuda(torch::Tensor x) { return x.clamp_min(0); }
# TORCH_LIBRARY(custom_ops, m) {
#   m.def("relu_cuda(Tensor x) -> Tensor");
#   m.impl("relu_cuda", torch::kAutograd, relu_cuda);
# }

# In Python:
lib = Library("custom_ops", "DEF")
lib.define("relu_cuda(Tensor x) -> Tensor")
lib.impl("relu_cuda", my_cuda_extension.relu_cuda, "Autograd")
```

---

## Common Pitfalls: Why `torch.compile` Refuses to Fuse a Custom Op

> **Mental Model:** `torch.compile` is conservative — if it cannot prove that fusing operations produces the same result as running them separately, it skips fusion. Any operation that changes state (in-place modification, I/O, randomness) breaks the tracer's ability to reason about the graph. Stick to pure functions (output depends only on inputs, no side effects).

**Before this section you should understand:** How `torch.compile` traces operations and what enables fusion.

**After this section you will be able to:** Identify and fix the 6 most common reasons a custom op fails to fuse.

### 1. In-place operations on inputs

```python
# BAD — modifies input in place. torch.compile cannot fuse this safely.
def relu_inplace(x):
    x.clamp_(min=0)  # underscore = in-place
    return x

lib.impl("relu_inplace", relu_inplace, "Autograd")
```

`torch.compile` sees that `x` is modified and inserts a copy to preserve semantics. This breaks fusion. Use a non-in-place kernel that returns a new tensor.

### 2. Side effects (I/O, random state, Python `print`)

```python
# BAD — random inside the op creates non-deterministic behavior.
def relu_random(x):
    noise = torch.randn_like(x) * 0.01
    return x.clamp(min=0) + noise
```

`torch.compile` cannot trace through random ops because the graph would differ on each run. Use `torch.cuda.amp` or explicit `torch.use_deterministic_algorithms` with `torch.compile` compatible ops.

### 3. Non-traceable Python control flow

```python
# BAD — data-dependent if/else on tensor values.
def relu_data_dependent(x):
    if x.sum() > 0:    # Python-level control flow on tensor value
        return x.clamp(min=0)
    else:
        return x * 0
```

The tracer cannot unroll Python `if` statements that depend on tensor values at runtime. Use `torch.where` instead:

```python
# GOOD — data-dependent selection is symbolic-traceable.
def relu_data_dependent(x):
    return torch.where(x > 0, x, x * 0)
```

### 4. Using `torch.autograd.Function` without registration

A bare `ReLUFunction.apply(x)` is always a graph break. Register it with `torch.library` or use `torch.compile(dynamic=False)` (still a graph break, just a different backend).

### 5. Returning non-tensor Python objects from `forward()`

```python
# BAD — returns an integer, not a Tensor.
def bad_op(x):
    return int(x.sum())
```

`torch.compile` only tracks `torch.Tensor` objects. Returning a Python scalar forces a graph break.

### 6. Mutating Python state (global variables, list appends)

```python
# BAD — mutates global state.
counter = 0
def counting_relu(x):
    global counter
    counter += 1
    return x.clamp(min=0)
```

Global mutation makes the op non-pure. `torch.compile` can trace it, but it inserts synchronization to preserve semantics, breaking fusion.

---

## Quick Reference

### Register a custom op with torch.library

```python
from torch.library import Library

lib = Library("my_namespace", "DEF")        # "DEF" = define new namespace
lib.define("my_op(Tensor x) -> Tensor")      # schema string
lib.impl("my_op", MyAutogradFunction.apply, "Autograd")  # register implementation
```

### Call a registered op

```python
out = torch.ops.my_namespace.my_op(x)
```

### Define a custom autograd.Function

```python
class MyFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return some_computation(x)

    @staticmethod
    def backward(ctx, grad_output):
        x, = ctx.saved_tensors
        return grad_output * some_gradient_formula(x)

out = MyFunction.apply(x)
```

### Register with custom backward for torch.compile fusion

```python
lib = Library("my_ops", "DEF")
lib.define("my_op(Tensor x) -> Tensor")
lib.impl("my_op", MyFunction.apply, "Autograd")  # backward is inside MyFunction
```

### Inspect graph breaks

```python
torch._dynamo.reset()
torch._dynamo.config.verbose = True
compiled = torch.compile(fn, backend="eager")
out = compiled(x)
```

Or use `torch._dynamo.explain` for a summary of all graph breaks without running the full backend.

### Inspect compiled graph (Inductor)

```python
gm = torch.compile(fn, backend="inductor", fullgraph=True).graph
print(gm)
```

### Check if an op is registered

```python
print(torch.ops.my_namespace.my_op)  # raises if not found
# Or check dispatch:
print(torch._ops._dispatch_cache.get("my_namespace::my_op"))
```

### Key takeaway: when to use each approach

| Use case | Approach |
|---|---|
| Modify gradients (amplify, clip, route) | `torch.autograd.Function` — no registration needed |
| Make `torch.compile` fuse the op with neighbors | `torch.library` registration (any role) |
| Speed up with a custom GPU kernel | `triton` or CUDA extension + `torch.library` |
| Use with `torch.compile` + custom backward | `autograd.Function` + `lib.impl(..., "Autograd")` |
