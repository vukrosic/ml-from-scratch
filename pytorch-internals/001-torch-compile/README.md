# torch.compile From Scratch

One line of code turns your PyTorch model into a fused GPU kernel — and it runs 2–5× faster. But the same line silently breaks your model if you do not understand what is happening inside. This is what `torch.compile` actually does.

---

## What Problem torch.compile Solves

### The Problem: Idle GPU Time

Every time you call `model(x)` in PyTorch, Python dispatches operations one at a time:

```python
x = self.linear1(x)   # Python → GPU: linear1 kernel launches
x = torch.relu(x)     # Python → GPU: ReLU kernel launches
x = self.linear2(x)   # Python → GPU: linear2 kernel launches
```

What the GPU actually sees:

```
GPU timeline:
|  linear1 kernel  |  idle  |  ReLU kernel  |  idle  |  linear2 kernel  |
|<---- ~X ms ---->|<-gap-->|<---- ~Y ms --->|<-gap-->|<----- ~Z ms ---->|
```

Each gap is dead time. The GPU finished one kernel and sat idle waiting for Python to send the next message. For a model with 100 layers, those 99 gaps compound.

### The Fix: Fuse Everything Into One Kernel

`torch.compile` takes the full sequence and fuses it into a single GPU kernel:

```
GPU timeline (compiled):
|           one fused kernel (linear1 + ReLU + linear2)            |
|<---------------------- ~X ms total ---------------------------->|
```

The GPU runs the whole thing without stopping. No gaps. No round-trips to memory between operations.

PyTorch's own benchmark on a 4096×4096 matrix shows a median 2.5× speedup after compilation (https://pytorch.org/tutorials/intermediate/torch_compile_tutorial.html). Run `benchmark.py` to see your speedup — expect roughly ~2-3x on steady-state runs.

### The Catch: Compilation Is Not Free

The first call is always slow — TorchDynamo has to trace your model, AOT Autograd has to capture the backward pass, and TorchInductor has to generate and compile a Triton kernel. This can take several seconds depending on model size.

```
Call 1:  |______________compilation_____________|  run once
Call 2:  |---fast---|---fast---|---fast---|      cached
Call 3:  |---fast---|---fast---|---fast---|      cached
...
```

After that, every call is fast. The compilation cost pays for itself over hundreds of training steps.

### The Other Catch: Graph Breaks Fragment the Graph

If TorchDynamo encounters something it cannot compile — like `if x.sum() > 0` — it splits the graph. Fusion cannot span the break. You end up with two smaller kernels instead of one big one, and the performance gains evaporate.

Every break is a lost fusion opportunity. Understanding why breaks happen (and how to fix them) is the rest of this article.

---

## The Three-Stage Pipeline

`torch.compile` is not one compiler — it is three components in sequence. Each does one job.

### Stage 1: TorchDynamo Traces the Graph

Dynamo hooks into Python's bytecode interpreter. As your model runs, it records each operation into an FX graph — a list of ops with inputs and outputs.

What Dynamo produces:

```
('t3', 'matmul', ['x', 'w1'])
('t4', 'relu',   ['t3'])
('t5', 'matmul', ['t4', 'w2'])
```

Each tuple is `(output_name, operation, input_names)`. When Dynamo encounters something it cannot compile — a data-dependent `if tensor.sum() > 0` — it ends the current graph, runs that piece in normal Python, then starts a new graph. Each gap is a **graph break**.

Dynamo is correct first. If it cannot trace something, it breaks rather than silently producing wrong results. (https://pytorch.org/tutorials/intermediate/torch_compile_tutorial.html)

### Stage 2: AOT Autograd Captures the Backward Pass

PyTorch normally computes gradients on the fly during training. That live computation would defeat fusion — the backward pass would re-introduce all the gaps that the forward fusion eliminated.

AOT Autograd runs the forward pass once internally, captures the backward computation graph, and packages both together as a joint graph:

```
Joint graph:
  Forward:  matmul → relu → matmul
  Backward: matmul_grad ← relu_grad ← matmul_grad
```

This is why `torch.compile` works in training mode — both passes are compiled as one unit.

### Stage 3: TorchInductor Emits Optimized Kernels

Inductor receives the joint graph and generates hardware-specific code. On NVIDIA GPUs, it produces Triton kernels — fast, fusion-friendly GPU code written in Python-like syntax. On CPUs, it generates C++ kernels.

What Inductor emits for the fused graph above — a single Triton kernel instead of three:

```triton
@triton.jit
def kernel(in_ptr, out_ptr, W1, W2, ...):
    x = tl.load(in_ptr)
    h = x @ W1           # matmul
    h = h * tl.sigmoid(h) * 1.702  # GELU
    out = h @ W2         # second matmul
    tl.store(out_ptr, out)
```

Inductor can fuse `linear → relu → linear → relu` into a single kernel. The GPU never writes intermediate results to memory — values stay in registers between operations.

### The Pipeline as a Diagram

```
Your Python model
       │
       ▼
[TorchDynamo]     intercepts bytecode, builds FX graph
       │
       ▼
[AOT Autograd]   captures backward pass, builds joint graph
       │
       ▼
[TorchInductor]  generates Triton/C++ kernels
       │
       ▼
Fused GPU kernel
```

---

## What Graph Breaks Actually Are

A graph break is not an error — it is Dynamo pausing compilation for a piece of code it cannot trace.

The consequence: each break fragments the graph. Fused kernels cannot span a break. You end up with several smaller kernels instead of one big one — and the performance gains evaporate.

```
Without break:     [==== fused: linear1 + ReLU + linear2 + GELU ====]
                        all one kernel

With a break:      [== fused: linear1 + ReLU ==]  | break |  [== fused: linear2 + GELU ==]
                                                 two separate kernels
```

### Detect graph breaks with dynamo.explain

```python
import torch._dynamo.explain

explain_output = dynamo.explain(model)
print(f"Graph breaks: {explain_output.graph_break_count}")
print(f"Graphs:       {explain_output.graph_count}")
```

Example output for a model with one graph break:

```
Graph breaks: 1
Graphs:       2
```

Dynamo produced 2 separate compiled graphs instead of 1 — because one piece of code could not be traced. Each graph is a separate GPU kernel launch.

Three common triggers:

**1. Data-dependent control flow**

```python
if x.sum() > 0:          # break: depends on tensor value at runtime
```

Use `torch.cond` for conditional execution that compiles.

**2. Printing or logging tensors**

```python
print(hidden_state)       # break: side effect Dynamo cannot trace safely
```

Move prints outside the compiled region or use `torch._logging`.

**3. Unsupported Python built-ins**

```python
len(tensor_shape)         # break: some len() patterns are not yet traced
```

---

## Tracing a Computation Graph

Before compiling anything, we need to understand what we are compiling. A computation graph is a list of operations where each op knows its inputs and produces an output. TorchDynamo does this — it traces your forward pass and builds an FX graph. Here is a toy version:

### Piece 1: TracedTensor holds a name and records operations

```python
class TracedTensor:
    _counter = 0

    def __init__(self, name=None):
        if name is None:
            name = f"t{TracedTensor._counter}"
            TracedTensor._counter += 1
        self.name = name    # unique name for this tensor in the graph
        self.graph = []     # shared list of all ops recorded so far

    def _record(self, op, other=None):
        # Create a new TracedTensor for the output
        result = TracedTensor()
        result.graph = self.graph   # share the same graph list
        # Build the op tuple: (output_name, operation, input_names)
        args = [self.name]
        if other is not None:
            args.append(other.name if isinstance(other, TracedTensor) else repr(other))
        self.graph.append((result.name, op, args))
        return result

    def __matmul__(self, other):
        return self._record("matmul", other)

    def __add__(self, other):
        return self._record("add", other)

    def relu(self):
        return self._record("relu")
```

### Piece 2: trace() runs a function and collects the graph

```python
def trace(fn, input_names):
    TracedTensor._counter = 0                    # reset naming counter
    inputs = [TracedTensor(name=n) for n in input_names]  # create input tensors
    output = fn(*inputs)                         # run the function, recording ops
    return output.graph                          # return the full operation list
```

### Piece 3: Run an MLP through the tracer

```python
def mlp(x, w1, w2):
    h = x @ w1
    h = h.relu()
    h = h @ w2
    return h

graph = trace(mlp, ["x", "w1", "w2"])
for node in graph:
    print(node)
```

Run this:

```
python tracer.py
```

You will see:

```
('t3', 'matmul', ['x', 'w1'])
('t4', 'relu', ['t3'])
('t5', 'matmul', ['t4', 'w2'])
```

Each tuple is `(output_name, operation, input_names)`. This is exactly the structure TorchDynamo produces as an FX graph.

---

## Fusing Operations

The compiled version is faster partly because TorchInductor fuses adjacent operations into a single kernel. Instead of launching a matmul kernel then a relu kernel, it runs one fused kernel that does both.

### Before and after fusion

The tracer gives us this:

```
Before fusion:
  ('t3', 'matmul', ['x', 'w1'])
  ('t4', 'relu',   ['t3'])
  ('t5', 'matmul', ['t4', 'w2'])
```

After the fusion pass:

```
After fusion:
  ('t4', 'fused_matmul_relu', ['x', 'w1'])
  ('t5', 'matmul', ['t4', 'w2'])
```

`matmul → relu` became one `fused_matmul_relu` node. One kernel instead of three.

### The fusion pass

```python
def fuse_graph(graph):
    fused = []
    skip_next = set()           # indices to skip after a fusion

    for i, (out, op, args) in enumerate(graph):
        if i in skip_next:       # already fused into a previous op
            continue

        # Check if this matmul is followed by a relu on the same output
        if op == "matmul" and i + 1 < len(graph):
            next_out, next_op, next_args = graph[i + 1]
            if next_op == "relu" and next_args[0] == out:
                # Merge: matmul + relu becomes one fused operation
                fused.append((next_out, "fused_matmul_relu", args))
                skip_next.add(i + 1)    # skip the relu since it's merged
                continue

        fused.append((out, op, args))

    return fused
```

TorchInductor does this in hardware — it scans the graph for adjacent compatible ops and merges them into single Triton kernels.

---

## Generating Executable Code

From an optimized graph, we can generate real Python code that uses torch operations. TorchInductor generates Triton kernel code; we generate torch Python. The idea is the same: turn the graph back into runnable code.

```python
def codegen(graph, fn_name="compiled_fn"):
    lines = [f"def {fn_name}(x, w1, w2):"]

    for out, op, args in graph:
        if op == "fused_matmul_relu":
            lines.append(f"    {out} = torch.relu({args[0]} @ {args[1]})")
        elif op == "matmul":
            lines.append(f"    {out} = {args[0]} @ {args[1]}")
        elif op == "relu":
            lines.append(f"    {out} = torch.relu({args[0]})")
        elif op == "add":
            lines.append(f"    {out} = {args[0]} + {args[1]}")

    lines.append(f"    return {graph[-1][0]}")
    return "\n".join(lines)
```

Running codegen on the fused graph produces:

```
def compiled_fn(x, w1, w2):
    t4 = torch.relu(x @ w1)
    t5 = t4 @ w2
    return t5
```

This is real runnable Python — it is what TorchInductor produces in hardware as a Triton kernel.

---

## A Minimal End-to-End Example

Here is a complete, runnable example with every piece explained.

### Piece 1: Define the model

```python
import torch
import torch.nn as nn

# A simple two-layer network
# Small enough to compile fast, complex enough to show fusion
class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        # First layer: 512 input features -> 256 output features
        self.linear1 = nn.Linear(512, 256)
        # Second layer: 256 -> 10 (for classification)
        self.linear2 = nn.Linear(256, 10)

    def forward(self, x):
        # Pass through first layer
        x = self.linear1(x)
        # Apply ReLU non-linearity
        x = torch.relu(x)
        # Pass through second layer
        x = self.linear2(x)
        return x
```

### Piece 2: Move to GPU

```python
# Create model instance
model = SimpleModel()

# Move model parameters to GPU
# torch.compile does not auto-detect device — you must move explicitly
model = model.cuda()

# Create a random input tensor on GPU
# Shape: (batch=1024, features=512)
x = torch.randn(1024, 512, device='cuda')
```

### Piece 3: Compile the model

```python
# One line. Same model, same forward method.
# The compiled model has the same API — nothing else changes.
compiled_model = torch.compile(model)
```

### Piece 4: First call compiles (slow — this is normal)

```python
# First call is always slow — compilation happens here.
# This can take several seconds depending on model size.
output = compiled_model(x)
```

### Piece 5: Subsequent calls are fast

```python
# Warmup complete. The compiled graph is cached.
# GPU kernel is fused: linear1 + relu + linear2 run as one unit.
for _ in range(10):
    output = compiled_model(x)
```

### The full code together

```python
import torch
import torch.nn as nn

class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(512, 256)
        self.linear2 = nn.Linear(256, 10)

    def forward(self, x):
        x = self.linear1(x)
        x = torch.relu(x)
        x = self.linear2(x)
        return x

model = SimpleModel().cuda()
x = torch.randn(1024, 512, device='cuda')

# Compile — one line
compiled_model = torch.compile(model)

# First call: compiles (slow)
output = compiled_model(x)

# Subsequent calls: fast
for _ in range(10):
    output = compiled_model(x)
```

### The benchmark function

This function measures milliseconds per step after warmup. It handles the GPU synchronization that accurate timing requires:

```python
import time

def benchmark(fn, x, steps=200, warmup=50):
    # Warmup runs the model without timing
    # This ensures the compiled graph is cached
    for _ in range(warmup):
        fn(x)

    # Synchronize before timing — GPU calls are asynchronous
    # Without this, we measure Python overhead, not GPU execution
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(steps):
        fn(x)

    # Synchronize again to get the true end time
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / steps
```

Run the full benchmark:

```python
model = SimpleModel().cuda()
x = torch.randn(1024, 512, device='cuda')

# Eager baseline
eager_time = benchmark(model, x)

# Compiled
compiled = torch.compile(model)
compiled_time = benchmark(compiled, x)

print(f"Eager:    {eager_time*1000:.2f} ms")
print(f"Compiled: {compiled_time*1000:.2f} ms")
print(f"Speedup:  {eager_time/compiled_time:.2f}x")
```

Run `benchmark.py` on your GPU to get your numbers. Expect roughly ~2-3x speedup on steady-state runs.

---

## Compilation Modes

`torch.compile` accepts a `mode` argument that controls optimization aggressiveness.

```python
torch.compile(model, mode="default")        # balanced — safe starting point
torch.compile(model, mode="reduce-overhead") # uses CUDA graphs to reduce launch overhead
torch.compile(model, mode="max-autotune")    # autotuner picks fastest kernel variants
```

| Mode | When to use it |
|------|----------------|
| `"default"` | General use — good balance |
| `"reduce-overhead"` | Small models where launch overhead dominates. Not guaranteed to work (no input mutation). |
| `"max-autotune"` | Production, large models. Autotuner tries multiple tiling strategies. Slower to compile. |

Choosing a mode:

```python
# Rapid iteration — compile time matters
compiled = torch.compile(model, mode="default")

# Production, large model — maximize runtime performance
compiled = torch.compile(model, mode="max-autotune")

# Small model — try reduce-overhead first
compiled = torch.compile(model, mode="reduce-overhead")
# If it errors at runtime (CUDA graph incompatibility): fall back
compiled = torch.compile(model, mode="max-autotune-no-cudagraphs")
```

### CUDA Graphs: One Sentence

`"reduce-overhead"` enables CUDA graph capture. A CUDA graph records a sequence of GPU operations as a single replayable unit — instead of launching 10 separate kernels, the GPU plays back one pre-recorded graph. This eliminates per-kernel launch overhead entirely.

---

## Graph Breaks and How to Fix Them

TorchDynamo traces Python operations into a graph. When it hits something it cannot represent — usually a data-dependent `if` — it stops tracing, runs that piece in Python, then resumes. Each gap is a graph break.

The most common cause is `.item()`:

```python
class BadModule(nn.Module):
    def forward(self, x):
        if x.sum().item() > 0:   # .item() forces Python fallback — graph break
            return x * 2
        return x - 2
```

Running `dynamo.explain` on this module shows the break clearly:

```
BadModule — dynamo.explain output:
  Graph breaks: 1
  Graphs:       2
```

The model compiled into 2 separate graphs instead of 1 — because `.item()` forced a Python fallback.

The fix is `torch.where`, which stays in tensor space:

```python
class FixedModule(nn.Module):
    def forward(self, x):
        cond = (x.sum() > 0)         # stays as a tensor — no break
        return torch.where(cond, x * 2, x - 2)
```

Running `dynamo.explain` on the fixed module:

```
FixedModule — dynamo.explain output:
  Graph breaks: 0
  Graphs:       1
```

One graph — fully compiled, fully fused.

---

## Recap

- **One line, same API, faster output.** `torch.compile(model)` wraps any PyTorch model.
- **Three stages:** TorchDynamo traces bytecode → AOT Autograd captures backward → TorchInductor generates Triton kernels.
- **Graph breaks fragment the graph.** Debug with `torch._dynamo.explain` and `torch._logging.set_logs(graph_breaks=True)`.
- **Recompilations cost real time.** Stabilize input shapes; use `dynamic=True` for genuine variation.
- **Always warm up.** First call compiles — never time it.
- **Synchronize before measuring.** GPU calls are asynchronous.
- **`.item()` in a forward pass causes graph breaks.** Use `torch.where` instead.
- **`torch.compile` increases VRAM usage.** Budget extra memory in GPU-constrained environments.

---

## Going Further

For real benchmark numbers on a 4-layer MLP, annotated Triton kernel output, a mode comparison table across batch sizes, architecture profiling across 5 model types, and the graph break scanner in practice — see [ADVANCED.md](./ADVANCED.md).

Sources:
- https://pytorch.org/tutorials/intermediate/torch_compile_tutorial.html
- https://pytorch.org/blog/sota-normalization-performance-with-torch-compile
