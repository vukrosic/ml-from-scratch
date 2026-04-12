# torch.compile From Scratch

Every PyTorch model runs eagerly by default: one Python operation, one GPU kernel, idle gap, repeat. `torch.compile` closes that gap -- it wraps a model and returns a faster version with the same API. Here is what actually happens inside.

## What we build

- A toy tracer that records operations into a graph
- A fusion pass that merges adjacent operations
- A code generator that produces executable Python from a graph
- A real torch.compile benchmark comparing eager vs compiled
- Examples of graph breaks and how to fix them

## Hook

Imagine a restaurant where the chef shouts one dish at a time to the kitchen, waits for it to finish, then shouts the next. Every gap between dishes is pure dead time. That is eager mode. `torch.compile` is the chef handing the kitchen a full ticket at once -- one trip, everything cooks together.

---

## Lesson 1 · Tracing a Computation Graph

Before compiling anything, we need to understand what we are compiling. A computation graph is a list of operations where each op knows its inputs and produces an output.

This is what TorchDynamo does -- it traces your forward pass and builds this graph. Here is a toy version:

```python
class TracedTensor:
    _counter = 0

    def __init__(self, name=None):
        if name is None:
            name = f"t{TracedTensor._counter}"
            TracedTensor._counter += 1
        self.name = name
        self.graph = []

    def _record(self, op, other=None):
        result = TracedTensor()
        result.graph = self.graph
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

Each operation returns a new `TracedTensor` and records itself into the shared graph. Running an MLP through `trace()` gives us the full operation list.

```python
def trace(fn, input_names):
    TracedTensor._counter = 0
    inputs = [TracedTensor(name=n) for n in input_names]
    output = fn(*inputs)
    return output.graph
```

Run it:

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

Output:

```
('t3', 'matmul', ['x', 'w1'])
('t4', 'relu', ['t3'])
('t5', 'matmul', ['t4', 'w2'])
```

Each tuple is `(output_name, operation, input_names)`. TorchDynamo produces an FX graph with the same structure.

---

## Lesson 2 · Fusing Operations

The compiled version is faster partly because TorchInductor fuses adjacent operations into a single kernel. Instead of launching a matmul kernel then a relu kernel, it runs one fused kernel that does both.

Our fusion pass does this for matmul + relu:

```python
def fuse_graph(graph):
    fused = []
    skip_next = set()

    for i, (out, op, args) in enumerate(graph):
        if i in skip_next:
            continue

        if op == "matmul" and i + 1 < len(graph):
            next_out, next_op, next_args = graph[i + 1]
            if next_op == "relu" and next_args[0] == out:
                fused.append((next_out, "fused_matmul_relu", args))
                skip_next.add(i + 1)
                continue

        fused.append((out, op, args))

    return fused
```

Before fusion: `matmul → relu → matmul`. After fusion: `fused_matmul_relu → matmul`. One kernel instead of three.

---

## Lesson 3 · Generating Executable Code

From an optimized graph, we can generate real Python code that uses torch operations:

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

TorchInductor generates Triton kernel code. We generate torch Python. The idea is the same: turn the graph back into runnable code.

---

## Lesson 4 · Eager vs Compiled Benchmark

Here is the real test on a 4-layer MLP with 4096 hidden dimension:

```python
class MLP(nn.Module):
    def __init__(self, d_in=1024, d_hidden=4096, d_out=1024):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, d_hidden),
            nn.GELU(),
            nn.Linear(d_hidden, d_out),
        )

    def forward(self, x):
        return self.net(x)
```

Benchmark:

```python
def benchmark(fn, x, steps=200, warmup=50):
    for _ in range(warmup):
        fn(x)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(steps):
        fn(x)
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / steps
```

Run:

```python
model = MLP().to("cuda").eval()
x = torch.randn(256, 1024, device="cuda")

eager_time   = benchmark(model, x)
compiled     = torch.compile(model)
compile_ms   = ...  # measure first call
steady_ms    = benchmark(compiled, x)

print(f"Eager:           {eager_time*1000:.2f} ms")
print(f"Compiled steady: {steady_ms*1000:.2f} ms")
print(f"Speedup:         {eager_time/steady_ms:.1f}x")
```

On a typical GPU this gives 1.5-3x speedup on steady-state runs. The first compiled call is always 2-10 seconds of compilation overhead.

---

## Lesson 5 · Graph Breaks and How to Fix Them

TorchDynamo traces Python operations into a graph. When it hits something it cannot represent -- usually a data-dependent `if` -- it stops tracing, runs that piece in Python, then resumes. Each gap is a graph break.

The most common cause is `.item()`:

```python
class BadModule(nn.Module):
    def forward(self, x):
        if x.sum().item() > 0:   # .item() forces Python fallback
            return x * 2
        return x - 2
```

`.item()` extracts a scalar from a tensor, forcing TorchDynamo to fall back to Python. The fix is `torch.where`, which stays in tensor space:

```python
class FixedModule(nn.Module):
    def forward(self, x):
        cond = (x.sum() > 0)         # stays as a tensor
        return torch.where(cond, x * 2, x - 2)
```

Detect graph breaks with `torch._dynamo.explain`:

```python
import torch._dynamo as dynamo

model = BadModule()
explanation = dynamo.explain(model)(torch.randn(10))
print(f"Graph breaks: {explanation.graph_break_count}")
print(f"Graphs:       {explanation.graph_count}")
```

---

## Recap

- `torch.compile(model)` returns a faster version with the same API
- TorchDynamo traces the forward pass into an FX graph
- TorchInductor fuses adjacent ops and emits Triton kernels
- The first call always compiles -- warm up before timing
- `.item()` in a forward pass causes graph breaks -- use `torch.where` instead

Get the video walkthrough of profiling across 5 architectures, annotated Triton kernel output, and the automated graph break scanner: [OpenSuperintelligenceLab on Skool](https://www.skool.com/opensuperintelligencelab)
