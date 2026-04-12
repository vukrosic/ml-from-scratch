# Autograd From Scratch — Extended Notebook Notes

*This is the extended content for the $49 Skool tier. It goes deeper than the video and the free GitHub lesson.*

---

## 1. The Jacobian: One Gradient to Rule Them All

Every function `f: ℝⁿ → ℝᵐ` is represented at the point of interest by its **Jacobian matrix** — the matrix of all partial derivatives:

```
J[f](x) = | ∂f₁/∂x₁  ∂f₁/∂x₂  ... ∂f₁/∂xₙ |
          | ∂f₂/∂x₁  ∂f₂/∂x₂  ... ∂f₂/∂xₙ |
          |   ...       ...   ...   ...   |
          | ∂fₘ/∂x₁  ∂fₘ/∂x₂  ... ∂fₘ/∂xₙ |
```

When `m = 1` (a scalar loss), the Jacobian collapses to a **row vector**.
When we compute gradients in a neural network, we almost always have a scalar output,
so we are really computing the Jacobian-vector product (JVP).

### Why This Matters for Autograd

PyTorch's `backward()` computes `∂L/∂w` for every parameter `w` by
applying the chain rule through this Jacobian matrix. Each primitive operation
( matmul, relu, tanh, … ) has a known Jacobian, and autograd composes them.

For a scalar function `g(x)` where `x ∈ ℝⁿ`, the chain rule in Jacobian form is:

```
∂L/∂x = (∂g/∂x)ᵀ · ∂L/∂g
```

The `∂L/∂g` is already known (it's the incoming gradient). We just need the
Jacobian `∂g/∂x` for the current operation, then do a dot product.

---

## 2. Chain Rule Visualization

Consider `z = sin(x²)`:

```
x  →  [x² = y]  →  [sin(y) = z]
```

The chain rule:

```
dz/dx = dz/dy · dy/dx = cos(y) · 2x = cos(x²) · 2x
```

In our `Value` system:

```python
x = Value(2.0)
y = x * x          # y = x²,  dy/dx = 2x = 4
z = y.sin()        # dz/dy = cos(y)
z.backward()       # dz/dx = dz/dy · dy/dx = cos(4) · 4
```

The `_backward` closure for `sin` is:

```python
def sin(self):
    s = math.sin(self.data)
    out = Value(s, _op="sin", _prev=(self,))

    def _backward():
        # d(sin)/d(self) = cos(self.data)
        self.grad += math.cos(self.data) * out.grad

    out._backward = _backward
    return out
```

For `z = sin(x²)` with `x = 2`:
- `y = 4`
- `z = sin(4) ≈ -0.7568`
- `dy/dx = 2x = 4`
- `dz/dy = cos(y) = cos(4) ≈ -0.6536`
- `dz/dx = dz/dy · dy/dx = -0.6536 · 4 = -2.6144`

---

## 3. How PyTorch's Autograd Engine Actually Works

PyTorch's engine is conceptually identical to ours but handles tensors, in-place
operations, and concurrent backward passes. The key structures are:

### The `Tensor` object

PyTorch's `Tensor` has a `grad_fn` attribute pointing to the `Function` that created it:

```python
import torch
x = torch.tensor([2.0], requires_grad=True)
y = x * x       # y.grad_fn = <MulBackward0>
```

`grad_fn` is the node in the autograd graph. Each `grad_fn` knows how to
compute its own backward pass.

### The Autograd Tape

When you call `loss.backward()`, PyTorch:

1. Allocates a `GradBucket` for each leaf tensor.
2. Performs a **topological sort** of the gradient graph (same as our `build_order`).
3. Calls each `grad_fn`'s `backward()` method in reverse topological order.
4. Accumulates gradients into `Tensor.grad` fields.

```python
# Simplified pseudo-code for torch.autograd.backward
def backward(tensors, grad_tensors=None, ...):
    # Build the graph
    for tensor in tensors:
        tensor.grad_fn.apply(*grad_tensors or [1.0])
```

### In-Place Operations and the `leaf` Flag

PyTorch tracks **leaf tensors** (model parameters, inputs with `requires_grad=True`).
Only leaf tensors accumulate gradients in their `.grad` field.
Non-leaf tensors are internal nodes — they also have `grad_fn`, but their `.grad`
field is `None` after backward unless you use `retain_grad()`.

```python
x = torch.tensor([2.0], requires_grad=True)   # leaf
y = x * x                                     # non-leaf (has grad_fn)
y.backward()
print(x.grad)    # tensor([4.])
print(y.grad)    # None (unless retain_grad() was called)
```

This is identical to our `Value` where we only populate `.grad` for the
original leaves (which we track via the topological sort starting from the output).

### `retain_graph`

By default, `backward()` frees the graph after it runs (to save memory).
Calling `backward(retain_graph=True)` keeps the graph intact so you can call
backward multiple times (useful for gradient checks or multiple optimizers).

---

## 4. The Hook System

PyTorch provides two hook mechanisms for intercepting gradient flow:

### `register_hook`

```python
x = torch.tensor([2.0], requires_grad=True)
y = x * x

# Register a hook to see/modify the gradient as it flows back
hook_handle = y.register_hook(lambda grad: grad * 2)  # double the gradient

y.backward()
print(x.grad)   # 8.0 instead of 4.0  (doubled by the hook)

hook_handle.remove()   # clean up when done
```

### `register_full_backward_hook`

More powerful — hooks into the actual backward pass of a module:

```python
model = torch.nn.Sequential(
    torch.nn.Linear(10, 10),
    torch.nn.ReLU(),
    torch.nn.Linear(10, 1)
)

def hook(module, grad_input, grad_output):
    print(f"Module: {module}")
    print(f"grad_output: {grad_output}")
    return grad_input   # or modified grad_input

for name, param in model.named_parameters():
    print(f"{name}: shape={param.shape}, requires_grad={param.requires_grad}")
```

### Practical Use Cases

1. **Gradient clipping** — scale gradients that exceed a threshold
2. **Gradient debugging** — print gradients to check for vanishing/exploding
3. **Gradient accumulation** — custom schemes beyond `optimizer.zero_grad()`
4. **Hook into activations** — capture intermediate gradients for visualization

---

## 5. Writing Custom `autograd.Function`

`torch.autograd.Function` gives you full control over the forward and backward
pass. Use it when you have a custom operation that PyTorch doesn't know how to differentiate.

### The Template

```python
from torch.autograd import Function

class MyReLU(Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return input.clamp(min=0)

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        return grad_output * (input > 0).float()
```

### Using It

```python
import torch

class MyModel(torch.nn.Module):
    def forward(self, x):
        return MyReLU.apply(x)

model = MyModel()
x = torch.randn(4, 10, requires_grad=True)
output = model(x)
loss = output.sum()
loss.backward()
print(x.grad)
```

### Why Use `ctx.save_for_backward`?

The backward pass may run on a different device or in a different order than
the forward pass. Anything needed for backward (intermediate activations,
hyperparameters, etc.) must be saved to `ctx` during forward.

### Real-World Example: Stepanov Differentiable Sorting

Custom autograd Functions are used in libraries like `numpy-grad` and
differentiable rendering pipelines where the forward pass calls a
non-differentiable library (like cuFFT or a sort routine) and the backward
pass provides the gradient approximation.

```python
class DifferentiableSort(Function):
    @staticmethod
    def forward(ctx, scores, temperature=1.0):
        # scores: (batch, n)
        ctx.temperature = temperature
        sorted_idx = scores.argsort(dim=-1)
        ctx.save_for_backward(sorted_idx, scores)
        return sorted_idx.float()

    @staticmethod
    def backward(ctx, grad_output):
        sorted_idx, scores = ctx.saved_tensors
        # Straight-through estimator: pass gradient through unchanged
        return grad_output, None
```

---

## 6. Profiling: When Does Autograd Become a Bottleneck?

For small models, autograd overhead is negligible. For large models with
millions of parameters, it adds up.

### Memory Overhead

Every intermediate tensor stored for backward is retained in the autograd graph.
With `torch.no_grad()`, you skip graph construction entirely — useful for inference.

### Computation Overhead

Forward and backward are both O(n) in the number of operations.
Backward is typically 1.2–1.5× slower than forward due to pointer chasing and
non-contiguous memory access patterns.

### Benchmark Script

```python
import torch, time

N = 1024
x = torch.randn(N, N, requires_grad=True)
y = torch.randn(N, N, requires_grad=True)

# Forward + backward with autograd
start = time.perf_counter()
z = (x @ y).relu()
z.sum().backward()
t_auto = time.perf_counter() - start

# Forward only (no grad tracking)
start = time.perf_counter()
with torch.no_grad():
    z2 = (x @ y).relu()
t_inference = time.perf_counter() - start

print(f"Forward+backward: {t_auto:.4f}s")
print(f"Forward only:     {t_inference:.4f}s")
print(f"Autograd overhead: {t_auto/t_inference:.2f}x")
```

---

## 7. Jacobians in Full: Vector-Jacobian Products

For outputs with `m > 1` (not a scalar), `backward()` requires a `grad_tensor`
of the same shape. This is the **upstream gradient** — the Jacobian of the
*next* function in the chain, multiplied into our Jacobian.

```python
# Two-output function
x = torch.tensor([1.0, 2.0], requires_grad=True)
y = x * x           # y = [1, 4]

# Upstream gradient for each output dimension
upstream = torch.tensor([1.0, 0.5])
y.backward(upstream)

print(x.grad)       # [2, 2]  —  ∂L/∂x = upstream · ∂y/∂x
```

This is the **vector-Jacobian product (VJP)**. Every `_backward` in our engine
does exactly this: `incoming_grad * local_jacobian`.

---

## 8. Edge Cases Our Toy Engine Doesn't Handle

- **In-place operations** (`a += b`) can overwrite gradients of other nodes.
  PyTorch raises a `RuntimeError` in this case.
- **Complex numbers** — gradients are different in the complex domain.
- **Sparse gradients** — PyTorch supports `sparse_coo_tensor` with different rules.
- **Higher-order derivatives** — `torch.autograd.grad` computes second-order gradients.
- **Distributed gradients** — `DistributedDataParallel` synchronizes gradients across processes.
