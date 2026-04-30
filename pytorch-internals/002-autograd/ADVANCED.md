# Autograd Advanced

> Read [README.md](./README.md) first.
>
> This guide assumes you already understand the tiny scalar `Value` engine in `autograd.py`. Here we connect that mental model to real PyTorch autograd, step by step.

---

## Files In This Folder

| File | Purpose |
| --- | --- |
| `autograd.py` | Tiny scalar autograd engine |
| `compare.py` | Compares our engine with `torch.autograd` |
| `visualize.py` | Prints the computation graph |

Recommended order:

1. Read `README.md`
2. Run `python autograd.py`
3. Run `python compare.py`
4. Run `python visualize.py`
5. Read this file

---

## Learning Path

This file is organized as a sequence of upgrades:

1. Recall what our scalar engine is doing.
2. Move from scalar derivatives to tensor gradients.
3. Inspect the graph PyTorch actually builds.
4. See why gradient accumulation uses `+=`.
5. Intercept gradients with hooks.
6. Write a custom `torch.autograd.Function`.
7. Understand leaves, `retain_grad()`, `detach()`, `no_grad()`, and `inference_mode()`.
8. Compute higher-order gradients.
9. Understand memory and gradient checkpointing.
10. Review the common ways autograd code breaks.

---

## Step 1: Re-anchor On The Scalar Engine

Our engine in `autograd.py` does three things:

1. Each forward operation creates a new `Value`.
2. Each new `Value` stores references to its parents in `_prev`.
3. Each new `Value` stores a `_backward` closure that knows the local derivative rule.

The full backward pass is still just:

1. Build a topological order of the graph.
2. Seed the output gradient with `1.0`.
3. Visit nodes in reverse topological order.
4. Let each node push gradient into its parents.

Core pattern from our engine:

```python
def __mul__(self, other):
    other = other if isinstance(other, Value) else Value(other)
    out = Value(self.data * other.data, _op="*", _prev=(self, other))

    def _backward():
        self.grad += other.data * out.grad
        other.grad += self.data * out.grad

    out._backward = _backward
    return out
```

What matters here is not multiplication specifically. The important pattern is:

1. Forward computes `out.data`.
2. Backward receives `out.grad`.
3. Backward multiplies by local derivatives.
4. Results accumulate into parent gradients.

That exact pattern still exists in PyTorch. The main difference is that PyTorch does it for tensors instead of single floats.

---

## Step 2: Move From Scalars To Tensors

### Step 2.1: Scalar case

For a scalar function `y = f(x)`, the gradient is one number:

```text
x = 3.0
y = x * x = 9.0
dy/dx = 2 * x = 6.0
```

Our engine handles this directly because every node stores:

- one scalar value
- one scalar gradient

### Step 2.2: Tensor case

For tensor-valued functions, the local derivative is no longer a single number.

If `x` has shape `(n,)` and `y` has shape `(m,)`, then the derivative of `y` with respect to `x` is a Jacobian:

```text
J[i, j] = d y[i] / d x[j]
shape(J) = (m, n)
```

Example:

```text
x shape: (3,)
W shape: (2, 3)
y = W @ x
y shape: (2,)
```

Expanded:

```text
y[0] = w00*x0 + w01*x1 + w02*x2
y[1] = w10*x0 + w11*x1 + w12*x2
```

So:

```text
d y[0] / d x = [w00, w01, w02]
d y[1] / d x = [w10, w11, w12]
```

The full Jacobian is:

```text
d y / d x = W
```

### Step 2.3: What PyTorch actually computes

In training, we usually do not need the full Jacobian.

We usually have a scalar loss `L`, and we want:

```text
dL/dx
```

By the chain rule:

```text
dL/dx = (dL/dy) @ (dy/dx)
```

This is a vector-Jacobian product, or VJP.

For `y = W @ x`, PyTorch computes:

```text
grad_y = dL/dy
grad_x = W.T @ grad_y
grad_W = grad_y outer x
```

The key idea:

1. PyTorch does not materialize the full Jacobian in normal backprop.
2. It uses operation-specific backward formulas directly.
3. The graph structure is the same as our engine.
4. Only the math inside each backward rule gets bigger.

Side-by-side:

```text
Our scalar engine:
    self.grad += other.data * out.grad

PyTorch tensor autograd:
    x.grad += W.T @ out.grad
```

Same algorithm. Bigger objects.

---

## Step 3: Inspect The Graph PyTorch Builds

Consider:

```python
import torch

x = torch.tensor([2.0], requires_grad=True)
y = x * 3
z = y.relu()
loss = z.sum()
```

In our toy engine, we would imagine something like:

```text
x -> mul -> relu -> sum
```

In PyTorch, the output tensors point to backward nodes through `grad_fn`:

```python
print(loss.grad_fn)
print(loss.grad_fn.next_functions)
print(z.grad_fn)
print(y.grad_fn)
print(x.grad_fn)
print(x.is_leaf)
```

Typical mental model:

```text
x (leaf tensor)
  -> MulBackward
  -> ReluBackward
  -> SumBackward
```

Map the toy engine to PyTorch like this:

| Our engine | PyTorch |
| --- | --- |
| `_prev` | `grad_fn.next_functions` |
| `_backward` closure | backward node object such as `MulBackward0` |
| `.grad` on every node | `.grad` kept on leaves by default |
| no mutation tracking | version counters catch unsafe in-place edits |

### Step 3.1: Leaf vs non-leaf tensors

A leaf tensor is a tensor you create directly:

```python
x = torch.tensor([2.0], requires_grad=True)
```

A non-leaf tensor is produced by an operation:

```python
y = x * 3
z = y.relu()
```

Why this matters:

1. Leaves usually represent parameters or user-provided inputs.
2. PyTorch stores `.grad` on leaves by default.
3. Non-leaf gradients are normally computed, used, and then discarded.

### Step 3.2: Why in-place ops are dangerous

Suppose you do:

```python
x = torch.tensor([1.0], requires_grad=True)
y = x * 2
z = y * y
y.add_(1)
z.backward()
```

`z = y * y` needs the original `y` during backward.

But `y.add_(1)` changed that saved value in place.

PyTorch tracks this with version counters and raises an error instead of silently returning the wrong gradient.

Our toy engine has no such protection. If we mutated saved data in place, it would just compute garbage.

Rule of thumb:

- In differentiable code, prefer out-of-place ops.
- Treat in-place ops as unsafe unless you know the backward rule does not need the old value.

---

## Step 4: Why Gradient Accumulation Uses `+=`

This is one of the most important details in the whole engine.

### Step 4.1: Single-use node

```python
a = Value(3.0)
b = Value(5.0)
c = a * b
```

Backward contribution:

```python
a.grad += b.data * c.grad
b.grad += a.data * c.grad
```

Here `+=` and `=` happen to give the same result because each parent receives one contribution.

### Step 4.2: Node reused twice

```python
a = Value(3.0)
c = a + a
```

Now `a` appears twice as an input to the same operation.

Backward for addition:

```python
self.grad += out.grad
other.grad += out.grad
```

Since `self` and `other` are the same object, `a.grad` gets hit twice:

```text
a.grad = 1.0 + 1.0 = 2.0
```

That is correct because:

```text
d(a + a)/da = 2
```

If you used `=` instead, the second assignment would overwrite the first one.

### Step 4.3: Branching graph

The same issue appears whenever one value feeds multiple downstream paths.

Example from `autograd.py`:

```python
a = Value(2.0, label="a")
b = a * a
c = Value(3.0, label="c")
d = b.relu()
e = c * a
f = d + e
g = f.tanh()
```

`a` influences the output through multiple routes:

1. through the left input of `b = a * a`
2. through the right input of `b = a * a`
3. through `e = c * a`

The total gradient is the sum of all path contributions.

That is why accumulation is not optional.

### Step 4.4: PyTorch does the same thing

PyTorch accumulates gradients into leaf `.grad` buffers.

That is also why training loops need:

```python
optimizer.zero_grad()
loss.backward()
optimizer.step()
```

Without `zero_grad()`, each new backward pass adds into the previous gradients.

That behavior is sometimes useful for gradient accumulation across micro-batches, but it is a bug if you did not mean to do it.

---

## Step 5: Intercept Gradients With Hooks

Hooks let you observe or modify gradients while backward is running.

### Step 5.1: Tensor hook with `register_hook()`

```python
import torch

x = torch.tensor([2.0], requires_grad=True)

def print_grad(grad):
    print("incoming grad:", grad)

x.register_hook(print_grad)

y = (x * x).sum()
y.backward()
```

What happens:

1. `y.backward()` starts the backward pass.
2. PyTorch computes the gradient flowing into `x`.
3. Your hook receives that gradient tensor.
4. PyTorch then stores the final value in `x.grad`.

You can also modify the gradient:

```python
x = torch.tensor([2.0], requires_grad=True)
x.register_hook(lambda grad: grad * 2)

y = (x * x).sum()
y.backward()
print(x.grad)  # tensor([8.]) instead of tensor([4.])
```

### Step 5.2: Module hook with `register_full_backward_hook()`

```python
def hook_fn(module, grad_input, grad_output):
    print(module.__class__.__name__)
    print("grad_input:", grad_input)
    print("grad_output:", grad_output)

handle = model.layer.register_full_backward_hook(hook_fn)
```

Use this when you want to inspect gradients flowing through an `nn.Module`, not just a single tensor.

### Step 5.3: What the equivalent would be in our engine

Our toy engine has no hook API, but conceptually you could wrap `_backward`:

```python
original_backward = node._backward

def hooked_backward():
    original_backward()
    print(node._label, node.grad)

node._backward = hooked_backward
```

Same idea: run custom code at a specific point in backward.

---

## Step 6: Write A Custom `torch.autograd.Function`

Use a custom `Function` when PyTorch cannot infer the backward you want, or when you want tighter control over saved tensors and backward math.

### Step 6.1: Compare our `relu()` with PyTorch's version

Our toy engine:

```python
def relu(self):
    out = Value(max(0.0, self.data), _op="relu", _prev=(self,))

    def _backward():
        self.grad += (self.data > 0) * out.grad

    out._backward = _backward
    return out
```

PyTorch custom `Function`:

```python
import torch

class MyReLU(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return x.clamp(min=0)

    @staticmethod
    def backward(ctx, grad_output):
        (x,) = ctx.saved_tensors
        grad_input = grad_output * (x > 0).to(grad_output.dtype)
        return grad_input
```

Mapping:

1. Our closure captures what it needs from outer scope.
2. PyTorch saves what it needs explicitly through `ctx`.
3. Our backward mutates parent `.grad` fields directly.
4. PyTorch backward returns gradients in the same order as the forward inputs.

### Step 6.2: Why `ctx.save_for_backward()` exists

In our engine, captured values are tiny Python floats.

In PyTorch, saved objects can be large tensors. PyTorch needs explicit control over what survives until backward so it can free everything else as early as possible.

Save only what backward actually needs.

### Step 6.3: How to call a custom `Function`

```python
x = torch.tensor([-1.0, 2.0], requires_grad=True)
y = MyReLU.apply(x)
loss = y.sum()
loss.backward()
```

### Step 6.4: Minimal custom example

```python
class MySin(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return x.sin()

    @staticmethod
    def backward(ctx, grad_output):
        (x,) = ctx.saved_tensors
        return grad_output * x.cos()
```

When to reach for this:

- custom CUDA or C++ ops
- numerically stable backward formulas
- memory-saving backward implementations
- fused operations where a hand-written backward is better than the default graph

---

## Step 7: Control What Keeps Gradients

This section is where many people get confused because several APIs sound similar but do different jobs.

### Step 7.1: `retain_grad()` for non-leaf tensors

By default:

- leaf tensors keep `.grad`
- non-leaf tensors usually do not

Example:

```python
x = torch.tensor([2.0], requires_grad=True)
y = x * 3
z = y.sum()

z.backward()
print(x.grad)  # tensor([3.])
print(y.grad)  # usually None
```

If you want `y.grad`, request it before backward:

```python
x = torch.tensor([2.0], requires_grad=True)
y = x * 3
y.retain_grad()
z = y.sum()

z.backward()
print(y.grad)  # tensor([1.])
```

Use `retain_grad()` for debugging and inspection, not as a default habit.

### Step 7.2: `detach()` stops gradient flow

```python
y = model(x).detach()
loss = criterion(y, target)
loss.backward()
```

This breaks the graph between `y` and the model that produced it.

After `detach()`:

- `y` shares data with the original tensor
- `y` has no `grad_fn`
- gradients do not flow back into the earlier graph

That is useful when you want a frozen value. It is a bug when you did it accidentally.

### Step 7.3: `torch.no_grad()`

Use `no_grad()` when you want operations inside a block to avoid building an autograd graph:

```python
with torch.no_grad():
    output = model(x)
```

Effects:

1. intermediate ops do not create backward graph nodes
2. outputs usually do not require gradients
3. memory usage drops during inference

### Step 7.4: `torch.inference_mode()`

```python
with torch.inference_mode():
    output = model(x)
```

This is a stronger inference-only mode.

Use it when you are doing pure inference and do not need autograd interaction inside the block.

Simple rule:

- use `no_grad()` when you just want "no graph here"
- use `inference_mode()` when the whole block is true inference work

### Step 7.5: How these ideas map to our toy engine

Our toy engine keeps everything:

- every node keeps `.grad`
- every operation records `_prev`
- every output stores a `_backward`

Real PyTorch is more selective because tensors are large and memory is expensive.

---

## Step 8: Compute Higher-Order Gradients

PyTorch can differentiate through a gradient computation itself.

Example:

```python
import torch

x = torch.tensor([2.0], requires_grad=True)
y = x ** 3
```

First derivative:

```python
dy_dx = torch.autograd.grad(y, x, create_graph=True)[0]
print(dy_dx)  # tensor([12.])
```

Why `create_graph=True` matters:

1. without it, PyTorch computes the gradient and stops there
2. with it, PyTorch builds a graph for the gradient computation too

Second derivative:

```python
d2y_dx2 = torch.autograd.grad(dy_dx, x)[0]
print(d2y_dx2)  # tensor([12.])
```

Math:

```text
y = x^3
dy/dx = 3x^2
d2y/dx2 = 6x
at x = 2: d2y/dx2 = 12
```

Why our toy engine cannot do this yet:

1. our backward closures do plain Python float math
2. those float operations do not build a new computation graph
3. so there is nothing to differentiate through on the second pass

To support higher-order gradients, the backward pass itself would need to create new `Value` nodes instead of raw floats.

---

## Step 9: Understand Memory And Checkpointing

### Step 9.1: Where autograd memory actually goes

For real models, memory is usually dominated by saved activations, not by the graph metadata itself.

Very rough breakdown:

```text
memory during training is roughly:
    parameters
  + saved activations
  + parameter gradients
  + optimizer state
```

Why activations matter:

1. backward needs certain forward-time values
2. those values must stay alive until the matching backward rule runs
3. deep networks therefore keep a lot of intermediate tensors around

Our toy engine hides this because each saved value is just one float.

### Step 9.2: Gradient checkpointing

Checkpointing trades compute for memory.

Normal training:

```text
forward: save many intermediate activations
backward: reuse those saved activations
```

Checkpointed training:

```text
forward: save only selected checkpoints
backward: recompute missing activations when needed
```

Conceptually:

```text
Normal:
    x -> a1 -> a2 -> a3 -> a4 -> loss
    save: a1, a2, a3, a4

Checkpointed:
    x -> a1 -> a2 -> a3 -> a4 -> loss
    save: a1, a4
    recompute: a2, a3 during backward
```

### Step 9.3: What happens step by step

Suppose:

```text
y = W3(W2(W1(x)))
```

Checkpointed view:

1. Forward computes all layers.
2. Only selected activations are kept.
3. Backward starts from the loss.
4. When a missing activation is needed, PyTorch re-runs part of the forward pass to recreate it.
5. Backward then continues as normal.

### Step 9.4: PyTorch API

```python
from torch.utils.checkpoint import checkpoint

y = checkpoint(block, x, use_reentrant=False)
```

Use checkpointing when activation memory is the bottleneck and extra compute is acceptable.

Do not use it blindly:

- it makes training slower
- debugging gets harder
- some code paths interact badly with side effects or mutation

---

## Step 10: Common Ways Autograd Code Breaks

### Bug 1: Replacing `+=` with `=`

Wrong:

```python
self.grad = other.data * out.grad
```

Right:

```python
self.grad += other.data * out.grad
```

Symptom: gradients look correct on simple chains but fail on reused nodes or branching graphs.

### Bug 2: Forgetting to clear gradients between optimizer steps

```python
optimizer.zero_grad()
loss.backward()
optimizer.step()
```

Symptom: gradients and parameter updates grow unexpectedly because values accumulate across batches.

### Bug 3: Unsafe in-place operations

```python
y.add_(1)
```

Symptom: runtime error about a variable needed for gradient computation being modified in place.

Fix: replace with an out-of-place op such as `y = y + 1` unless you are sure the in-place version is safe.

### Bug 4: Accidental `detach()`

```python
features = model(x).detach()
loss = head(features).sum()
loss.backward()
```

Symptom: the upstream model stops receiving gradients.

### Bug 5: Expecting `.grad` on non-leaf tensors

```python
print(y.grad)  # often None
```

Fix: call `y.retain_grad()` before backward if you need to inspect it.

### Debug checklist

When gradients look wrong, inspect these first:

1. `tensor.requires_grad`
2. `tensor.is_leaf`
3. `tensor.grad_fn`
4. whether you called `zero_grad()`
5. whether any `detach()` or `no_grad()` block cuts the graph
6. whether an in-place op changed a saved tensor

Useful tools:

```python
print(loss.grad_fn)
print(loss.grad_fn.next_functions)

with torch.autograd.detect_anomaly():
    loss.backward()
```

---

## Quick Reference

```python
# Compute gradients
loss.backward()
grads = torch.autograd.grad(loss, [x, y])

# Higher-order gradients
g1 = torch.autograd.grad(loss, x, create_graph=True)[0]
g2 = torch.autograd.grad(g1, x)[0]

# Hooks
x.register_hook(lambda grad: print(grad))
handle = module.register_full_backward_hook(hook_fn)

# Non-leaf gradients
y.retain_grad()

# Disable graph building
with torch.no_grad():
    out = model(x)

with torch.inference_mode():
    out = model(x)

# Checkpointing
from torch.utils.checkpoint import checkpoint
out = checkpoint(block, x, use_reentrant=False)
```

---

## Takeaways

1. PyTorch autograd is the same idea as our toy engine: record forward structure, then run local backward rules in reverse.
2. The tensor version uses vector-Jacobian products instead of scalar derivatives.
3. Gradient accumulation with `+=` is mandatory whenever a node can influence the output through multiple paths.
4. PyTorch keeps leaf gradients by default and drops most intermediate gradients to save memory.
5. Hooks, custom `Function`s, and checkpointing are extensions of the same basic graph-and-backward model.
6. Most autograd bugs come from overwritten gradients, stale grads, in-place mutation, or accidentally breaking the graph.

---

## Official References

- PyTorch autograd docs: https://docs.pytorch.org/docs/stable/autograd.html
- Autograd mechanics note: https://docs.pytorch.org/docs/stable/notes/autograd.html
- Beginner autograd tutorial: https://docs.pytorch.org/tutorials/beginner/blitz/autograd_tutorial.html
- Activation checkpointing docs: https://docs.pytorch.org/docs/stable/checkpoint.html
