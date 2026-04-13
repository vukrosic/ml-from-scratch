# Autograd Advanced

> **Before reading this, read [README.md](./README.md) first.** This article assumes you understand the `Value` class, the forward/backward two-pass strategy, and how the chain rule propagates gradients through a computation graph. ADVANCED goes deeper — how PyTorch scales this to tensors, the hook system, custom autograd Functions, gradient accumulation pitfalls, and memory.
>
> **Run the Python files in this directory** to verify everything against PyTorch.

---

## Which File to Run For What

| File | What it demonstrates |
|------|---------------------|
| `autograd.py` | The core engine — `Value` class with forward/backward |
| `compare.py` | Side-by-side gradient comparison against `torch.autograd` |
| `visualize.py` | ASCII computation graph — tree view and flat node view |

---

## From Scalars to Tensors: What Changes

Our engine operates on floats. PyTorch operates on tensors. The core algorithm is identical — the difference is what "gradient" means.

### Scalar case (our engine)

For a scalar `y = f(x)`, the gradient is a single number `dy/dx`.

```
x = 3.0
y = x * x = 9.0
dy/dx = 2 * x = 6.0      ← one number
```

In our engine, `_backward` multiplies the upstream gradient by this one number.

### Vector case (PyTorch)

For a vector `y = f(x)` where `x` has shape `(n,)` and `y` has shape `(m,)`, the gradient is a **Jacobian matrix** of shape `(m, n)`:

```
J[i,j] = d(y_i) / d(x_j)
```

Concrete example — `y = W @ x` where `W` is a 2x3 matrix and `x` is a 3-vector:

```
x = [x0, x1, x2]       ← shape (3,)
W = [[w00, w01, w02],
     [w10, w11, w12]]   ← shape (2, 3)

y = W @ x = [w00*x0 + w01*x1 + w02*x2,
             w10*x0 + w11*x1 + w12*x2]    ← shape (2,)

Jacobian dy/dx has shape (2, 3). Expanding each element:

```
d(y_i)/d(x_j) = w_ij   (the (i,j)-th element of W)
```

So the Jacobian matrix is simply W itself:

```
          ∂y               ∂y
dy/dx = -------   =  [[w00, w01, w02],     =  [[w00, w01, w02],
        ∂x                 [w10, w11, w12]]      [w10, w11, w12]]

                         =  W
```

This is a key insight: for a linear layer `y = W @ x`, the Jacobian `∂y/∂x` is just `W`. PyTorch never materializes this matrix — it uses `W` directly in the VJP.

### The VJP trick

For a scalar loss `L = g(y)`, what we actually want is `dL/dx` — a vector of shape `(n,)`. By the chain rule:

```
dL/dx = (dL/dy) @ J
```

This is a **vector-Jacobian product** (VJP). The "vector" is `dL/dy` (shape `(m,)`), the Jacobian is `J` (shape `(m, n)`), and the result is `dL/dx` (shape `(n,)`).

PyTorch never builds the full Jacobian matrix. For `y = W @ x`:

```
dL/dx = (dL/dy) @ W     ← this is W.T @ (dL/dy), computed directly
```

No `(m, n)` matrix is ever allocated. The VJP function for matrix multiply knows how to compute the result using `W` directly.

```
Scalar autograd (our engine):
  self.grad += other.data * out.grad          ← scalar * scalar

Tensor autograd (PyTorch):
  x.grad += W.T @ out.grad                   ← matrix-vector product (VJP)
```

The topology of the graph is the same. The `_backward` closures just do matrix math instead of scalar math.

---

## The Computation Graph in PyTorch

PyTorch builds the same DAG we build, but with different node types. Let's trace through a concrete example:

```python
x = torch.tensor([2.0], requires_grad=True)
y = x * 3
z = y.relu()
loss = z.sum()
```

In our engine, this graph would look like:

```
Value("x", data=2.0) → Value("*", data=6.0) → Value("relu", data=6.0) → Value("sum", data=6.0)
  ._prev = ()           ._prev = (x, 3)        ._prev = (y,)             ._prev = (z,)
  ._backward = noop     ._backward = mul_bw     ._backward = relu_bw      ._backward = sum_bw
```

In PyTorch, the same graph uses different objects:

```
x (leaf Tensor)  →  MulBackward0  →  ReluBackward0  →  SumBackward0
  .grad_fn = None     .next_functions = [(AccumulateGrad, 0)]
  .requires_grad = True                    │
  .grad = None (until backward)            │
                                   .next_functions = [(MulBackward0, 0)]
```

You can inspect this directly:

```python
print(loss.grad_fn)                    # <SumBackward0 object>
print(loss.grad_fn.next_functions)     # ((ReluBackward0, 0),)
print(z.grad_fn)                       # <ReluBackward0 object>
print(z.grad_fn.next_functions)        # ((MulBackward0, 0),)
print(y.grad_fn)                       # <MulBackward0 object>
print(x.grad_fn)                       # None — x is a leaf
print(x.is_leaf)                       # True
```

Key differences:

| Our engine | PyTorch |
|-----------|---------|
| `Value._backward` closure | `grad_fn` object (e.g., `MulBackward0`) |
| `Value._prev` tuple | `grad_fn.next_functions` tuple |
| `Value.grad` on every node | `.grad` only on leaf tensors (unless `retain_grad()`) |
| No version tracking | Version counter detects in-place mutations |

**Why `.grad` only on leaves?** In a real network, intermediate activations are huge. Storing gradients on every intermediate tensor would double memory usage. PyTorch only stores gradients where you need them — on parameters (leaves) that the optimizer will update. Intermediate gradients are computed, used to push backward, then freed.

**The version counter.** If you modify a tensor in-place after it was used in a computation:

```python
x = torch.tensor([1.0], requires_grad=True)
y = x * 2
x.add_(1)     # in-place modification — increments version counter
y.backward()  # RuntimeError: one of the variables needed for gradient
              # computation has been modified by an inplace operation
```

What happened: `y = x * 2` recorded that `y`'s backward needs `x.data`. But `x.add_(1)` changed `x.data` in place. When `y.backward()` runs, the `x.data` it sees is no longer the value used during the forward pass — the gradient would be wrong. PyTorch catches this with a version counter: `x._version` increments on every in-place op, and backward checks that it has not changed.

Our engine has no such protection — in-place modifications would silently produce wrong gradients.

---

## Gradient Accumulation: The `+=` Problem

In our engine, `_backward` uses `+=` to accumulate gradients. This is not just a convenience — it is mathematically required. Let's trace through exactly why.

### Simple case: value used once

```python
a = Value(3.0)
b = Value(5.0)
c = a * b         # c = 15.0
```

During backward, `c._backward()` runs:

```python
a.grad += b.data * c.grad    # a.grad += 5.0 * 1.0 = 5.0
b.grad += a.data * c.grad    # b.grad += 3.0 * 1.0 = 3.0
```

One path, one contribution. `+=` and `=` give the same result here.

### Problem case: value used twice

```python
a = Value(3.0)
c = a + a          # c = 6.0
```

The graph has two edges from `a` to `c`:

```
a ──(left input)──→ c
a ──(right input)──→ c
```

`c._backward()` runs:

```python
# __add__ backward:
self.grad  += out.grad     # a.grad += 1.0 (first edge: a is self)
other.grad += out.grad     # a.grad += 1.0 (second edge: a is also other)
```

Result: `a.grad = 2.0`. Correct — `d(a+a)/da = 2`.

With `=` instead of `+=`:

```python
self.grad  = out.grad      # a.grad = 1.0
other.grad = out.grad      # a.grad = 1.0 (overwrites!)
```

Result: `a.grad = 1.0`. Wrong — should be 2.

### Complex case: value used in multiple branches

```
        a
       / \
      /   \
   b=a*a  e=c*a
     |      |
   d=relu  (e)
     |      |
      \    /
       f=d+e
         |
       g=tanh(f)
```

`a` appears three times: twice in `b = a*a` (left and right input to multiply) and once in `e = c*a`. The total gradient is the **sum** of contributions from all three paths:

```
dg/da = dg/da via (a as left input to b)      ← a.data appears as other.data in b's backward
      + dg/da via (a as right input to b)     ← a.data appears as self.data in b's backward
      + dg/da via (a as right input to e)     ← c.data appears as other.data in e's backward
```

Each `_backward` call does `a.grad += ...`. Three calls, three contributions, all summed. With `=`, only the last call's contribution survives.

This is mathematically the **multivariate chain rule**: when a variable appears in multiple terms, you sum the partial derivatives from each term.

PyTorch handles this identically — `AccumulateGrad` nodes sum gradients from all incoming paths before storing the result in `tensor.grad`.

---

## Hooks: Intercepting Gradients

PyTorch lets you attach functions that run during the backward pass. There are three types:

### Tensor hooks — `register_hook()`

Fires when the gradient for a specific tensor is computed. The hook function receives one argument: the gradient tensor.

```python
x = torch.tensor([2.0], requires_grad=True)

def print_grad(grad):
    print(f"x.grad flowing through: {grad}")

x.register_hook(print_grad)

y = (x * x).sum()
y.backward()
# prints: x.grad flowing through: tensor([4.])
```

What happens step by step:
1. `y.backward()` starts the backward pass
2. Gradients flow from `y` through `sum`, through `mul`
3. When the gradient for `x` is ready, PyTorch calls `print_grad(tensor([4.]))`
4. Then stores it in `x.grad`

The hook sees the gradient **before** it lands in `x.grad`. You can modify it by returning a new tensor:

```python
def double_grad(grad):
    return grad * 2    # modifies the gradient in-flight

x.register_hook(double_grad)
```

Now `x.grad` will be `tensor([8.])` instead of `tensor([4.])`. The hook intercepted the gradient and doubled it. This is how per-parameter gradient clipping works — the hook caps the gradient magnitude before the optimizer sees it.

### Module hooks — `register_full_backward_hook()`

Fires when gradients flow through a module's backward pass:

```python
def hook_fn(module, grad_input, grad_output):
    print(f"{module.__class__.__name__}: grad_output={grad_output[0].shape}")

model.linear1.register_full_backward_hook(hook_fn)
```

This is how gradient clipping per-layer, gradient logging, and debugging tools work.

### Global hooks — `register_module_full_backward_hook()`

Fires for every module in the network. Useful for gradient norm monitoring across the entire model.

### What hooks look like in our engine

Our engine does not have hooks, but the equivalent would be wrapping `_backward`:

```python
# Pseudo-hook on our Value engine
original_backward = node._backward

def hooked_backward():
    original_backward()
    print(f"Gradient at {node._label}: {node.grad}")

node._backward = hooked_backward
```

---

## Custom Autograd Functions

When PyTorch does not know how to differentiate an operation, you write a custom `torch.autograd.Function`. Let's compare our ReLU with PyTorch's custom Function version, side by side.

**Our engine's relu:**

```python
def relu(self):
    out = Value(max(0.0, self.data), _op="relu", _prev=(self,))

    def _backward():
        self.grad += (self.data > 0) * out.grad

    out._backward = _backward
    return out
```

**PyTorch's custom Function:**

```python
class MyReLU(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)       # save input for backward
        return x.clamp(min=0)

    @staticmethod
    def backward(ctx, grad_output):
        x, = ctx.saved_tensors
        grad_input = grad_output * (x > 0).float()
        return grad_input
```

Line-by-line comparison:

```python
# Ours: the closure captures self.data from the enclosing scope
def _backward():
    self.grad += (self.data > 0) * out.grad

# PyTorch: ctx explicitly saves and retrieves tensors
def forward(ctx, x):
    ctx.save_for_backward(x)       # explicitly save x for later
def backward(ctx, grad_output):
    x, = ctx.saved_tensors         # retrieve x from ctx
```

Why the explicit save? In our engine, `self.data` is a float — 8 bytes. In PyTorch, `x` could be a 100MB tensor. If the closure just captured it implicitly, Python's garbage collector could never free it. `ctx.save_for_backward()` tells PyTorch exactly which tensors need to survive until backward, so everything else can be freed.

```python
# Ours:
(self.data > 0) * out.grad          # scalar bool * scalar float

# PyTorch:
(x > 0).float() * grad_output       # tensor bool → tensor float * tensor float
```

Same math, different scale. `(x > 0)` produces a boolean tensor the same shape as `x`. `.float()` converts `True/False` to `1.0/0.0`. Element-wise multiply with `grad_output` gates each gradient element.

You call the custom Function like this:

```python
# Instead of:  y = x.relu()
# Use:         y = MyReLU.apply(x)
```

When would you write a custom Function? When you have an operation that:
- PyTorch cannot differentiate automatically (custom CUDA kernels, numerical methods)
- You know a more memory-efficient or numerically stable backward than autograd would produce
- You want to fuse forward and backward into a single efficient kernel

---

## `retain_grad()` and Non-Leaf Tensors

In PyTorch, only **leaf tensors** (tensors you create directly, like parameters) keep their `.grad` after `backward()`. Intermediate tensors have their gradients freed to save memory.

What is a leaf tensor? Any tensor you create directly:

```python
x = torch.tensor([2.0], requires_grad=True)   # leaf — you created it
w = torch.randn(10, 5, requires_grad=True)    # leaf — you created it
# model.linear.weight                          # leaf — nn.Module created it
```

What is a non-leaf tensor? Any tensor produced by an operation:

```python
y = x * 3     # non-leaf — produced by multiplication
z = y + 1     # non-leaf — produced by addition
```

After backward:

```python
x = torch.tensor([2.0], requires_grad=True)   # leaf
y = x * 3                                       # non-leaf, y.data = 6.0
z = y.sum()                                      # non-leaf, z.data = 6.0

z.backward()
print(x.grad)   # tensor([3.]) — works, x is a leaf
print(y.grad)   # None — freed! y is non-leaf
```

Why? In a network with 100 layers, storing gradients on all 100 intermediate activations would double memory usage. You only need gradients on parameters (leaves) to update them. Intermediate gradients are computed, used to push backward to the next layer, then freed.

To keep gradients on intermediate tensors, call `retain_grad()` **before** backward:

```python
y.retain_grad()    # tell PyTorch to keep y's gradient
z.backward()
print(y.grad)      # tensor([1.]) — now available
```

Step by step what happens during backward:
1. `z.grad = 1.0` (seed)
2. `z._backward()` computes `y.grad = 1.0` (sum backward)
3. Normally PyTorch would free `y.grad` here — but `retain_grad()` prevents that
4. `y._backward()` computes `x.grad = 3.0 * 1.0 = 3.0` (mul backward)
5. `x.grad = 3.0` is kept because `x` is a leaf

Our engine keeps gradients on every node because we have no memory pressure — every `Value.grad` persists. `compare.py` uses `retain_grad()` on every intermediate tensor so it can read their gradients and compare against our engine.

---

## `no_grad()` and Inference Mode

During inference, you do not need gradients. Building the computation graph wastes memory and time. PyTorch provides ways to skip it.

**Without `no_grad()` — graph is built (wasteful during inference):**

```python
output = model(x)
# PyTorch records every operation:
#   linear1 → grad_fn=AddmmBackward0
#   relu → grad_fn=ReluBackward0
#   linear2 → grad_fn=AddmmBackward0
# All these grad_fn objects and saved tensors sit in memory — for nothing
```

**With `no_grad()` — no graph:**

```python
with torch.no_grad():
    output = model(x)
# No grad_fn objects created. No tensors saved for backward.
# output.requires_grad = False
# Memory usage drops significantly for large models.
```

You can also use it as a decorator:

```python
@torch.no_grad()
def predict(model, x):
    return model(x)
```

**`inference_mode()` — even faster:**

```python
with torch.inference_mode():
    output = model(x)
# No graph, no version tracking, no autograd metadata at all.
```

`inference_mode` is faster because it also disables version counting (the mechanism that catches in-place modifications). Use `inference_mode` for pure inference. Use `no_grad()` when you need to mix grad and no-grad operations in the same scope.

In our engine, the equivalent would be creating Value objects that never set `_backward` — the `_prev` tuples would be empty, so `backward()` would have no edges to traverse.

---

## Common Bugs and Debugging

### Bug 1: Forgetting `+=` (gradient overwrite)

```python
# Wrong:
self.grad = other.data * out.grad

# Right:
self.grad += other.data * out.grad
```

**Symptom**: gradients are wrong when a value is used in more than one place. Only the last path contributes.

### Bug 2: Missing `zero_grad()` between iterations

```python
# PyTorch accumulates gradients across backward() calls
optimizer.zero_grad()     # clear previous gradients
loss.backward()           # accumulate new gradients
optimizer.step()           # update parameters
```

Without `zero_grad()`, gradients from the previous batch add to the current batch:

```python
# Iteration 1:
loss1.backward()          # w.grad = 0.5

# Iteration 2 — WITHOUT zero_grad:
loss2.backward()          # w.grad = 0.5 + 0.3 = 0.8  ← accumulated!
optimizer.step()          # updates using 0.8 instead of 0.3

# Iteration 2 — WITH zero_grad:
optimizer.zero_grad()     # w.grad = 0.0
loss2.backward()          # w.grad = 0.3               ← correct
optimizer.step()          # updates using 0.3
```

Why does PyTorch accumulate by default? Because **gradient accumulation** is a real technique: when your GPU cannot fit a large batch, you run several small batches and accumulate their gradients before stepping. Calling `zero_grad()` only every N batches gives you effectively N times the batch size. But if you always want fresh gradients each step, you must call `zero_grad()` explicitly.

### Bug 3: In-place operations breaking the graph

```python
x = torch.tensor([1.0], requires_grad=True)
y = x * 2       # y's backward needs x.data (which is currently 1.0)
x += 1           # in-place: x.data is now 2.0, but y's backward still expects 1.0
y.backward()     # RuntimeError!
```

What went wrong: `y = x * 2` recorded that `y._backward` needs `x.data`. The `mul` backward computes `x.grad = 2 * y.grad` — the `2` comes from `other.data`, but `x.data` is used for the other direction. When `x += 1` changes `x.data` in place, the value that `y._backward` will read is no longer the value from the forward pass.

**Fix**: use `x = x + 1` (creates a new tensor) instead of `x += 1` (modifies in place).

```python
x = torch.tensor([1.0], requires_grad=True)
y = x * 2
x = x + 1     # creates NEW tensor, original x is untouched
y.backward()  # works — y's backward still sees the original x.data = 1.0
```

### Bug 4: Detaching when you should not

```python
y = model(x).detach()   # disconnects y from the graph
loss = criterion(y, target)
loss.backward()          # gradients stop at y — model gets no updates
```

What `detach()` does: creates a new tensor that shares the same data but has `grad_fn = None` and `requires_grad = False`. The backward pass hits `y` and stops — it sees no graph behind it.

```
With detach:    loss → criterion → y (detached, no grad_fn) ← backward stops here
Without detach: loss → criterion → y → model layers → parameters ← backward reaches parameters
```

Use `detach()` intentionally when you want to stop gradients (e.g., target networks in RL, where you want to compute a loss against a frozen copy). But do not use it accidentally in the middle of a training pipeline.

### Debugging tools

```python
# Print the backward graph
print(loss.grad_fn)                    # the last operation
print(loss.grad_fn.next_functions)     # its inputs

# Check if a tensor requires grad
print(x.requires_grad)

# Check if a tensor is a leaf
print(x.is_leaf)

# Detect anomalies (NaN/Inf in gradients)
with torch.autograd.detect_anomaly():
    loss.backward()
```

---

## Higher-Order Gradients

PyTorch can differentiate through the backward pass itself — gradients of gradients. Let's trace through a concrete example:

```python
x = torch.tensor([2.0], requires_grad=True)
y = x ** 3          # y = x^3 = 8.0
```

**First derivative:**

```python
dy_dx = torch.autograd.grad(y, x, create_graph=True)[0]
# dy/dx = 3x^2 = 3 * 4 = 12.0
```

`torch.autograd.grad` computes the gradient without storing it in `x.grad`. The key is `create_graph=True` — this tells PyTorch: "build a computation graph for the gradient computation itself." Without it, `dy_dx` would be a plain tensor with no graph attached.

With `create_graph=True`, `dy_dx` is a tensor that knows it came from `3 * x * x`. It has its own `grad_fn`.

**Second derivative:**

```python
d2y_dx2 = torch.autograd.grad(dy_dx, x)[0]
# d^2y/dx^2 = 6x = 6 * 2 = 12.0
print(d2y_dx2)      # tensor([12.])
```

This differentiates `dy_dx = 3x^2` with respect to `x`, giving `6x = 12`. It works because `dy_dx` has a computation graph (thanks to `create_graph=True`), so PyTorch can run backward through it again.

```
Normal backward:
  y = x^3
  backward computes: dy/dx = 3x^2    ← result has no graph, cannot differentiate further

create_graph=True backward:
  y = x^3
  backward computes: dy/dx = 3x^2    ← result HAS a graph (3 * x * x)
  second backward:   d^2y/dx^2 = 6x  ← differentiates through the first backward's graph
```

This is used in:

- **Meta-learning** (MAML) — gradients through the optimization step: you need `d(loss_test) / d(theta_initial)`, which requires differentiating through `theta_updated = theta_initial - lr * grad`
- **Regularization** — penalizing gradient magnitude: `loss += lambda * grad.norm()` requires the gradient of a gradient
- **Physics-informed neural networks** — enforcing PDE constraints like `d^2u/dx^2 + d^2u/dy^2 = 0` requires second derivatives

Our engine cannot do this because `_backward` closures use plain Python arithmetic (`float` operations) that do not build a Value graph. To support higher-order gradients, each `_backward` closure would need to create new Value nodes for its own computations — the backward pass would itself be recorded on the tape.

**What would need to change in our engine?** Today, `backward()` does this:

```python
self.grad = 1.0
for node in reversed(order):
    node._backward()   # just Python math — result is a float, not a Value
```

To support higher-order gradients, `_backward()` would need to **record** its computation as a new Value node, not just do arithmetic:

```python
# Hypothetical: backward that builds a graph
def _backward():
    # Instead of: self.grad += other.data * out.grad  (plain float)
    # We would do:
    grad_value = other.data * out.grad    # creates a NEW Value node
    self.grad_node = self.grad_node + grad_value   # records this op in the tape
```

Then calling `backward()` on the gradient *itself* would replay *that* tape. The backward pass becomes a forward pass — you need a tape of the backward pass to differentiate through it. PyTorch handles this with `create_graph=True` in `torch.autograd.grad`.

---

## Memory: Where Autograd Bytes Go

For a model with `N` parameters and `L` layers:

```
Forward pass stores:
  - Activations at each layer: O(L * batch_size * hidden_dim)
  - The computation graph nodes: O(num_operations)

Backward pass stores:
  - Gradients: same shape as activations
  - Parameter gradients: same shape as parameters

Peak memory ≈ parameters + activations + gradients
           ≈ N + 2 * (L * B * H)
```

For a 1B parameter model with batch size 32 and 32 layers of hidden dim 4096:

```
Parameters:  ~4 GB  (1B * 4 bytes FP32)
Activations: ~16 GB (32 layers * 32 batch * 4096 dim * 4 bytes)
Gradients:   ~16 GB (same as activations)
Param grads: ~4 GB  (same as parameters)
Total:       ~40 GB
```

This is why gradient checkpointing, mixed precision (FP16 halves activation memory), and activation offloading exist. The autograd graph itself is small — the activations it holds references to are large.

---

## Gradient Checkpointing: Trading Compute for Memory

In a deep network, the forward pass stores activations at every layer for the backward pass. For a 100-layer network, that is 100 sets of activations in memory simultaneously.

**Gradient checkpointing** drops intermediate activations and recomputes them during backward:

```
Normal (high memory):
  Forward:   L1 → L2 → L3 → L4 → L5 → loss
  Stored:    [a1] [a2] [a3] [a4] [a5]         ← all in memory

Checkpointed (low memory):
  Forward:   L1 → L2 → L3 → L4 → L5 → loss
  Stored:    [a1]       [a3]       [a5]        ← only checkpoints kept
  Backward:  recompute a2 from a1, recompute a4 from a3
```

Memory drops from O(n) to O(sqrt(n)) layers. The cost: each non-checkpointed layer is computed twice — once in forward, once in backward.

In PyTorch:

```python
from torch.utils.checkpoint import checkpoint

# Instead of:  y = self.block(x)
# Use:         y = checkpoint(self.block, x, use_reentrant=False)
```

Our tiny engine has no memory pressure — everything is a single float. But the principle is the same: you can drop intermediate values and recompute them from checkpoints.

---

## Quick Reference

```python
# ============================================================
# PyTorch autograd essentials
# ============================================================

# Compute gradients
loss.backward()                          # populates .grad on all leaves
torch.autograd.grad(loss, [x, y])        # returns gradients without populating .grad

# Higher-order gradients
g = torch.autograd.grad(y, x, create_graph=True)[0]
g2 = torch.autograd.grad(g, x)[0]       # second derivative

# Disable gradient tracking
with torch.no_grad(): ...               # no graph, still allows version checks
with torch.inference_mode(): ...         # no graph, no version checks (fastest)
x.detach()                               # new tensor, no grad_fn

# Inspect the graph
loss.grad_fn                             # last operation
loss.grad_fn.next_functions              # inputs to last operation
x.requires_grad                          # does this tensor track gradients?
x.is_leaf                                # is this a leaf tensor?

# Hooks
x.register_hook(lambda g: print(g))      # fires when x.grad is computed
module.register_full_backward_hook(fn)   # fires when gradients flow through module

# Keep gradients on non-leaf tensors
y.retain_grad()

# Detect NaN/Inf in gradients
with torch.autograd.detect_anomaly():
    loss.backward()

# Gradient checkpointing
from torch.utils.checkpoint import checkpoint
y = checkpoint(block, x, use_reentrant=False)

# Clear gradients between training steps
optimizer.zero_grad()
```

---

## Takeaways

- **Scalar autograd and tensor autograd are the same algorithm.** The only difference is VJPs (vector-Jacobian products) instead of scalar derivatives.
- **`+=` is not optional.** Multi-use nodes require gradient accumulation from all paths.
- **Hooks let you intercept, log, and modify gradients mid-flight.** Use them for debugging and gradient clipping.
- **Custom Functions define forward and backward explicitly.** Use them when PyTorch does not know how to differentiate your operation.
- **Gradient checkpointing trades compute for memory.** O(sqrt(n)) memory instead of O(n).
- **`no_grad()` and `inference_mode()` skip graph construction.** Always use them during inference.
- **In-place operations break the graph.** Prefer out-of-place operations in differentiable code.
- **Memory cost is dominated by activations, not the graph itself.** Reducing activation memory (checkpointing, mixed precision, offloading) is where the wins are.

Sources:
- https://pytorch.org/docs/stable/autograd.html
- https://pytorch.org/docs/stable/notes/autograd.html
- https://pytorch.org/tutorials/beginner/blitz/autograd_tutorial.html
