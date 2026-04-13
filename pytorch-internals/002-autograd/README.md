# Autograd From Scratch

When you call `loss.backward()` in PyTorch, gradients appear on every parameter. There is no magic — just two passes and the chain rule. The forward pass records a **tape** of every operation. The backward pass replays that tape in reverse, pushing gradients from the output back to the inputs.

This lesson builds that engine from scratch in ~100 lines of Python. By the end you will understand exactly what `torch.autograd` does internally.

---

## The Problem: Why Do We Need Autograd?

Training a neural network means adjusting parameters to minimize a loss function. The adjustment uses gradients — partial derivatives of the loss with respect to each parameter. For a network with millions of parameters, computing gradients by hand is impossible.

Automatic differentiation (autograd) breaks every computation into primitive operations (`+`, `*`, `relu`, `tanh`, ...) whose derivatives are known, then composes them using the chain rule:

```
loss = tanh(relu(a^2) + 3a)

Chain rule:  d(loss)/d(a)  =  d(loss)/d(f) * d(f)/d(a)
                            =  d(tanh)/d(f) * [d(relu(a^2))/d(a) + d(3a)/d(a)]
                            =  (1 - tanh^2(f)) * [2a * 1{a^2>0} + 3]
```

No matter how deep or wide the network, the same two-pass strategy applies: record forward, replay backward.

---

## Value: The Atom of Autograd

Every scalar in our engine is a `Value` object. It stores five things:

| Field | What it holds |
|-------|--------------|
| `data` | The current scalar value (a float) |
| `grad` | The accumulated gradient `d(loss)/d(this node)` — starts at 0.0 |
| `_op` | Which operation produced this node (`+`, `*`, `relu`, `tanh`) |
| `_prev` | The input Value nodes this one depends on — these form the graph edges |
| `_backward` | A closure that pushes gradients from this node to `_prev` via the chain rule |

```python
class Value:
    def __init__(self, data, _op=None, _prev=(), label=None):
        self.data = float(data)
        self.grad = 0.0
        self._op = _op
        self._prev = _prev
        self._backward = lambda: None
        self._label = label or f"v{id(self)}"
```

Line by line:

```python
self.data = float(data)
```
The actual number. When you write `Value(3.0)`, `data` is `3.0`.

```python
self.grad = 0.0
```
The gradient starts at zero. It will be filled in later when `backward()` runs.

```python
self._op = _op
```
A string like `"+"` or `"*"` — tells us what operation created this node. Leaf nodes (inputs you create directly) have `_op=None`.

```python
self._prev = _prev
```
A tuple of the input Values. For `c = a + b`, `c._prev` is `(a, b)`. For a leaf node, `_prev` is empty `()`. These pointers are the edges of the computation graph.

```python
self._backward = lambda: None
```
A function that does nothing — for now. Each operation (`__add__`, `__mul__`, `relu`, `tanh`) will replace this with a closure that knows how to push gradients backward. Leaf nodes keep the no-op because there is nothing behind them.

```python
self._label = label or f"v{id(self)}"
```
A human-readable name for debugging and visualization. You can pass `label="a"` or let it auto-generate.

The `_prev` edges form the **computation graph** — a directed acyclic graph (DAG) where each node is an operation and each edge is a data dependency. The graph is built implicitly as you write normal Python expressions. When you write `c = a + b`, Python calls `a.__add__(b)`, which creates a new Value with `_prev=(a, b)` — no explicit graph-building step needed.

---

## Addition and Multiplication

### Addition

When you compute `out = a + b`, the derivative of the output with respect to either input is 1:

```
out = a + b
d(out)/d(a) = 1       ← changing a by 1 changes out by 1
d(out)/d(b) = 1       ← changing b by 1 changes out by 1
```

Here is the full `__add__` method:

```python
def __add__(self, other):
    other = other if isinstance(other, Value) else Value(other)
    out = Value(self.data + other.data, _op="+", _prev=(self, other))

    def _backward():
        self.grad  += out.grad
        other.grad += out.grad

    out._backward = _backward
    return out
```

Step by step:

```python
other = other if isinstance(other, Value) else Value(other)
```
If someone writes `a + 5`, the `5` is a plain int. This line wraps it in a `Value(5)` so the rest of the code can treat it uniformly.

```python
out = Value(self.data + other.data, _op="+", _prev=(self, other))
```
Create the result node. If `a.data = 2.0` and `b.data = 3.0`, then `out.data = 5.0`. The `_prev=(self, other)` links this node back to `a` and `b` in the graph.

```python
def _backward():
    self.grad  += out.grad
    other.grad += out.grad
```
This closure is called during `backward()`. It takes the gradient that has flowed into `out` and passes it straight through to both inputs. For addition, the local derivative is 1, so each input gets `1 * out.grad = out.grad`.

Concrete example: if `out.grad = 0.7` (meaning the loss changes by 0.7 when `out` changes by 1), then both `a.grad` and `b.grad` get `+= 0.7`.

```python
out._backward = _backward
return out
```
Attach the backward function to the output node and return it.

**Why `+=` and not `=`?** If a Value is used in multiple places (e.g., `a + a`), its gradient must **accumulate** contributions from every path. Using `=` would overwrite the first contribution:

```
a + a:
  path 1: a is left input  → a.grad += out.grad    (first contribution)
  path 2: a is right input → a.grad += out.grad    (second contribution)
  total: a.grad = 2 * out.grad                      (correct: d(a+a)/da = 2)

If we used = instead of +=:
  path 1: a.grad = out.grad     (set to out.grad)
  path 2: a.grad = out.grad     (overwrites — still out.grad)
  total: a.grad = out.grad                           (wrong: should be 2*out.grad)
```

### Multiplication

The product rule gives us cross-derivatives — each input's gradient depends on the **other** input's value:

```
out = a * b
d(out)/d(a) = b       ← changing a by 1 changes out by b
d(out)/d(b) = a       ← changing b by 1 changes out by a
```

This makes intuitive sense: if `a = 3` and `b = 5`, then `out = 15`. If you increase `a` by 1 (to 4), `out` becomes `4 * 5 = 20` — it changed by 5, which is `b`.

```python
def __mul__(self, other):
    other = other if isinstance(other, Value) else Value(other)
    out = Value(self.data * other.data, _op="*", _prev=(self, other))

    def _backward():
        self.grad  += other.data * out.grad
        other.grad += self.data  * out.grad

    out._backward = _backward
    return out
```

The key lines are in `_backward`:

```python
self.grad  += other.data * out.grad
```
The gradient for `self` is `d(out)/d(self) * out.grad = other.data * out.grad`. This is the chain rule: multiply the local derivative (`other.data`) by the upstream gradient (`out.grad`).

```python
other.grad += self.data  * out.grad
```
Same logic, swapped: the gradient for `other` uses `self.data` as the local derivative.

Concrete example with `a=3.0, b=5.0`:

```
Forward:  out = a * b = 3.0 * 5.0 = 15.0

Backward (say out.grad = 0.4):
  a.grad += b.data * out.grad = 5.0 * 0.4 = 2.0
  b.grad += a.data * out.grad = 3.0 * 0.4 = 1.2
```

Why `out.grad` and not just the local derivative? Because `out.grad` carries `d(loss)/d(out)` — the gradient from everything downstream. We multiply it by `d(out)/d(a)` to get `d(loss)/d(a)`. This is the chain rule chaining through the entire graph.

---

## Activation Functions: ReLU and Tanh

### ReLU

`relu(x) = max(0, x)`. It passes positive values through unchanged, and kills negative values to zero.

```python
def relu(self):
    out = Value(max(0.0, self.data), _op="relu", _prev=(self,))

    def _backward():
        self.grad += (self.data > 0) * out.grad

    out._backward = _backward
    return out
```

Line by line:

```python
out = Value(max(0.0, self.data), _op="relu", _prev=(self,))
```
Forward pass: if `self.data = 3.0`, out is `3.0`. If `self.data = -2.0`, out is `0.0`.

```python
self.grad += (self.data > 0) * out.grad
```
The derivative of ReLU is a gate: 1 if input was positive, 0 if input was negative. `(self.data > 0)` evaluates to `True` (which Python treats as `1`) or `False` (`0`). Multiplying by `out.grad` either passes the gradient through or kills it.

```
Example 1 — positive input:
  Forward:   self.data = 3.0  →  out.data = max(0, 3.0) = 3.0
  Backward:  (3.0 > 0) = True = 1
             self.grad += 1 * out.grad = out.grad    (gradient passes through)

Example 2 — negative input:
  Forward:   self.data = -2.0  →  out.data = max(0, -2.0) = 0.0
  Backward:  (-2.0 > 0) = False = 0
             self.grad += 0 * out.grad = 0.0          (gradient is killed)
```

This is why "dead neurons" happen in ReLU networks — once a neuron's input is always negative, its gradient is always zero, so it never updates.

### Tanh

`tanh(x)` squashes values to the range `[-1, 1]`. Its derivative is `1 - tanh(x)^2`.

```python
def tanh(self):
    t = math.tanh(self.data)
    out = Value(t, _op="tanh", _prev=(self,))

    def _backward():
        self.grad += (1 - t * t) * out.grad

    out._backward = _backward
    return out
```

Line by line:

```python
t = math.tanh(self.data)
```
Compute `tanh` and save it in `t`. We save it because we need it twice: once for the forward value, once for the derivative.

```python
out = Value(t, _op="tanh", _prev=(self,))
```
Forward pass: `out.data = tanh(self.data)`.

```python
self.grad += (1 - t * t) * out.grad
```
The derivative of `tanh(x)` is `1 - tanh(x)^2`. Near zero, `tanh(0) = 0`, so the derivative is `1 - 0 = 1` (full pass-through). At extremes like `tanh(10) ≈ 1.0`, the derivative is `1 - 1 = 0` (gradient vanishes). This is the **vanishing gradient problem** — deep networks with tanh activations struggle because gradients shrink to near-zero as they flow backward through many layers.

```
Example — small input:
  self.data = 0.5  →  t = tanh(0.5) ≈ 0.462
  derivative = 1 - 0.462^2 = 1 - 0.214 = 0.786    (most gradient passes through)

Example — large input:
  self.data = 5.0  →  t = tanh(5.0) ≈ 0.9999
  derivative = 1 - 0.9999^2 ≈ 0.0002              (almost no gradient passes through)
```

---

## The Backward Pass: Replaying the Tape

The forward pass builds the graph by linking each new Value to its inputs. The backward pass needs to visit nodes in the right order: a node's gradient must be fully accumulated before it pushes gradients to its inputs.

This order is **reverse topological** — children before parents. The algorithm:

1. Topologically sort the graph (depth-first, post-order)
2. Seed the output gradient to 1.0 (for a scalar loss, `d(loss)/d(loss) = 1`)
3. Walk the sorted list in reverse, calling `_backward()` on each node

```python
def backward(self):
    visited = set()
    order = []

    def build_order(v):
        if v in visited:
            return
        visited.add(v)
        for child in v._prev:
            build_order(child)
        order.append(v)

    build_order(self)

    self.grad = 1.0
    for node in reversed(order):
        node._backward()
```

Line by line:

```python
visited = set()
order = []
```
`visited` tracks which nodes we have already seen (to avoid processing a node twice). `order` will hold all nodes in topological order.

```python
def build_order(v):
    if v in visited:
        return
    visited.add(v)
    for child in v._prev:
        build_order(child)
    order.append(v)
```
Recursive depth-first traversal. For each node: mark it visited, recurse into all its inputs first, then append self. This puts inputs before outputs in `order` — so when we reverse it, outputs come first.

```python
build_order(self)
```
Start from the output node (e.g., the loss). This visits every node reachable from the output.

```python
self.grad = 1.0
```
Seed gradient. `d(loss)/d(loss) = 1` — the loss changes by 1 when the loss changes by 1. This is the starting point for all gradient computation.

```python
for node in reversed(order):
    node._backward()
```
Walk from output to inputs. Each `_backward()` call pushes gradients from this node into its inputs. Because we go in reverse topological order, by the time we reach a node, all nodes that depend on it have already pushed their gradients into it.

### Full Worked Example

Here is the complete flow for `g = tanh(relu(a^2) + 3a)` with `a=2.0, c=3.0`:

**Step 1 — Forward pass builds the graph:**

```python
a = Value(2.0, label="a")           # leaf node, data=2.0
b = a * a                           # b.data = 2.0 * 2.0 = 4.0, _prev=(a, a)
c = Value(3.0, label="c")           # leaf node, data=3.0
d = b.relu()                        # d.data = max(0, 4.0) = 4.0, _prev=(b,)
e = c * a                           # e.data = 3.0 * 2.0 = 6.0, _prev=(c, a)
f = d + e                           # f.data = 4.0 + 6.0 = 10.0, _prev=(d, e)
g = f.tanh()                        # g.data = tanh(10.0) ≈ 1.0, _prev=(f,)
```

The graph looks like this:

```
        a (2.0)
       / \    \
      /   \    \
   b=a*a   \  e=c*a
   (4.0)    |  (6.0)
     |      |    |
   d=relu   |    |
   (4.0)    |    |
     |      |   /
     +------+--+
     f = d + e
     (10.0)
       |
     g = tanh(f)
     (≈1.0)
```

**Step 2 — Topological sort:**

`build_order(g)` visits: `a → b → d → c → e → f → g`

**Step 3 — Backward pass (reversed: g → f → e → c → d → b → a):**

```
Step 3a: g.grad = 1.0 (seed)

Step 3b: g._backward() — tanh backward
  t = tanh(10.0) ≈ 0.99999999587
  f.grad += (1 - t*t) * g.grad
  f.grad += (1 - 0.9999...^2) * 1.0
  f.grad ≈ 0.0000000827                  ← very small because tanh is saturated

Step 3c: f._backward() — add backward
  d.grad += f.grad ≈ 0.0000000827        ← gradient passes straight through
  e.grad += f.grad ≈ 0.0000000827        ← same gradient to both inputs

Step 3d: e._backward() — mul backward (e = c * a)
  c.grad += a.data * e.grad              ← c.grad += 2.0 * 0.0000000827
  a.grad += c.data * e.grad              ← a.grad += 3.0 * 0.0000000827 (first contribution to a)

Step 3e: d._backward() — relu backward
  b.grad += (b_input.data > 0) * d.grad
  b.grad += (4.0 > 0) * 0.0000000827
  b.grad += 1 * 0.0000000827             ← positive input, gradient passes through

Step 3f: b._backward() — mul backward (b = a * a)
  a.grad += a.data * b.grad              ← a.grad += 2.0 * 0.0000000827 (second contribution)
  a.grad += a.data * b.grad              ← a.grad += 2.0 * 0.0000000827 (third contribution — a appears twice in a*a)
```

Notice three things:
1. `a.grad` gets **three** `+=` contributions because `a` appears in three places in the graph (twice in `a*a`, once in `c*a`)
2. The gradients are tiny because `tanh(10)` is saturated — this is the vanishing gradient problem in action
3. Every step is just the chain rule: `local_derivative * upstream_gradient`

---

## Running It

```python
python autograd.py
```

```
Individual gradients:
  dg/da = ...    # the full derivative through both paths
  dg/dc = ...    # derivative through the c*a branch only
  g     = ...    # the output value
```

---

## Visualizing the Graph

`visualize.py` draws the computation graph as ASCII art — both a tree view and a flat node view showing data and gradient values at each node.

```bash
python visualize.py
```

The tree view shows the structure of operations:

```
├── g [tanh]
│   └── f [+]
│       ├── d [relu]
│       │   └── b [*]
│       │       ├── a
│       │       └── a
│       └── e [*]
│           ├── c
│           └── a
```

---

## Verifying Against PyTorch

`compare.py` runs the same computation graph with both our engine and `torch.autograd`, then prints a side-by-side comparison across multiple test cases.

```bash
python compare.py
```

```
=======================================================
  a=2.0, b=3.0, c=4.0
=======================================================
  Node   Our grad   PyTorch grad    Match
  ------ ---------- -------------- --------
  a        ...          ...           OK
  b        ...          ...           OK
  ...
```

Every gradient should match PyTorch to 1e-6 tolerance. If you see a `FAIL`, the implementation is wrong.

---

## What This Engine Does NOT Do

This is a scalar engine — it operates on single floats, not tensors. PyTorch's real autograd handles:

- **Tensors** — gradients are matrices (Jacobians), not scalars
- **In-place operations** — version tracking to detect mutations
- **Hooks** — `register_hook()` to inspect or modify gradients mid-flight
- **Custom Functions** — `torch.autograd.Function` with explicit `forward`/`backward`
- **Gradient checkpointing** — trading compute for memory in deep networks
- **Higher-order gradients** — gradients of gradients for meta-learning

These are covered in [ADVANCED.md](./ADVANCED.md).

---

## Recap

| Concept | How it works |
|---------|-------------|
| `Value` node | Stores `data`, `grad`, the operation, and pointers to inputs |
| Forward pass | Normal Python math — each operation records itself in the graph |
| `_backward` closure | Each operation defines how to push gradients to its inputs |
| `+=` accumulation | Nodes used in multiple places get gradient contributions from all paths |
| Topological sort | Ensures gradients are fully computed before propagating further |
| Chain rule | `d(loss)/d(x) = d(loss)/d(out) * d(out)/d(x)` — applied at every node |

The same two-pass strategy (forward record, backward replay) is exactly what powers `torch.autograd` — just scaled to tensors and thousands of operation types.

---

Get the video walkthrough of Jacobian derivation, PyTorch hook system internals, custom autograd Functions, and profiling: [OpenSuperintelligenceLab on Skool](https://www.skool.com/opensuperintelligencelab)
