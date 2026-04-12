# Autograd From Scratch

When you call `loss.backward()` in PyTorch, gradients for every parameter appear like magic. But the magic has a simple core: a **tape** that records every operation in the forward pass, and a **reversed tape** that replays them to compute gradients. No magic — just two passes and the chain rule.

This lesson builds that engine from scratch. You will see exactly what happens inside `torch.autograd`.

---

## The Problem: Why Do We Need Autograd?

Training a neural network means updating parameters to minimize a loss. The update rule needs the gradient of the loss with respect to every parameter. For a network with millions of parameters, computing those gradients by hand is impossible.

Automatic differentiation (autograd) solves this by breaking every complex function into a sequence of primitive operations (`+`, `*`, `relu`, `tanh`, ...) whose derivatives are known. The chain rule composes them.

---

## Value: The Atom of Autograd

Every scalar in our engine is a `Value` object. It stores:

- `data` — the current value
- `grad` — the accumulated gradient `d(loss)/d(this node)`
- `_op` — what operation produced this node (`+`, `*`, `relu`, `tanh`)
- `_prev` — the input Value nodes this one depends on
- `_backward` — a closure that pushes gradients to `_prev`

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

This tiny structure is all we need. The `_prev` edges form the **computation graph**.

---

## Addition and Multiplication

When you add two Values, the output depends on both inputs. The gradient flows straight through:

```
d(out)/d(a) = 1
d(out)/d(b) = 1
```

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

Multiplication is slightly more interesting. The product rule gives us:

```
d(out)/d(a) = b
d(out)/d(b) = a
```

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

---

## Activation Functions: ReLU and Tanh

ReLU is `max(0, x)`. Its derivative is `1` if `x > 0`, else `0`.

```python
def relu(self):
    out = Value(max(0.0, self.data), _op="relu", _prev=(self,))

    def _backward():
        self.grad += (self.data > 0) * out.grad

    out._backward = _backward
    return out
```

Tanh's derivative is `1 - tanh²(x)`.

```python
def tanh(self):
    t = math.tanh(self.data)
    out = Value(t, _op="tanh", _prev=(self,))

    def _backward():
        self.grad += (1 - t * t) * out.grad

    out._backward = _backward
    return out
```

---

## The Backward Pass: Replaying the Tape

The forward pass builds the graph by linking each new Value to its inputs. The backward pass traverses that graph in **reverse topological order** — children before parents — and calls each `_backward` closure to accumulate gradients.

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

    # Seed gradient — for a scalar loss this is 1.0
    self.grad = 1.0
    for node in reversed(order):
        node._backward()
```

The topological sort ensures we never use a gradient before it has been fully computed by its children.

---

## Running It

```python
if __name__ == "__main__":
    a = Value(2.0, label="a")
    b = a * a           # b = a²
    c = Value(3.0, label="c")
    d = b.relu()        # d = relu(a²)
    e = c * a           # e = 3a
    f = d + e           # f = relu(a²) + 3a
    g = f.tanh()        # g = tanh(f)

    g.backward()

    print(f"dg/da = {a.grad:.4f}")   # ≈ 1.151
    print(f"g     = {g.data:.4f}")   # ≈ 0.864
```

---

## Visualizing the Graph

The `visualize.py` script draws the computation graph as ASCII art so you can trace exactly how gradients flow.

```bash
python visualize.py
```

Output shows the tree structure of operations and their data/gradient values at each node.

---

## Verifying Against PyTorch

The `compare.py` script runs the same computation graph with both our engine and `torch.autograd`, then prints a side-by-side comparison.

```bash
python compare.py
```

Every gradient should match PyTorch to 1e-6 tolerance. If you see a `FAIL`, the implementation is wrong.

---

## Recap

- A `Value` node stores `data`, `grad`, and the operation that created it.
- Every operation defines a `_backward` closure that pushes gradients to its inputs via the chain rule.
- `backward()` topologically sorts the graph and calls `_backward` in reverse order.
- `*` and `+` are the primitive building blocks; `relu` and `tanh` are implemented on top of them.
- The same two-pass strategy (forward record, backward replay) is what powers `torch.autograd`.

---

## Get the extended notebook with Jacobian derivation, PyTorch hook system internals, custom autograd Functions, and profiling across batch sizes:

**https://www.skool.com/opensuperintelligencelab**
