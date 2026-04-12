"""Tiny autograd engine — forward records a tape, backward replays it.

Value class wraps a scalar and tracks every operation in a computation graph.
Each node stores:
  - data   : the actual float value
  - grad   : the gradient d(loss)/d(this node)
  - _op    : the operation that produced this node (for visualization)
  - _prev  : tuple of input Value nodes
  - _backward : function to call to propagate gradients
"""

from __future__ import annotations
import math


class Value:
    """A scalar that knows how to compute its own gradient."""

    _id = 0

    def __init__(self, data, _op=None, _prev=(), label=None):
        self.data = float(data)
        self.grad = 0.0
        self._op = _op          # operation that created this node
        self._prev = _prev      # tuple of input Value nodes
        self._backward = lambda: None  # accumulate gradient into inputs
        self._label = label or f"v{Value._id}"
        Value._id += 1

    # ------------------------------------------------------------------
    # Operations
    # ------------------------------------------------------------------

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, _op="+", _prev=(self, other))

        def _backward():
            self.grad += out.grad
            other.grad += out.grad

        out._backward = _backward
        return out

    def __radd__(self, other):
        return self.__add__(other)

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, _op="*", _prev=(self, other))

        def _backward():
            self.grad  += other.data * out.grad
            other.grad += self.data  * out.grad

        out._backward = _backward
        return out

    def __rmul__(self, other):
        return self.__mul__(other)

    def relu(self):
        out = Value(max(0.0, self.data), _op="relu", _prev=(self,))

        def _backward():
            self.grad += (self.data > 0) * out.grad

        out._backward = _backward
        return out

    def tanh(self):
        t = math.tanh(self.data)
        out = Value(t, _op="tanh", _prev=(self,))

        def _backward():
            self.grad += (1 - t * t) * out.grad

        out._backward = _backward
        return out

    def backward(self):
        """Topologically sort the graph and replay the tape."""
        # Reset gradients so we can call backward() multiple times cleanly.
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

        # out.grad is the seed gradient.  For a single scalar output
        # (e.g. a loss) this is 1.0.
        self.grad = 1.0
        for node in reversed(order):
            node._backward()

    def __repr__(self):
        if abs(self.grad) < 1e-6:
            grad_str = f"{self.grad:.2e}"
        else:
            grad_str = f"{self.grad:.6f}"
        return f"Value({self._label}, data={self.data:.4f}, grad={grad_str})"


# ------------------------------------------------------------------
# GradientTape — context manager that records every Value created
# ------------------------------------------------------------------

class GradientTape:
    """Records every Value created inside the block.

    Usage:
        with GradientTape() as tape:
            a = Value(2.0, label="a")
            b = a * a
        print(tape.history)   # list of (label, op, input_labels)
    """

    def __init__(self):
        self.history = []
        self._orig_id = Value._id

    def __enter__(self):
        self._orig_id = Value._id
        return self

    def __exit__(self, *args):
        # Nothing to clean up — Value objects are long-lived.
        pass


# ------------------------------------------------------------------
# Build a simple expression and take its gradients
# ------------------------------------------------------------------

if __name__ == "__main__":
    #        a
    #       / \
    #      b   c
    #      |   |
    #      d   e
    #       \ /
    #        f
    #        |
    #        g

    a = Value(2.0, label="a")
    b = a * a        # b = a^2
    c = Value(3.0, label="c")
    d = b.relu()     # d = relu(a^2)
    e = c * a        # e = 3a
    f = d + e        # f = relu(a^2) + 3a
    g = f.tanh()     # g = tanh(f)

    g.backward()

    def fmt(x):
        return f"{x:.6e}" if abs(x) < 1e-4 else f"{x:.6f}"

    print("Individual gradients:")
    print(f"  dg/da = {fmt(a.grad)}")
    print(f"  dg/db = {fmt(b.grad)}")
    print(f"  dg/dc = {fmt(c.grad)}")
    print(f"  dg/dd = {fmt(d.grad)}")
    print(f"  dg/de = {fmt(e.grad)}")
    print(f"  dg/df = {fmt(f.grad)}")
    print(f"  g     = {g.data:.6f}")
