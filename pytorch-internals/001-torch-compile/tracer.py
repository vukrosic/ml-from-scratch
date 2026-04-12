"""Toy tracer — records operations instead of running them.

What it does:
  - Intercepts __matmul__, relu, and add on TracedTensor objects.
  - Instead of computing math, each op records (output_name, operation, input_names)
    into a shared graph list.
  - Running an MLP through trace() produces the same structure as TorchDynamo's FX graph.

Run:
  python tracer.py

Expected output:
  ('t3', 'matmul', ['x', 'w1'])
  ('t4', 'relu', ['t3'])
  ('t5', 'matmul', ['t4', 'w2'])
"""

import torch
import torch.nn as nn

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


def trace(fn, input_names):
    TracedTensor._counter = 0                    # reset naming counter
    inputs = [TracedTensor(name=n) for n in input_names]  # create input tensors
    output = fn(*inputs)                         # run the function, recording ops
    return output.graph                          # return the full operation list


if __name__ == "__main__":
    def mlp(x, w1, w2):
        h = x @ w1
        h = h.relu()
        h = h @ w2
        return h

    graph = trace(mlp, ["x", "w1", "w2"])
    for node in graph:
        print(node)
