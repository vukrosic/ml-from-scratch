"""Toy tracer — records operations instead of running them."""


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


def trace(fn, input_names):
    TracedTensor._counter = 0
    inputs = [TracedTensor(name=n) for n in input_names]
    output = fn(*inputs)
    return output.graph


if __name__ == "__main__":
    def mlp(x, w1, w2):
        h = x @ w1
        h = h.relu()
        h = h @ w2
        return h

    graph = trace(mlp, ["x", "w1", "w2"])
    for node in graph:
        print(node)
