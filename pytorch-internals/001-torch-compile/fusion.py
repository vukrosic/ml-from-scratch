"""Graph fusion pass — merge adjacent operations."""

from tracer import trace


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


if __name__ == "__main__":
    def mlp(x, w1, w2):
        h = x @ w1
        h = h.relu()
        h = h @ w2
        return h

    graph = trace(mlp, ["x", "w1", "w2"])
    print("Before fusion:")
    for node in graph:
        print(f"  {node}")

    optimized = fuse_graph(graph)
    print("\nAfter fusion:")
    for node in optimized:
        print(f"  {node}")
