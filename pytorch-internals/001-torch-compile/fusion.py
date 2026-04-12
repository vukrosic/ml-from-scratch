"""Graph fusion pass — merges adjacent operations into one.

What it does:
  - Scans a graph (list of op tuples) for matmul -> relu patterns.
  - Merges each match into a single fused_matmul_relu node.
  - Demonstrates what TorchInductor does: fuse adjacent ops so one GPU kernel
    runs instead of two.

Run:
  python fusion.py

Expected output:
  Before fusion:
    ('t3', 'matmul', ['x', 'w1'])
    ('t4', 'relu', ['t3'])
    ('t5', 'matmul', ['t4', 'w2'])

  After fusion:
    ('t4', 'fused_matmul_relu', ['x', 'w1'])
    ('t5', 'matmul', ['t4', 'w2'])
"""

from tracer import trace


def fuse_graph(graph):
    fused = []
    skip_next = set()           # indices to skip after a fusion

    for i, (out, op, args) in enumerate(graph):
        if i in skip_next:       # already fused into a previous op
            continue

        # Check if this matmul is followed by a relu on the same output
        if op == "matmul" and i + 1 < len(graph):
            next_out, next_op, next_args = graph[i + 1]
            if next_op == "relu" and next_args[0] == out:
                # Merge: matmul + relu becomes one fused operation
                fused.append((next_out, "fused_matmul_relu", args))
                skip_next.add(i + 1)    # skip the relu since it's merged
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
