"""Code generation from an optimized graph."""

from tracer import trace
from fusion import fuse_graph


def codegen(graph, fn_name="compiled_fn"):
    lines = [f"def {fn_name}(x, w1, w2):"]

    for out, op, args in graph:
        if op == "fused_matmul_relu":
            lines.append(f"    {out} = torch.relu({args[0]} @ {args[1]})")
        elif op == "matmul":
            lines.append(f"    {out} = {args[0]} @ {args[1]}")
        elif op == "relu":
            lines.append(f"    {out} = torch.relu({args[0]})")
        elif op == "add":
            lines.append(f"    {out} = {args[0]} + {args[1]}")

    lines.append(f"    return {graph[-1][0]}")
    return "\n".join(lines)


if __name__ == "__main__":
    def mlp(x, w1, w2):
        h = x @ w1
        h = h.relu()
        h = h @ w2
        return h

    graph = trace(mlp, ["x", "w1", "w2"])
    optimized = fuse_graph(graph)
    print(codegen(optimized))
