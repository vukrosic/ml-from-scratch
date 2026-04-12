"""Code generation from an optimized graph back into runnable Python.

What it does:
  - Takes a fused graph (from fuse_graph) and emits a Python function string.
  - Each op in the graph becomes one line of Python: fused_matmul_relu -> torch.relu(x @ w).
  - TorchInductor does the same thing but generates Triton GPU code instead of Python.

Run:
  python codegen.py

Expected output:
  def compiled_fn(x, w1, w2):
      t4 = torch.relu(x @ w1)
      t5 = t4 @ w2
      return t5
"""

from tracer import trace
from fusion import fuse_graph


def codegen(graph, fn_name="compiled_fn"):
    lines = [f"def {fn_name}(x, w1, w2):"]

    for out, op, args in graph:
        if op == "fused_matmul_relu":
            # Fused: matmul and relu in one kernel
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
