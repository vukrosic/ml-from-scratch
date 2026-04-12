"""Text-based computation graph visualizer.

Draws the graph built by autograd.py as ASCII art so you can see
exactly which operations connect which Value nodes.
"""

from __future__ import annotations


def build_label(node, show_data=True, show_grad=True):
    """Human-readable label for a single Value node."""
    parts = [node._label]
    if show_data:
        parts.append(f"d={node.data:.2f}")
    if show_grad:
        parts.append(f"g={node.grad:.2f}")
    return "\n".join(parts)


def label_width(node):
    lines = build_label(node).split("\n")
    return max(len(l) for l in lines)


def visualize(*roots, show_data=True, show_grad=True):
    """Print an ASCII representation of the computation graph.

    Args:
        *roots: one or more Value nodes to visualize (typically the output(s)).
        show_data: display the data value inside each node.
        show_grad: display the gradient inside each node.
    """
    # Collect all nodes via BFS from roots.
    visited = set()
    queue = list(roots)
    nodes = []

    while queue:
        v = queue.pop(0)
        if id(v) in visited:
            continue
        visited.add(id(v))
        nodes.append(v)
        queue.extend(v._prev)

    # Build node-to-index map.
    v2i = {id(v): i for i, v in enumerate(nodes)}

    # Compute column width for each node.
    widths = [label_width(n) for n in nodes]

    # Pretty-print each node and its children.
    for i, v in enumerate(nodes):
        lw = lw_i = widths[i]

        # Header line: the node label + op.
        op_str = f"[{v._op}]" if v._op else ""
        data_line = build_label(v, show_data, show_grad)

        print(f"  {data_line}  {op_str}")

        # Draw edges to children.
        if v._prev:
            child_indices = [v2i[id(c)] for c in v._prev]
            print("  |" + " " * (lw_i - 1))
            print("  " + "+" + "-" * (lw_i - 1) + "* ", end="")
            # Print child labels on the same line
            for j, ci in enumerate(child_indices):
                cw = widths[ci]
                child_label = build_label(nodes[ci], show_data, show_grad).split("\n")[0]
                sep = "  " if j < len(child_indices) - 1 else ""
                print(f"{child_label}"[:cw].ljust(cw) + sep, end="")
            print()


def node_tree(root, indent=0, visited=None):
    """Print a top-down tree view of the computation graph.

    Example output:
        g [tanh]
        └── f [+]
            ├── d [relu]
            │   └── b [*]
            │       ├── a
            │       └── a
            └── e [*]
                ├── c
                └── a
    """
    if visited is None:
        visited = set()

    prefix = "│   " * indent
    node_id = id(root)
    connector = "└── " if node_id in visited else "├── "
    loop_marker = " (loop)" if node_id in visited else ""

    op = f" [{root._op}]" if root._op else ""
    print(f"{prefix}{connector}{root._label}{op}{loop_marker}")

    if node_id not in visited:
        visited.add(node_id)
        for i, child in enumerate(root._prev):
            child_prefix = "│   " * (indent + 1)
            child_connector = "└── " if i == len(root._prev) - 1 else "├── "
            child_op = f" [{child._op}]" if child._op else ""
            print(f"{child_prefix}{child_connector}{child._label}{child_op}")
            node_tree(child, indent + 2, visited)


if __name__ == "__main__":
    from autograd import Value

    # Rebuild the graph from autograd.py
    a = Value(2.0, label="a")
    b = a * a
    c = Value(3.0, label="c")
    d = b.relu()
    e = c * a
    f = d + e
    g = f.tanh()

    g.backward()

    print("=== Tree view ===")
    node_tree(g)

    print("\n=== Flat node view ===")
    visualize(g)
