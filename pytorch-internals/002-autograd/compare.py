"""Compare our tiny autograd engine against torch.autograd on the same computation.

We run the exact same arithmetic graph (same values, same operations) with both
engines and print a side-by-side comparison of the gradients.
"""

from __future__ import annotations
import torch

from autograd import Value


def close(a, b, tol=1e-6):
    return abs(a - b) < tol


# ------------------------------------------------------------------
# Our autograd
# ------------------------------------------------------------------

def our_autograd(a_val, b_val, c_val):
    """mirrors torch_autograd below"""
    a = Value(a_val, label="a")
    b = Value(b_val, label="b")
    c = Value(c_val, label="c")

    d = a * b          # d = a * b
    e = b * c          # e = b * c
    f = d + e          # f = d + e
    g = f.relu()       # g = relu(f)
    h = g.tanh()       # h = tanh(g)

    h.backward()

    return {"a": a.grad, "b": b.grad, "c": c.grad,
            "d": d.grad, "e": e.grad, "f": f.grad,
            "g": g.grad, "h": h.grad}


# ------------------------------------------------------------------
# torch.autograd
# ------------------------------------------------------------------

def torch_autograd(a_val, b_val, c_val):
    a = torch.tensor(a_val, requires_grad=True)
    b = torch.tensor(b_val, requires_grad=True)
    c = torch.tensor(c_val, requires_grad=True)

    d = a * b
    e = b * c
    f = d + e
    g = torch.nn.functional.relu(f)
    h = torch.tanh(g)

    # Retain gradients for non-leaf tensors so we can access .grad
    d.retain_grad()
    e.retain_grad()
    f.retain_grad()
    g.retain_grad()
    h.retain_grad()

    h.backward()

    return {
        "a": a.grad.item(),
        "b": b.grad.item(),
        "c": c.grad.item(),
        "d": d.grad.item(),
        "e": e.grad.item(),
        "f": f.grad.item(),
        "g": g.grad.item(),
        "h": h.grad.item(),
    }


if __name__ == "__main__":
    test_cases = [
        (2.0, 3.0, 4.0),
        (1.0, 0.5, 2.0),
        (-1.5, 2.0, 0.0),
    ]

    all_pass = True

    for a_val, b_val, c_val in test_cases:
        ours  = our_autograd(a_val, b_val, c_val)
        theirs = torch_autograd(a_val, b_val, c_val)

        print(f"\n{'='*55}")
        print(f"  a={a_val}, b={b_val}, c={c_val}")
        print(f"{'='*55}")
        print(f"  {'Node':<6} {'Our grad':>12} {'PyTorch grad':>14} {'Match':>8}")
        print(f"  {'-'*6} {'-'*12} {'-'*14} {'-'*8}")

        for key in ["a", "b", "c", "d", "e", "f", "g", "h"]:
            o = ours[key]
            t = theirs[key]
            ok = "OK" if close(o, t) else "FAIL"
            if ok == "FAIL":
                all_pass = False
            print(f"  {key:<6} {o:>12.6f} {t:>14.6f} {ok:>8}")

    print(f"\n{'='*55}")
    if all_pass:
        print("All gradients match PyTorch to 1e-6 tolerance.")
    else:
        print("SOME GRADIENTS DO NOT MATCH — check the FAIL rows above.")
        exit(1)
