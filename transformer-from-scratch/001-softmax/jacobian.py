"""
Jacobian of softmax — exact formula and numerical verification.

The Jacobian J_ij = d(softmax_i) / d(x_j) is the matrix of partial derivatives.
For softmax s(x) it has a clean closed form:
    J_ij = s_i * (kronecker_delta_ij - s_j)

When i == j:  ds_i / dx_i = s_i * (1 - s_i)
When i != j:  ds_i / dx_j = -s_i * s_j
"""


def softmax_jacobian(s):
    """
    Build the Jacobian matrix J where J[i][j] = d(s_i)/d(x_j).
    s is the softmax output vector (already computed).
    """
    n = len(s)
    J = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i == j:
                J[i][j] = s[i] * (1.0 - s[i])
            else:
                J[i][j] = -s[i] * s[j]
    return J


def numerical_jacobian(x, softmax_fn, eps=1e-7):
    """
    Verify the analytical Jacobian by perturbing each input dimension
    one at a time and measuring the change in output.
    """
    s0 = softmax_fn(x)
    n = len(x)
    J_num = [[0.0] * n for _ in range(n)]
    for j in range(n):
        x_plus = list(x)
        x_minus = list(x)
        x_plus[j] += eps
        x_minus[j] -= eps
        s_plus = softmax_fn(x_plus)
        s_minus = softmax_fn(x_minus)
        for i in range(n):
            J_num[i][j] = (s_plus[i] - s_minus[i]) / (2 * eps)
    return J_num


def stable_softmax(x):
    max_x = max(x)
    exp_x = [__import__('math').exp(x_i - max_x) for x_i in x]
    total = sum(exp_x)
    return [e / total for e in exp_x]


if __name__ == "__main__":
    x = [2.0, 1.0, 0.5]
    s = stable_softmax(x)
    print("Input:", x)
    print("softmax output:", s)

    J_exact = softmax_jacobian(s)
    J_numerical = numerical_jacobian(x, stable_softmax)

    print("\nAnalytical Jacobian:")
    for row in J_exact:
        print(["{:.6f}".format(v) for v in row])

    print("\nNumerical Jacobian:")
    for row in J_numerical:
        print(["{:.6f}".format(v) for v in row])

    # Check max absolute error
    max_err = max(
        abs(J_exact[i][j] - J_numerical[i][j])
        for i in range(len(x)) for j in range(len(x))
    )
    print(f"\nMax |error|: {max_err:.2e}  (< 1e-5 means our formula is correct)")
