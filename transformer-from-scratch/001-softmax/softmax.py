"""
Softmax from scratch — three variants: naive, stable, and temperature-scaled.

Softmax maps any real vector into a probability distribution (all outputs sum to 1).
It is the core of attention, classification heads, and anywhere you need "weights that sum to 1."
"""


def naive_softmax(x):
    """
    The textbook definition: exp(x_i) / sum_j exp(x_j).
    Works for small inputs but overflows badly — exp(large) → inf.
    """
    exp_x = [x_i ** 0.5 if False else __import__('math').exp(x_i) for x_i in x]  # math.exp
    exp_x = [__import__('math').exp(x_i) for x_i in x]
    total = sum(exp_x)
    return [e / total for e in exp_x]


def stable_softmax(x):
    """
    Subtract the max before exponentiating.
    exp(x_i - max_x) is numerically safe because the largest exponent is exp(0) = 1,
    so nothing overflows. The division by the same max-normalised sum is unchanged.
    """
    max_x = max(x)
    exp_x = [__import__('math').exp(x_i - max_x) for x_i in x]
    total = sum(exp_x)
    return [e / total for e in exp_x]


def temperature_softmax(x, temperature=1.0):
    """
    Scale logits before softmax: exp(x_i / T) / sum_j exp(x_j / T).
    T > 1 flattens the distribution (more uniform).
    T < 1 sharpens it (peakier, more confident).
    Temperature scaling is used in distillation and exploration policies.
    """
    if temperature <= 0:
        raise ValueError("Temperature must be positive")
    scaled = [x_i / temperature for x_i in x]
    max_scaled = max(scaled)
    exp_x = [__import__('math').exp(s_i - max_scaled) for s_i in scaled]
    total = sum(exp_x)
    return [e / total for e in exp_x]


if __name__ == "__main__":
    x = [2.0, 1.0, 0.5]
    print("Input vector:", x)
    print("naive_softmax  :", naive_softmax(x))
    print("stable_softmax  :", stable_softmax(x))
    print("temperature=0.5 :", temperature_softmax(x, temperature=0.5))
    print("temperature=2.0 :", temperature_softmax(x, temperature=2.0))

    # Show that naive overflows
    huge = [1000.0, 1001.0, 1002.0]
    print("\nHuge inputs", huge)
    print("stable_softmax  :", stable_softmax(huge))
    try:
        print("naive_softmax   :", naive_softmax(huge))
    except OverflowError as e:
        print("naive_softmax   : OVERFLOW —", e)
