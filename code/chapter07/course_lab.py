"""Chapter 7 lab: hinge loss, a primal update, and a solvable dual example."""

from __future__ import annotations

import numpy as np


def hinge_loss(signed_margin: np.ndarray) -> np.ndarray:
    return np.maximum(0.0, 1.0 - signed_margin)


def one_primal_step(
    x: np.ndarray,
    y: np.ndarray,
    weight: np.ndarray,
    bias: float,
    learning_rate: float,
    c: float,
) -> tuple[np.ndarray, float]:
    margin = y * (x @ weight + bias)
    active = margin < 1.0
    gradient_weight = weight - c * (x[active].T @ y[active]) / len(x)
    gradient_bias = -c * y[active].sum() / len(x)
    return weight - learning_rate * gradient_weight, bias - learning_rate * gradient_bias


def two_point_dual_example() -> tuple[np.ndarray, float, float]:
    # x=-1 has label -1; x=+1 has label +1. The constraint forces alpha_1=alpha_2=a.
    x = np.array([[-1.0], [1.0]])
    y = np.array([-1.0, 1.0])
    gram = x @ x.T

    # W(a)=2a-2a^2, so dW/da=2-4a=0 gives a=0.5.
    alpha = np.array([0.5, 0.5])
    weight = float(np.sum(alpha * y * x[:, 0]))
    bias = 0.0
    margin = y * (x[:, 0] * weight + bias)
    print(f"gram_matrix={gram.tolist()}")
    print(f"dual_alpha={alpha.tolist()}, weight={weight:.1f}, margin={margin.tolist()}")

    assert np.isclose(alpha @ y, 0.0)
    assert np.allclose(margin, 1.0)
    return alpha, weight, bias


def main() -> None:
    margins = np.array([-0.5, 0.4, 1.3])
    print(f"hinge_loss={hinge_loss(margins).tolist()}")

    x = np.array([[-2.0, -1.0], [-1.0, -1.0], [1.0, 1.0], [2.0, 1.0]])
    y = np.array([-1.0, -1.0, 1.0, 1.0])
    initial_weight = np.zeros(2)
    updated_weight, updated_bias = one_primal_step(
        x,
        y,
        initial_weight,
        bias=0.0,
        learning_rate=0.1,
        c=1.0,
    )
    print(f"one_step_weight={updated_weight}, bias={updated_bias:.3f}")
    assert np.all(y * (x @ updated_weight + updated_bias) > 0)

    two_point_dual_example()


if __name__ == "__main__":
    main()
