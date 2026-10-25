"""Chapter 2: linear regression, train-only scaling, and gradient checking."""

from __future__ import annotations

import numpy as np


def mse_and_gradients(
    x: np.ndarray, y: np.ndarray, w: np.ndarray, b: float
) -> tuple[float, np.ndarray, float]:
    """Return mean squared error and its analytic gradients."""
    error = x @ w + b - y
    loss = float(np.mean(error**2))
    dw = (2.0 / len(x)) * x.T @ error
    db = float(2.0 * np.mean(error))
    return loss, dw, db


def numerical_gradient(
    x: np.ndarray, y: np.ndarray, w: np.ndarray, b: float, eps: float = 1e-6
) -> tuple[np.ndarray, float]:
    """Check a derivation using slow central finite differences."""
    dw = np.zeros_like(w)
    for j in range(len(w)):
        w_plus, w_minus = w.copy(), w.copy()
        w_plus[j] += eps
        w_minus[j] -= eps
        loss_plus = mse_and_gradients(x, y, w_plus, b)[0]
        loss_minus = mse_and_gradients(x, y, w_minus, b)[0]
        dw[j] = (loss_plus - loss_minus) / (2.0 * eps)
    db = (
        mse_and_gradients(x, y, w, b + eps)[0]
        - mse_and_gradients(x, y, w, b - eps)[0]
    ) / (2.0 * eps)
    return dw, float(db)


def main() -> None:
    rng = np.random.default_rng(42)
    x = rng.uniform(0.0, 10.0, size=(120, 1))
    y = 2.5 * x[:, 0] + 1.2 + rng.normal(0.0, 1.0, size=len(x))

    order = rng.permutation(len(x))
    train_ids, test_ids = order[:90], order[90:]
    x_train, y_train = x[train_ids], y[train_ids]
    x_test, y_test = x[test_ids], y[test_ids]

    # Fit preprocessing on training data only: test statistics must remain unseen.
    mean = x_train.mean(axis=0)
    std = x_train.std(axis=0)
    x_train_scaled = (x_train - mean) / std
    x_test_scaled = (x_test - mean) / std

    w = np.zeros(x.shape[1])
    b = 0.0
    analytic = mse_and_gradients(x_train_scaled, y_train, w, b)[1:]
    numeric = numerical_gradient(x_train_scaled, y_train, w, b)
    grad_error = max(
        float(np.max(np.abs(analytic[0] - numeric[0]))),
        abs(analytic[1] - numeric[1]),
    )

    learning_rate = 0.05
    for step in range(1000):
        loss, dw, db = mse_and_gradients(x_train_scaled, y_train, w, b)
        w -= learning_rate * dw
        b -= learning_rate * db
        if step in {0, 9, 99, 999}:
            print(f"step={step + 1:4d} train_mse={loss:.4f}")

    prediction = x_test_scaled @ w + b
    test_rmse = float(np.sqrt(np.mean((prediction - y_test) ** 2)))

    # Convert scaled parameters back to the original x coordinate system.
    original_w = w / std
    original_b = b - float(mean @ original_w)
    print(f"gradient_check_max_error={grad_error:.3e}")
    print(f"learned: y = {original_w[0]:.3f}x + {original_b:.3f}")
    print(f"test_rmse={test_rmse:.3f}")


if __name__ == "__main__":
    main()
