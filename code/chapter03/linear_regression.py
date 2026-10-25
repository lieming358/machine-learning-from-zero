"""Chapter 3: compare least-squares and gradient-descent solutions."""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LinearRegression


def add_intercept(x: np.ndarray) -> np.ndarray:
    return np.column_stack([np.ones(len(x)), x])


def fit_gradient_descent(
    x: np.ndarray, y: np.ndarray, learning_rate: float = 0.05, steps: int = 2000
) -> np.ndarray:
    theta = np.zeros(x.shape[1])
    for _ in range(steps):
        error = x @ theta - y
        gradient = (2.0 / len(y)) * x.T @ error
        theta -= learning_rate * gradient
    return theta


def rmse(y: np.ndarray, prediction: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y - prediction) ** 2)))


def locally_weighted_predict(
    x_train: np.ndarray, y_train: np.ndarray, query: float, bandwidth: float
) -> float:
    """Predict near `query` by giving nearby one-dimensional examples more weight."""
    distance = x_train[:, 0] - query
    weights = np.exp(-(distance**2) / (2.0 * bandwidth**2))
    design = add_intercept(x_train)
    # Multiplying by sqrt(weight) converts weighted least squares to least squares.
    weighted_design = design * np.sqrt(weights)[:, None]
    weighted_target = y_train * np.sqrt(weights)
    theta = np.linalg.lstsq(weighted_design, weighted_target, rcond=None)[0]
    return float(np.array([1.0, query]) @ theta)


def main() -> None:
    rng = np.random.default_rng(7)
    area = rng.uniform(25.0, 120.0, size=(160, 1))
    rent = 38.0 * area[:, 0] + 800.0 + rng.normal(0.0, 180.0, len(area))

    ids = rng.permutation(len(area))
    train_ids, test_ids = ids[:120], ids[120:]
    x_train, x_test = area[train_ids], area[test_ids]
    y_train, y_test = rent[train_ids], rent[test_ids]

    mean, std = x_train.mean(axis=0), x_train.std(axis=0)
    train_design = add_intercept((x_train - mean) / std)
    test_design = add_intercept((x_test - mean) / std)

    theta_closed = np.linalg.lstsq(train_design, y_train, rcond=None)[0]
    theta_gd = fit_gradient_descent(train_design, y_train)

    framework_model = LinearRegression().fit(x_train, y_train)

    pred_closed = test_design @ theta_closed
    pred_gd = test_design @ theta_gd
    pred_framework = framework_model.predict(x_test)
    baseline = np.full_like(y_test, y_train.mean())

    print(f"closed_form_theta={theta_closed.round(3)}")
    print(f"gradient_descent_theta={theta_gd.round(3)}")
    print(f"max_prediction_gap={np.max(np.abs(pred_closed - pred_gd)):.6f}")
    print(
        "sklearn_max_prediction_gap="
        f"{np.max(np.abs(pred_closed - pred_framework)):.6f}"
    )
    print(f"baseline_test_rmse={rmse(y_test, baseline):.3f}")
    print(f"model_test_rmse={rmse(y_test, pred_closed):.3f}")
    local_prediction = locally_weighted_predict(x_train, y_train, query=80.0, bandwidth=15.0)
    print(f"locally_weighted_prediction_at_80m2={local_prediction:.3f}")

    assert np.allclose(pred_closed, pred_gd, atol=1e-6)
    assert np.allclose(pred_closed, pred_framework, atol=1e-8)


if __name__ == "__main__":
    main()
