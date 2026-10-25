"""Chapter 3: linear regression on scikit-learn's built-in diabetes dataset."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


def bootstrap_rmse_interval(
    y: np.ndarray,
    prediction: np.ndarray,
    *,
    samples: int = 2000,
    seed: int = 42,
) -> tuple[float, float]:
    """Return a paired bootstrap 95% interval for test RMSE."""
    rng = np.random.default_rng(seed)
    estimates = np.empty(samples)
    for index in range(samples):
        selected = rng.integers(0, len(y), size=len(y))
        estimates[index] = np.sqrt(
            mean_squared_error(y[selected], prediction[selected])
        )
    low, high = np.percentile(estimates, [2.5, 97.5])
    return float(low), float(high)


def save_diagnostics(y: np.ndarray, prediction: np.ndarray) -> Path:
    """Save an actual-vs-predicted and residual diagnostic figure."""
    output = Path(__file__).resolve().parents[2] / "docs" / "assets"
    output.mkdir(parents=True, exist_ok=True)
    target = output / "diabetes-real-data-diagnostics.png"
    residual = y - prediction

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].scatter(y, prediction, alpha=0.72, edgecolor="white")
    limits = [float(min(y.min(), prediction.min())), float(max(y.max(), prediction.max()))]
    axes[0].plot(limits, limits, "--", color="#d94b4b", label="perfect prediction")
    axes[0].set(xlabel="True target", ylabel="Predicted target", title="Actual vs. predicted")
    axes[0].legend(frameon=False)

    axes[1].scatter(prediction, residual, alpha=0.72, color="#3a9d5d", edgecolor="white")
    axes[1].axhline(0, color="#d94b4b", linestyle="--")
    axes[1].set(xlabel="Predicted target", ylabel="Residual", title="Residual diagnosis")
    for axis in axes:
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(alpha=0.16)
    fig.tight_layout()
    fig.savefig(target, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return target


def main() -> None:
    dataset = load_diabetes()
    x_train, x_test, y_train, y_test = train_test_split(
        dataset.data, dataset.target, test_size=0.25, random_state=42
    )
    model = LinearRegression().fit(x_train, y_train)
    prediction = model.predict(x_test)
    baseline = np.full_like(y_test, y_train.mean(), dtype=float)

    rmse = float(np.sqrt(mean_squared_error(y_test, prediction)))
    baseline_rmse = float(np.sqrt(mean_squared_error(y_test, baseline)))
    low, high = bootstrap_rmse_interval(y_test, prediction)

    print(f"samples={dataset.data.shape[0]} features={dataset.data.shape[1]}")
    print(f"baseline_rmse={baseline_rmse:.3f}")
    print(f"linear_rmse={rmse:.3f} bootstrap_95pct_ci=[{low:.3f}, {high:.3f}]")
    print(f"linear_mae={mean_absolute_error(y_test, prediction):.3f}")
    print(f"linear_r2={r2_score(y_test, prediction):.3f}")
    print(f"figure={save_diagnostics(y_test, prediction)}")

    assert rmse < baseline_rmse
    assert np.isfinite(prediction).all()


if __name__ == "__main__":
    main()
