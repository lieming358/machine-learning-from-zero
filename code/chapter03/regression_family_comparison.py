"""Compare OLS, polynomial, Ridge, and Lasso regression on the same split."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import root_mean_squared_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler


def build_data(seed: int = 21) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    x = rng.uniform(-3.0, 3.0, 100)
    y = 3.0 + 1.6 * x - 0.75 * x**2 + 0.22 * x**3 + rng.normal(0, 1.15, len(x))
    order = rng.permutation(len(x))
    train, test = order[:65], order[65:]
    return x[train, None], x[test, None], y[train], y[test]


def models() -> dict[str, object]:
    return {
        "linear OLS": make_pipeline(StandardScaler(), LinearRegression()),
        "degree-12 polynomial": make_pipeline(
            PolynomialFeatures(12, include_bias=False), StandardScaler(), LinearRegression()
        ),
        "degree-12 Ridge": make_pipeline(
            PolynomialFeatures(12, include_bias=False), StandardScaler(), Ridge(alpha=3.0)
        ),
        "degree-12 Lasso": make_pipeline(
            PolynomialFeatures(12, include_bias=False), StandardScaler(), Lasso(alpha=0.08, max_iter=100_000)
        ),
    }


def save_feature_map_figure(output: Path) -> None:
    """Show how a linear model in transformed features can draw a curve."""
    x = np.linspace(-2.0, 2.0, 300)
    terms = [x, x**2, x**3]
    colors = ["#3478c0", "#f28e2b", "#3a9d5d"]
    labels = ["feature x", "feature x²", "feature x³"]
    fig, axes = plt.subplots(1, 4, figsize=(12.5, 3.2), sharex=True)
    for axis, term, color, label in zip(axes[:3], terms, colors, labels):
        axis.plot(x, term, color=color, lw=2.5)
        axis.set_title(label)
    prediction = 2.0 + 1.2 * x - 0.8 * x**2 + 0.25 * x**3
    axes[3].plot(x, prediction, color="#6f4dbf", lw=2.8)
    axes[3].set_title("weighted sum = curve")
    for axis in axes:
        axis.axhline(0, color="#777777", lw=0.8)
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(alpha=0.12)
        axis.set_xlabel("original x")
    axes[0].set_ylabel("transformed value")
    fig.suptitle("Polynomial regression first creates features, then uses a linear model", y=1.03)
    fig.tight_layout()
    fig.savefig(output / "polynomial-feature-map.png", dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def save_regularization_paths(output: Path, seed: int = 9) -> None:
    """Contrast smooth Ridge shrinkage with sparse Lasso paths."""
    rng = np.random.default_rng(seed)
    base = rng.normal(size=(180, 1))
    x = np.hstack(
        [base + rng.normal(scale=0.15, size=(180, 1)) for _ in range(3)]
        + [rng.normal(size=(180, 1)) for _ in range(3)]
    )
    y = 3.0 * base[:, 0] + rng.normal(scale=0.8, size=len(base))
    x = StandardScaler().fit_transform(x)
    strengths = np.logspace(-3, 1.2, 80)
    ridge_paths, lasso_paths = [], []
    for strength in strengths:
        ridge_paths.append(Ridge(alpha=strength).fit(x, y).coef_)
        lasso_paths.append(Lasso(alpha=strength, max_iter=100_000).fit(x, y).coef_)

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0), sharey=True)
    for j in range(x.shape[1]):
        name = f"feature {j + 1}"
        axes[0].semilogx(strengths, np.asarray(ridge_paths)[:, j], lw=2, label=name)
        axes[1].semilogx(strengths, np.asarray(lasso_paths)[:, j], lw=2, label=name)
    axes[0].set_title("Ridge: coefficients shrink smoothly")
    axes[1].set_title("Lasso: coefficients can become exactly zero")
    for axis in axes:
        axis.axhline(0, color="#555555", lw=0.8)
        axis.set_xlabel("regularization strength λ")
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(alpha=0.15)
    axes[0].set_ylabel("fitted coefficient")
    axes[1].legend(frameon=False, fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(output / "regularization-coefficient-paths.png", dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    x_train, x_test, y_train, y_test = build_data()
    fitted = models()
    for model in fitted.values():
        model.fit(x_train, y_train)

    print("Same train/test split; lower test RMSE is better")
    for name, model in fitted.items():
        train_rmse = root_mean_squared_error(y_train, model.predict(x_train))
        test_rmse = root_mean_squared_error(y_test, model.predict(x_test))
        print(f"{name:22s} train={train_rmse:.3f} test={test_rmse:.3f}")

    grid = np.linspace(-3.1, 3.1, 500)[:, None]
    colors = ["#19324d", "#d94b4b", "#3a9d5d", "#f28e2b"]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4))
    axes[0].scatter(x_train[:, 0], y_train, color="#3478c0", alpha=0.65, label="training data")
    for (name, model), color in zip(fitted.items(), colors):
        axes[0].plot(grid[:, 0], model.predict(grid), color=color, lw=2.1, label=name)
    axes[0].set(xlabel="x", ylabel="y", title="Capacity and regularization change the fitted curve")
    axes[0].legend(frameon=False, fontsize=8)
    axes[0].set_ylim(-14, 9)

    labels, train_scores, test_scores = [], [], []
    for name, model in fitted.items():
        labels.append(name.replace("degree-12 ", "poly "))
        train_scores.append(root_mean_squared_error(y_train, model.predict(x_train)))
        test_scores.append(root_mean_squared_error(y_test, model.predict(x_test)))
    positions = np.arange(len(labels))
    width = 0.36
    axes[1].bar(positions - width / 2, train_scores, width, label="train RMSE", color="#3478c0")
    axes[1].bar(positions + width / 2, test_scores, width, label="test RMSE", color="#f28e2b")
    axes[1].set_xticks(positions, labels, rotation=18, ha="right")
    axes[1].set(ylabel="RMSE", title="Training fit is not generalization")
    axes[1].legend(frameon=False)
    for axis in axes:
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(alpha=0.15)
    fig.tight_layout()
    output = Path(__file__).resolve().parents[2] / "docs" / "assets"
    target = output / "regression-family-comparison.png"
    fig.savefig(target, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    save_feature_map_figure(output)
    save_regularization_paths(output)
    print(f"figure={target}")


if __name__ == "__main__":
    main()
