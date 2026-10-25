"""Generate Chapter 4's conceptual sigmoid and regression-vs-classification figures."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression


OUTPUT = Path(__file__).resolve().parents[2] / "docs" / "assets"


def sigmoid_figure() -> None:
    z = np.linspace(-8, 8, 500)
    probability = 1 / (1 + np.exp(-z))
    fig, ax = plt.subplots(figsize=(7.4, 4.4))
    ax.plot(z, probability, color="#3478c0", lw=3)
    ax.axhline(0.5, color="#3a9d5d", linestyle="--")
    ax.axvline(0, color="#3a9d5d", linestyle="--")
    ax.fill_between(z[z < 0], probability[z < 0], 0.5, color="#3478c0", alpha=0.1)
    ax.fill_between(z[z >= 0], 0.5, probability[z >= 0], color="#f28e2b", alpha=0.1)
    ax.annotate("z = 0 → p = 0.5", (0, 0.5), xytext=(1.5, 0.3), arrowprops={"arrowstyle": "->"})
    ax.set(xlabel="linear score z = wᵀx + b", ylabel="probability p(y=1|x)", title="Sigmoid maps any score into (0, 1)")
    ax.set_ylim(-0.03, 1.03)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(alpha=0.15)
    fig.tight_layout()
    fig.savefig(OUTPUT / "sigmoid-function.png", dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def linear_vs_logistic() -> None:
    x = np.array([0.8, 1.3, 1.8, 2.2, 2.7, 3.1, 3.6, 4.0, 4.5, 5.0, 8.5])[:, None]
    y = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1])
    linear = LinearRegression().fit(x, y)
    logistic = LogisticRegression(C=1e6).fit(x, y)
    grid = np.linspace(0, 9, 500)[:, None]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.1), sharey=True)
    axes[0].scatter(x[:, 0], y, color="#19324d", zorder=3)
    axes[0].plot(grid[:, 0], linear.predict(grid), color="#d94b4b", lw=2.5)
    axes[0].axhline(0.5, color="#3a9d5d", linestyle="--")
    axes[0].set(title="Linear regression is unbounded", xlabel="study hours", ylabel="class / predicted value")
    axes[1].scatter(x[:, 0], y, color="#19324d", zorder=3)
    axes[1].plot(grid[:, 0], logistic.predict_proba(grid)[:, 1], color="#3478c0", lw=2.5)
    axes[1].axhline(0.5, color="#3a9d5d", linestyle="--")
    axes[1].set(title="Logistic regression predicts probability", xlabel="study hours")
    for ax in axes:
        ax.set_ylim(-0.35, 1.35)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(alpha=0.15)
    fig.tight_layout()
    fig.savefig(OUTPUT / "linear-vs-logistic-classification.png", dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    sigmoid_figure()
    linear_vs_logistic()
    print(f"Generated Chapter 4 figures in {OUTPUT}")


if __name__ == "__main__":
    main()
