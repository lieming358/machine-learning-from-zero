"""Generate the reproducible figures embedded in Chapters 1–3."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "docs" / "assets"
NAVY = "#19324d"
BLUE = "#3478c0"
ORANGE = "#f28e2b"
RED = "#d94b4b"
GREEN = "#3a9d5d"
LIGHT = "#eef4fa"


def finish(fig: plt.Figure, name: str) -> None:
    fig.tight_layout()
    fig.savefig(OUTPUT / name, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def set_style() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titleweight": "bold",
            "axes.titlecolor": NAVY,
            "axes.labelcolor": NAVY,
            "xtick.color": NAVY,
            "ytick.color": NAVY,
            "figure.facecolor": "white",
        }
    )


def rent_data(seed: int = 7) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    area = rng.uniform(25, 120, 90)
    rent = 38 * area + 800 + rng.normal(0, 210, len(area))
    return area, rent


def plot_rent_distribution() -> None:
    area, rent = rent_data()
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    axes[0].scatter(area, rent, s=30, alpha=0.78, color=BLUE, edgecolor="white")
    axes[0].set(title="Observed rental data", xlabel="Area (m²)", ylabel="Monthly rent")
    axes[0].grid(alpha=0.18)
    axes[1].hist(rent, bins=12, color=ORANGE, alpha=0.88, edgecolor="white")
    axes[1].axvline(rent.mean(), color=RED, ls="--", lw=2, label=f"mean = {rent.mean():.0f}")
    axes[1].set(title="Target distribution", xlabel="Monthly rent", ylabel="Number of homes")
    axes[1].legend(frameon=False)
    finish(fig, "rent-data-distribution.png")


def plot_fit_and_residuals() -> None:
    area, rent = rent_data()
    slope, intercept = np.polyfit(area, rent, 1)
    prediction = slope * area + intercept
    order = np.argsort(area)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    axes[0].scatter(area, rent, s=28, alpha=0.7, color=BLUE, edgecolor="white", label="observed")
    axes[0].plot(area[order], prediction[order], lw=2.5, color=RED, label="fitted line")
    axes[0].set(title="A line summarizes the trend", xlabel="Area (m²)", ylabel="Monthly rent")
    axes[0].legend(frameon=False)
    residual = rent - prediction
    axes[1].scatter(prediction, residual, s=28, alpha=0.72, color=GREEN, edgecolor="white")
    axes[1].axhline(0, color=RED, lw=2, ls="--")
    axes[1].set(title="Residuals should look patternless", xlabel="Predicted rent", ylabel="Residual")
    finish(fig, "rent-fit-residuals.png")


def plot_model_complexity() -> None:
    rng = np.random.default_rng(12)
    x = np.linspace(-3, 3, 18)
    true_x = np.linspace(-3, 3, 400)
    true_fn = lambda z: 0.45 * z**2 + np.sin(1.7 * z)
    y = true_fn(x) + rng.normal(0, 0.45, len(x))
    degrees = [1, 3, 15]
    titles = ["Underfit: too rigid", "Good fit: captures signal", "Overfit: learns noise"]
    colors = [ORANGE, GREEN, RED]
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.7), sharey=True)
    for ax, degree, title, color in zip(axes, degrees, titles, colors):
        coef = np.polyfit(x, y, degree)
        ax.scatter(x, y, s=28, color=BLUE, edgecolor="white", zorder=3)
        ax.plot(true_x, true_fn(true_x), color=NAVY, ls="--", lw=1.6, label="true pattern")
        ax.plot(true_x, np.polyval(coef, true_x), color=color, lw=2.4, label=f"degree {degree}")
        ax.set(title=title, xlabel="x", ylim=(-2, 7))
        ax.grid(alpha=0.15)
        ax.legend(frameon=False, fontsize=8)
    axes[0].set_ylabel("y")
    finish(fig, "underfit-goodfit-overfit.png")


def plot_learning_curves() -> None:
    epochs = np.arange(1, 81)
    train = 0.22 + 1.25 * np.exp(-epochs / 17)
    validation = 0.31 + 1.2 * np.exp(-epochs / 13) + 0.00024 * np.maximum(epochs - 34, 0) ** 2
    best = int(np.argmin(validation))
    fig, ax = plt.subplots(figsize=(8.2, 4.2))
    ax.plot(epochs, train, lw=2.5, color=BLUE, label="training loss")
    ax.plot(epochs, validation, lw=2.5, color=ORANGE, label="validation loss")
    ax.axvline(epochs[best], color=GREEN, ls="--", lw=2, label="early stopping point")
    ax.axvspan(epochs[best], epochs[-1], color=RED, alpha=0.08, label="overfitting region")
    ax.annotate(
        "validation worsens\nwhile training improves",
        (62, validation[61]),
        xytext=(45, 1.15),
        arrowprops={"arrowstyle": "->", "color": RED},
        color=RED,
    )
    ax.set(title="Learning curves reveal overfitting", xlabel="Training epoch", ylabel="Loss")
    ax.legend(frameon=False, ncol=2)
    ax.grid(alpha=0.16)
    finish(fig, "learning-curves-overfitting.png")


def plot_confusion_matrix() -> None:
    """Visualize the four binary-classification outcomes with a fraud example."""
    matrix = np.array([[978, 12], [2, 8]])
    labels = np.array([["TN = 978\nlegitimate passed", "FP = 12\nfalse alarms"],
                       ["FN = 2\nfraud missed", "TP = 8\nfraud caught"]])
    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    ax.imshow(matrix, cmap="Blues", vmin=0, vmax=1000)
    for row in range(2):
        for column in range(2):
            color = "white" if matrix[row, column] > 500 else NAVY
            ax.text(column, row, labels[row, column], ha="center", va="center",
                    fontsize=13, fontweight="bold", color=color)
    ax.set_xticks([0, 1], ["predicted negative", "predicted positive"])
    ax.set_yticks([0, 1], ["actual negative", "actual positive"])
    ax.set_xlabel("Model decision")
    ax.set_ylabel("Ground truth")
    ax.set_title("Confusion matrix: 1,000 transactions, 10 are fraud")
    ax.tick_params(length=0)
    finish(fig, "confusion-matrix-fraud-example.png")


def binary_curve_points(y_true: np.ndarray, score: np.ndarray) -> tuple[np.ndarray, ...]:
    thresholds = np.r_[np.inf, np.sort(np.unique(score))[::-1], -np.inf]
    tpr, fpr, precision, recall = [], [], [], []
    positives = np.sum(y_true == 1)
    negatives = np.sum(y_true == 0)
    for threshold in thresholds:
        predicted = score >= threshold
        tp = np.sum(predicted & (y_true == 1))
        fp = np.sum(predicted & (y_true == 0))
        fn = positives - tp
        tpr.append(tp / positives)
        fpr.append(fp / negatives)
        precision.append(tp / (tp + fp) if tp + fp else 1.0)
        recall.append(tp / (tp + fn))
    return thresholds, np.array(tpr), np.array(fpr), np.array(precision), np.array(recall)


def plot_roc_and_pr_curves() -> None:
    """Show how one threshold sweep creates ROC and precision-recall curves."""
    rng = np.random.default_rng(31)
    negative_scores = rng.beta(1.4, 5.2, 990)
    positive_scores = rng.beta(4.4, 1.9, 10)
    y_true = np.r_[np.zeros(990, dtype=int), np.ones(10, dtype=int)]
    scores = np.r_[negative_scores, positive_scores]
    thresholds, tpr, fpr, precision, recall = binary_curve_points(y_true, scores)
    roc_auc = np.trapezoid(tpr, fpr)
    pr_auc = np.trapezoid(precision, recall)

    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.5))
    axes[0].plot(fpr, tpr, color=BLUE, lw=2.6, label=f"model (AUC={roc_auc:.3f})")
    axes[0].plot([0, 1], [0, 1], color="#777777", ls="--", label="random ranking")
    axes[0].set(title="ROC: recall versus false-positive rate", xlabel="FPR", ylabel="TPR / recall",
                xlim=(-.02, 1.02), ylim=(-.02, 1.02))
    axes[0].legend(frameon=False)

    prevalence = y_true.mean()
    axes[1].plot(recall, precision, color=ORANGE, lw=2.6, label=f"model (area={pr_auc:.3f})")
    axes[1].axhline(prevalence, color="#777777", ls="--",
                    label=f"random baseline = prevalence ({prevalence:.1%})")
    axes[1].set(title="PR: precision versus recall", xlabel="Recall", ylabel="Precision",
                xlim=(-.02, 1.02), ylim=(-.02, 1.02))
    axes[1].legend(frameon=False)
    for threshold, marker in [(0.7, "o"), (0.5, "s"), (0.3, "^")]:
        index = int(np.argmin(np.abs(thresholds - threshold)))
        axes[0].scatter(fpr[index], tpr[index], marker=marker, s=55, color=RED, zorder=4)
        axes[0].annotate(f"τ={threshold}", (fpr[index], tpr[index]), xytext=(5, -16),
                         textcoords="offset points", fontsize=8, color=RED)
        axes[1].scatter(recall[index], precision[index], marker=marker, s=55, color=RED, zorder=4)
        pr_offset = (-35, 8) if threshold == 0.3 else (5, -16)
        axes[1].annotate(f"τ={threshold}", (recall[index], precision[index]), xytext=pr_offset,
                         textcoords="offset points", fontsize=8, color=RED)
    for axis in axes:
        axis.grid(alpha=.16)
    fig.suptitle("Every point is a different decision threshold", color=NAVY, fontweight="bold")
    finish(fig, "roc-pr-threshold-curves.png")


def plot_dataset_anatomy() -> None:
    data = np.array([[28, 5, 1], [45, 10, 0], [72, 18, 1], [96, 7, 0]], dtype=float)
    shown = np.array([[28, 5, 1], [45, 10, 0], [72, 18, 1], [96, 7, 0]])
    fig, ax = plt.subplots(figsize=(8.6, 4.8))
    normalized = (data - data.min(axis=0)) / np.maximum(np.ptp(data, axis=0), 1)
    ax.imshow(normalized, cmap="Blues", vmin=-0.25, vmax=1.25, aspect="auto")
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            ax.text(j, i, str(shown[i, j]), ha="center", va="center", fontsize=15, color=NAVY)
    ax.set_xticks(range(3), ["x₁: area (m²)", "x₂: age (years)", "x₃: elevator"])
    ax.set_yticks(range(4), ["x⁽¹⁾", "x⁽²⁾", "x⁽³⁾", "x⁽⁴⁾"])
    ax.tick_params(length=0)
    ax.set_title("Dataset matrix X: rows are examples, columns are features", pad=18)
    ax.annotate(
        "x₂⁽³⁾ = X₃₂ = 18",
        xy=(1, 2),
        xytext=(2.25, 2.75),
        arrowprops={"arrowstyle": "->", "color": RED, "lw": 2},
        color=RED,
        fontsize=12,
        fontweight="bold",
    )
    ax.text(-0.95, 1.5, "m = 4\nexamples", ha="center", va="center", color=GREEN, fontsize=12)
    ax.text(1, 3.75, "d = 3 features", ha="center", va="center", color=GREEN, fontsize=12)
    finish(fig, "dataset-matrix-anatomy.png")


def plot_vector_geometry() -> None:
    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    origin = np.array([0.0, 0.0])
    x = np.array([3.0, 1.4])
    w = np.array([2.1, 2.8])
    for vector, color, label in [(x, BLUE, "feature vector x"), (w, ORANGE, "weight vector w")]:
        ax.annotate("", xy=vector, xytext=origin, arrowprops={"arrowstyle": "->", "lw": 3, "color": color})
        ax.text(*(vector * 1.05), label, color=color, fontsize=11, fontweight="bold")
    angle_x, angle_w = np.arctan2(x[1], x[0]), np.arctan2(w[1], w[0])
    arc = np.linspace(angle_x, angle_w, 80)
    ax.plot(0.8 * np.cos(arc), 0.8 * np.sin(arc), color=GREEN, lw=2)
    ax.text(0.78, 0.62, "θ", color=GREEN, fontsize=15)
    projection = (x @ w) / (w @ w) * w
    ax.plot([x[0], projection[0]], [x[1], projection[1]], ls="--", color=RED, lw=1.7)
    ax.scatter(*projection, color=RED, zorder=4)
    ax.set(xlim=(-0.3, 4.2), ylim=(-0.3, 4.2), xlabel="feature 1", ylabel="feature 2")
    ax.set_aspect("equal")
    ax.grid(alpha=0.18)
    ax.set_title("A vector has length and direction")
    ax.text(0.05, 3.55, "xᵀw = ||x|| ||w|| cos(θ)", color=NAVY, fontsize=12)
    finish(fig, "vector-dot-product-geometry.png")


def plot_gradient_descent_landscape() -> None:
    w = np.linspace(-1.0, 5.0, 300)
    b = np.linspace(-2.0, 5.0, 300)
    ww, bb = np.meshgrid(w, b)
    loss = (ww - 2.0) ** 2 + 0.45 * (bb - 1.0) ** 2 + 0.22 * (ww - 2.0) * (bb - 1.0)
    path = [np.array([-0.5, 4.5])]
    for _ in range(10):
        pw, pb = path[-1]
        grad = np.array([2 * (pw - 2) + 0.22 * (pb - 1), 0.9 * (pb - 1) + 0.22 * (pw - 2)])
        path.append(path[-1] - 0.28 * grad)
    path_array = np.array(path)
    fig, ax = plt.subplots(figsize=(7.4, 5.1))
    contours = ax.contour(ww, bb, loss, levels=16, cmap="Blues")
    ax.clabel(contours, inline=True, fontsize=7)
    ax.plot(path_array[:, 0], path_array[:, 1], "o-", color=RED, lw=2.2, ms=5, label="gradient updates")
    ax.scatter([2], [1], marker="*", s=180, color=GREEN, label="minimum")
    ax.annotate("start", path_array[0], xytext=(-0.85, 4.75), color=RED, fontweight="bold")
    ax.set(title="Loss landscape: each contour has equal loss", xlabel="weight w", ylabel="bias b")
    ax.legend(frameon=False)
    finish(fig, "gradient-descent-landscape.png")


def plot_mean_variance() -> None:
    x = np.linspace(-5, 5, 500)
    density = lambda value, mean, std: np.exp(-0.5 * ((value - mean) / std) ** 2) / (std * np.sqrt(2 * np.pi))
    fig, ax = plt.subplots(figsize=(8.2, 4.2))
    ax.plot(x, density(x, 0, 0.75), color=BLUE, lw=2.7, label="same mean, small variance")
    ax.plot(x, density(x, 0, 1.7), color=ORANGE, lw=2.7, label="same mean, large variance")
    ax.axvline(0, color=GREEN, ls="--", lw=2, label="mean = 0")
    ax.fill_between(x, density(x, 0, 0.75), alpha=0.12, color=BLUE)
    ax.fill_between(x, density(x, 0, 1.7), alpha=0.12, color=ORANGE)
    ax.set(title="Mean locates the center; variance measures spread", xlabel="possible value", ylabel="probability density")
    ax.legend(frameon=False)
    ax.grid(alpha=0.15)
    finish(fig, "mean-and-variance.png")


def main() -> None:
    set_style()
    plot_rent_distribution()
    plot_fit_and_residuals()
    plot_model_complexity()
    plot_learning_curves()
    plot_confusion_matrix()
    plot_roc_and_pr_curves()
    plot_dataset_anatomy()
    plot_vector_geometry()
    plot_gradient_descent_landscape()
    plot_mean_variance()
    print(f"Generated 10 figures in {OUTPUT}")


if __name__ == "__main__":
    main()
