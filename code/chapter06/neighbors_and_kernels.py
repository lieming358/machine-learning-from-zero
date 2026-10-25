"""Chapter 6: kNN and kernels from scratch, with reproducible figures."""

from __future__ import annotations

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_moons
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

OUTPUT = Path(__file__).resolve().parents[2] / "docs" / "assets"


def pairwise_squared_distance(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.maximum(np.sum(a * a, axis=1)[:, None] + np.sum(b * b, axis=1)[None, :] - 2 * a @ b.T, 0)


class KNNClassifier:
    def __init__(self, k: int = 5) -> None: self.k = k
    def fit(self, x: np.ndarray, y: np.ndarray) -> "KNNClassifier": self.x, self.y = x.copy(), y.copy(); return self
    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        nearest = np.argpartition(pairwise_squared_distance(x, self.x), self.k - 1, axis=1)[:, : self.k]
        positive = self.y[nearest].mean(axis=1)
        return np.column_stack([1 - positive, positive])
    def predict(self, x: np.ndarray) -> np.ndarray: return (self.predict_proba(x)[:, 1] >= 0.5).astype(int)


def rbf_kernel(a: np.ndarray, b: np.ndarray, gamma: float) -> np.ndarray:
    return np.exp(-gamma * pairwise_squared_distance(a, b))


def plot_boundary(axis, model, x, y, title: str) -> None:
    xx, yy = np.meshgrid(np.linspace(-2.2, 3.2, 250), np.linspace(-1.7, 2.2, 250))
    grid = np.c_[xx.ravel(), yy.ravel()]
    probability = model.predict_proba(grid)[:, 1].reshape(xx.shape)
    axis.contourf(xx, yy, probability, levels=np.linspace(0, 1, 11), cmap="RdBu_r", alpha=.28)
    axis.contour(xx, yy, probability, levels=[.5], colors="#222", linewidths=2)
    axis.scatter(x[y == 0, 0], x[y == 0, 1], color="#3478c0", edgecolor="white", alpha=.75)
    axis.scatter(x[y == 1, 0], x[y == 1, 1], color="#f28e2b", edgecolor="white", alpha=.75)
    axis.set(title=title, xlabel="feature 1", ylabel="feature 2")
    axis.spines[["top", "right"]].set_visible(False)


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    x, y = make_moons(n_samples=360, noise=.24, random_state=7)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.3, stratify=y, random_state=19)
    scaler = StandardScaler().fit(x_train)
    train, test = scaler.transform(x_train), scaler.transform(x_test)

    scratch = KNNClassifier(7).fit(train, y_train)
    library = KNeighborsClassifier(7).fit(train, y_train)
    agreement = np.mean(scratch.predict(test) == library.predict(test))
    print(f"scratch_test_accuracy={accuracy_score(y_test, scratch.predict(test)):.3f}")
    print(f"sklearn_prediction_agreement={agreement:.3f}")

    # Figure 0: one prediction, step by step.
    query = np.array([[0.15, 0.25]])
    distances = np.sqrt(pairwise_squared_distance(query, train)[0])
    neighbor_index = np.argsort(distances)[:7]
    fig, ax = plt.subplots(figsize=(7.1, 5.0))
    ax.scatter(train[y_train == 0, 0], train[y_train == 0, 1], color="#3478c0", alpha=.35, label="class 0 history")
    ax.scatter(train[y_train == 1, 0], train[y_train == 1, 1], color="#f28e2b", alpha=.35, label="class 1 history")
    for index in neighbor_index:
        ax.plot([query[0, 0], train[index, 0]], [query[0, 1], train[index, 1]], color="#777777", lw=1, ls="--", alpha=.7)
    ax.scatter(train[neighbor_index, 0], train[neighbor_index, 1], s=145, facecolors="none", edgecolors="#222222", lw=2, label="7 nearest neighbors")
    ax.scatter(query[:, 0], query[:, 1], marker="*", s=300, color="#d94b4b", edgecolor="white", label="new query")
    class_one_votes = int(y_train[neighbor_index].sum())
    class_zero_votes = 7 - class_one_votes
    prediction = int(class_one_votes > class_zero_votes)
    ax.set(
        title=(f"Neighbor vote: class 0 = {class_zero_votes}, class 1 = {class_one_votes}"
               f"  →  predict class {prediction}"),
        xlabel="standardized feature 1",
        ylabel="standardized feature 2",
    )
    ax.legend(frameon=False, loc="upper right"); ax.grid(alpha=.12); ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(OUTPUT / "knn-one-prediction.png", dpi=180, bbox_inches="tight", facecolor="white"); plt.close(fig)

    # Figure 1: k controls smoothness.
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharex=True, sharey=True)
    for axis, k in zip(axes, [1, 7, 45]):
        plot_boundary(axis, KNNClassifier(k).fit(train, y_train), train, y_train, f"k={k}")
    fig.suptitle("k controls the bias–variance trade-off")
    fig.tight_layout(); fig.savefig(OUTPUT / "knn-k-boundaries.png", dpi=180, bbox_inches="tight", facecolor="white"); plt.close(fig)

    # Figure 2: choose k with validation, never test.
    x_fit, x_val, y_fit, y_val = train_test_split(train, y_train, test_size=.3, stratify=y_train, random_state=23)
    ks = np.arange(1, 52, 2); fit_scores, val_scores = [], []
    for k in ks:
        model = KNNClassifier(int(k)).fit(x_fit, y_fit)
        fit_scores.append(accuracy_score(y_fit, model.predict(x_fit)))
        val_scores.append(accuracy_score(y_val, model.predict(x_val)))
    fig, ax = plt.subplots(figsize=(7.3, 4.4)); ax.plot(ks, fit_scores, marker="o", label="training", color="#3478c0"); ax.plot(ks, val_scores, marker="o", label="validation", color="#f28e2b")
    ax.set(xlabel="number of neighbors k", ylabel="accuracy", title="Select k on validation data"); ax.legend(frameon=False); ax.grid(alpha=.15); ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(OUTPUT / "knn-k-validation.png", dpi=180, bbox_inches="tight", facecolor="white"); plt.close(fig)

    # Figure 3: units can dominate Euclidean distance.
    query = np.array([70., 2.]); candidates = np.array([[72., 5.], [85., 2.1]])
    raw = np.sqrt(np.sum((candidates - query) ** 2, axis=1))
    # Representative training-set scales: 20 m² for area and 1 room for bedrooms.
    feature_scale = np.array([20., 1.])
    scaled = np.sqrt(np.sum(((candidates - query) / feature_scale) ** 2, axis=1))
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for axis, values, title in [(axes[0], raw, "raw units"), (axes[1], scaled, "after standardization")]:
        axis.bar(["candidate A", "candidate B"], values, color=["#3478c0", "#f28e2b"]); axis.set(title=title, ylabel="distance to query"); axis.spines[["top", "right"]].set_visible(False); axis.grid(axis="y", alpha=.15)
    fig.suptitle("Distance depends on feature scale"); fig.tight_layout(); fig.savefig(OUTPUT / "distance-scaling-effect.png", dpi=180, bbox_inches="tight", facecolor="white"); plt.close(fig)

    # Figure 4: RBF similarity is a soft neighborhood.
    distance = np.linspace(0, 3, 300); fig, ax = plt.subplots(figsize=(7.3, 4.4))
    for gamma, color in zip([.3, 1., 4.], ["#3478c0", "#3a9d5d", "#f28e2b"]): ax.plot(distance, np.exp(-gamma * distance**2), lw=2.5, label=f"gamma={gamma}", color=color)
    ax.set(xlabel="distance ||x-z||", ylabel="RBF similarity", title="The RBF kernel turns distance into soft similarity"); ax.legend(frameon=False); ax.grid(alpha=.15); ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(OUTPUT / "rbf-similarity.png", dpi=180, bbox_inches="tight", facecolor="white"); plt.close(fig)
    assert agreement == 1.0
    assert np.allclose(rbf_kernel(train[:4], train[:5], 1), rbf_kernel(train[:5], train[:4], 1).T)


if __name__ == "__main__": main()
