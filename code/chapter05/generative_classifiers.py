"""Chapter 5: GDA and multinomial Naive Bayes from scratch.

The script generates every Chapter 5 scientific figure and checks the NumPy
implementations against scikit-learn on the same data.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import StandardScaler


OUTPUT = Path(__file__).resolve().parents[2] / "docs" / "assets"


class GaussianDiscriminantAnalysis:
    """Binary GDA with either one shared covariance or one covariance per class."""

    def __init__(self, shared_covariance: bool = True, jitter: float = 1e-6) -> None:
        self.shared_covariance = shared_covariance
        self.jitter = jitter

    def fit(self, x: np.ndarray, y: np.ndarray) -> "GaussianDiscriminantAnalysis":
        self.prior_ = np.array([np.mean(y == 0), np.mean(y == 1)])
        self.means_ = np.vstack([x[y == c].mean(axis=0) for c in (0, 1)])
        if self.shared_covariance:
            residual = x - self.means_[y.astype(int)]
            covariance = residual.T @ residual / len(x)
            self.covariances_ = np.stack([covariance, covariance])
        else:
            self.covariances_ = np.stack(
                [
                    (x[y == c] - self.means_[c]).T @ (x[y == c] - self.means_[c])
                    / np.sum(y == c)
                    for c in (0, 1)
                ]
            )
        identity = np.eye(x.shape[1])
        self.covariances_ += self.jitter * identity[None, :, :]
        return self

    def _joint_log_likelihood(self, x: np.ndarray) -> np.ndarray:
        values = []
        dimension = x.shape[1]
        for c in (0, 1):
            difference = x - self.means_[c]
            sign, logdet = np.linalg.slogdet(self.covariances_[c])
            if sign <= 0:
                raise ValueError("covariance must be positive definite")
            mahalanobis = np.sum(
                difference * np.linalg.solve(self.covariances_[c], difference.T).T,
                axis=1,
            )
            values.append(
                np.log(self.prior_[c])
                - 0.5 * (dimension * np.log(2.0 * np.pi) + logdet + mahalanobis)
            )
        return np.column_stack(values)

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        logits = self._joint_log_likelihood(x)
        logits -= logits.max(axis=1, keepdims=True)
        probability = np.exp(logits)
        return probability / probability.sum(axis=1, keepdims=True)

    def predict(self, x: np.ndarray) -> np.ndarray:
        return np.argmax(self.predict_proba(x), axis=1)


class MultinomialNaiveBayes:
    """Multinomial Naive Bayes for non-negative word-count features."""

    def __init__(self, alpha: float = 1.0) -> None:
        self.alpha = alpha

    def fit(self, x: np.ndarray, y: np.ndarray) -> "MultinomialNaiveBayes":
        classes = np.unique(y)
        if not np.array_equal(classes, np.arange(len(classes))):
            raise ValueError("this teaching implementation expects classes 0, 1, ...")
        self.class_log_prior_ = np.log([np.mean(y == c) for c in classes])
        counts = np.vstack([x[y == c].sum(axis=0) for c in classes])
        smoothed = counts + self.alpha
        self.feature_log_prob_ = np.log(smoothed / smoothed.sum(axis=1, keepdims=True))
        return self

    def predict_log_proba(self, x: np.ndarray) -> np.ndarray:
        joint = x @ self.feature_log_prob_.T + self.class_log_prior_
        maximum = joint.max(axis=1, keepdims=True)
        log_normalizer = maximum + np.log(np.exp(joint - maximum).sum(axis=1, keepdims=True))
        return joint - log_normalizer

    def predict(self, x: np.ndarray) -> np.ndarray:
        return np.argmax(self.predict_log_proba(x), axis=1)


def make_gda_data(n_per_class: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    x0 = rng.multivariate_normal([-1.1, -0.5], [[1.2, 0.75], [0.75, 1.0]], n_per_class)
    x1 = rng.multivariate_normal([1.2, 0.8], [[0.65, -0.35], [-0.35, 1.25]], n_per_class)
    return np.vstack([x0, x1]), np.r_[np.zeros(n_per_class), np.ones(n_per_class)].astype(int)


def vocabulary_and_counts(documents: list[str]) -> tuple[list[str], np.ndarray]:
    vocabulary = sorted({word for document in documents for word in document.lower().split()})
    index = {word: j for j, word in enumerate(vocabulary)}
    counts = np.zeros((len(documents), len(vocabulary)))
    for i, document in enumerate(documents):
        for word in document.lower().split():
            counts[i, index[word]] += 1
    return vocabulary, counts


def transform_documents(documents: list[str], vocabulary: list[str]) -> np.ndarray:
    index = {word: j for j, word in enumerate(vocabulary)}
    counts = np.zeros((len(documents), len(vocabulary)))
    for i, document in enumerate(documents):
        for word in document.lower().split():
            if word in index:
                counts[i, index[word]] += 1
    return counts


def save_gda_boundaries(x: np.ndarray, y: np.ndarray) -> None:
    models = [
        ("shared covariance → linear boundary", GaussianDiscriminantAnalysis(True).fit(x, y)),
        ("class-specific covariance → curved boundary", GaussianDiscriminantAnalysis(False).fit(x, y)),
    ]
    xx, yy = np.meshgrid(np.linspace(-4, 4, 260), np.linspace(-3.5, 4, 260))
    grid = np.column_stack([xx.ravel(), yy.ravel()])
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.4), sharex=True, sharey=True)
    for axis, (title, model) in zip(axes, models):
        probability = model.predict_proba(grid)[:, 1].reshape(xx.shape)
        axis.contourf(xx, yy, probability, levels=np.linspace(0, 1, 11), cmap="RdBu_r", alpha=0.22)
        axis.contour(xx, yy, probability, levels=[0.5], colors="#202020", linewidths=2.5)
        axis.scatter(x[y == 0, 0], x[y == 0, 1], color="#3478c0", edgecolor="white", alpha=0.72, label="class 0")
        axis.scatter(x[y == 1, 0], x[y == 1, 1], color="#f28e2b", edgecolor="white", alpha=0.72, label="class 1")
        axis.set(title=title, xlabel="feature 1", ylabel="feature 2")
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(alpha=0.12)
    axes[1].legend(frameon=False)
    fig.suptitle("A covariance assumption decides the shape of a GDA boundary")
    fig.tight_layout()
    fig.savefig(OUTPUT / "gda-boundaries.png", dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def save_bimodal_gaussian_failure(seed: int = 41) -> None:
    """Show why one Gaussian is a poor description of a two-cluster class."""
    rng = np.random.default_rng(seed)
    city_small = rng.multivariate_normal([-2.1, 0.0], [[0.18, 0.03], [0.03, 0.22]], 75)
    suburb_large = rng.multivariate_normal([2.1, 0.0], [[0.22, -0.02], [-0.02, 0.18]], 75)
    homes = np.vstack([city_small, suburb_large])
    mean = homes.mean(axis=0)
    covariance = np.cov(homes, rowvar=False, bias=True)

    xx, yy = np.meshgrid(np.linspace(-3.5, 3.5, 320), np.linspace(-2.0, 2.0, 240))
    grid = np.column_stack([xx.ravel(), yy.ravel()])
    difference = grid - mean
    mahalanobis = np.sum(difference * np.linalg.solve(covariance, difference.T).T, axis=1)
    density = np.exp(-0.5 * mahalanobis).reshape(xx.shape)

    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.3), sharex=True, sharey=True)
    axes[0].scatter(homes[:, 0], homes[:, 1], color="#3478c0", alpha=0.65, edgecolor="white")
    axes[0].scatter(*mean, marker="*", s=240, color="#d94b4b", edgecolor="white", label="sample mean")
    axes[0].annotate(
        "mean lies in an empty region",
        xy=mean,
        xytext=(-1.25, 1.45),
        arrowprops={"arrowstyle": "->", "color": "#333333"},
    )
    axes[0].set_title("Reality: one class contains two clusters")
    axes[0].legend(frameon=False, loc="lower center")

    contour = axes[1].contourf(xx, yy, density, levels=14, cmap="Blues", alpha=0.8)
    axes[1].scatter(homes[:, 0], homes[:, 1], color="#173f6d", alpha=0.38, s=20)
    axes[1].scatter(*mean, marker="*", s=240, color="#d94b4b", edgecolor="white")
    axes[1].annotate(
        "single Gaussian assigns\nhighest density here",
        xy=mean,
        xytext=(0.65, 1.35),
        arrowprops={"arrowstyle": "->", "color": "#333333"},
    )
    axes[1].set_title("Wrong assumption: one Gaussian fills the gap")
    fig.colorbar(contour, ax=axes[1], label="relative fitted density", fraction=0.046)
    for axis in axes:
        axis.set(
            xlabel="location/size direction: city-small  ←  →  suburb-large",
            ylabel="second housing feature",
        )
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(alpha=0.12)
    fig.suptitle("Optimization succeeds, but the chosen distribution family is wrong")
    fig.tight_layout()
    fig.savefig(OUTPUT / "single-gaussian-bimodal-failure.png", dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def save_sample_efficiency_curve() -> None:
    sizes = np.array([5, 10, 20, 40, 80, 160])
    repeats = 24
    gda_scores = np.zeros((len(sizes), repeats))
    logistic_scores = np.zeros_like(gda_scores)
    x_test, y_test = make_gda_data(1200, seed=999)
    for i, size in enumerate(sizes):
        for repeat in range(repeats):
            x_train, y_train = make_gda_data(int(size), seed=1000 + 31 * i + repeat)
            scaler = StandardScaler().fit(x_train)
            train_scaled, test_scaled = scaler.transform(x_train), scaler.transform(x_test)
            gda = GaussianDiscriminantAnalysis(True).fit(train_scaled, y_train)
            logistic = LogisticRegression(C=1e6, max_iter=2000).fit(train_scaled, y_train)
            gda_scores[i, repeat] = accuracy_score(y_test, gda.predict(test_scaled))
            logistic_scores[i, repeat] = accuracy_score(y_test, logistic.predict(test_scaled))
    fig, ax = plt.subplots(figsize=(7.4, 4.5))
    for scores, color, label in [
        (gda_scores, "#3a9d5d", "GDA with a strong Gaussian assumption"),
        (logistic_scores, "#6f4dbf", "logistic regression"),
    ]:
        mean, std = scores.mean(axis=1), scores.std(axis=1)
        ax.plot(sizes * 2, mean, marker="o", lw=2.5, color=color, label=label)
        ax.fill_between(sizes * 2, mean - std, mean + std, color=color, alpha=0.14)
    ax.set(xlabel="training examples", ylabel="test accuracy", ylim=(0.68, 0.91), title="A strong assumption can help with little data; both converge here")
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(alpha=0.15)
    fig.tight_layout()
    fig.savefig(OUTPUT / "generative-vs-discriminative-learning-curve.png", dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def save_naive_bayes_figures(vocabulary: list[str], x: np.ndarray, y: np.ndarray) -> None:
    selected = ["free", "winner", "meeting", "project", "invoice", "offer"]
    indices = [vocabulary.index(word) for word in selected]
    raw = np.vstack([x[y == c].sum(axis=0) for c in (0, 1)])
    unsmoothed = raw / raw.sum(axis=1, keepdims=True)
    smoothed = (raw + 1.0) / (raw.sum(axis=1, keepdims=True) + len(vocabulary))

    positions = np.arange(len(selected))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for axis, probability, title in [
        (axes[0], unsmoothed, "MLE: unseen words receive probability 0"),
        (axes[1], smoothed, "Laplace smoothing: add one virtual count"),
    ]:
        axis.bar(positions - 0.18, probability[0, indices], 0.36, label="normal", color="#3478c0")
        axis.bar(positions + 0.18, probability[1, indices], 0.36, label="spam", color="#f28e2b")
        axis.set_xticks(positions, selected, rotation=20)
        axis.set(title=title, ylabel="estimated P(word | class)")
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(axis="y", alpha=0.15)
    axes[1].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(OUTPUT / "laplace-smoothing-word-probabilities.png", dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    evidence = np.log(smoothed[1, indices]) - np.log(smoothed[0, indices])
    colors = np.where(evidence >= 0, "#f28e2b", "#3478c0")
    fig, ax = plt.subplots(figsize=(7.2, 4.3))
    ax.barh(selected, evidence, color=colors)
    ax.axvline(0, color="#333333", lw=1)
    ax.set(xlabel="log evidence for spam  log P(word|spam) − log P(word|normal)", title="Naive Bayes adds evidence from each observed word")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="x", alpha=0.15)
    fig.tight_layout()
    fig.savefig(OUTPUT / "naive-bayes-word-evidence.png", dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    x, y = make_gda_data(180, seed=13)
    order = np.random.default_rng(8).permutation(len(x))
    train, test = order[:260], order[260:]
    scratch_lda = GaussianDiscriminantAnalysis(True).fit(x[train], y[train])
    scratch_qda = GaussianDiscriminantAnalysis(False).fit(x[train], y[train])
    library_lda = LinearDiscriminantAnalysis().fit(x[train], y[train])
    library_qda = QuadraticDiscriminantAnalysis(reg_param=1e-6).fit(x[train], y[train])
    print(f"scratch_LDA_accuracy={accuracy_score(y[test], scratch_lda.predict(x[test])):.3f}")
    print(f"scratch_QDA_accuracy={accuracy_score(y[test], scratch_qda.predict(x[test])):.3f}")
    print(f"LDA_probability_max_gap={np.max(np.abs(scratch_lda.predict_proba(x[test]) - library_lda.predict_proba(x[test]))):.3e}")
    print(f"QDA_prediction_agreement={np.mean(scratch_qda.predict(x[test]) == library_qda.predict(x[test])):.3f}")

    documents = [
        "project meeting schedule", "team meeting project", "invoice project update",
        "family dinner schedule", "meeting notes invoice", "project review tomorrow",
        "free winner offer", "winner prize free", "urgent offer winner",
        "free prize claim", "limited offer free", "winner claim prize",
    ]
    labels = np.r_[np.zeros(6), np.ones(6)].astype(int)
    vocabulary, counts = vocabulary_and_counts(documents)
    scratch_nb = MultinomialNaiveBayes(alpha=1.0).fit(counts, labels)
    library_nb = MultinomialNB(alpha=1.0).fit(counts, labels)
    test_documents = ["free prize offer", "project meeting tomorrow", "free invoice offer"]
    test_counts = transform_documents(test_documents, vocabulary)
    gap = np.max(np.abs(scratch_nb.predict_log_proba(test_counts) - library_nb.predict_log_proba(test_counts)))
    print(f"NB_log_probability_max_gap={gap:.3e}")
    for document, prediction in zip(test_documents, scratch_nb.predict(test_counts)):
        print(f"NB_prediction={prediction} text={document!r}")

    save_gda_boundaries(x[train], y[train])
    save_bimodal_gaussian_failure()
    save_sample_efficiency_curve()
    save_naive_bayes_figures(vocabulary, counts, labels)
    # Libraries may use a slightly different covariance normalization convention.
    assert np.max(np.abs(scratch_lda.predict_proba(x[test]) - library_lda.predict_proba(x[test]))) < 0.01
    assert np.mean(scratch_qda.predict(x[test]) == library_qda.predict(x[test])) > 0.98
    assert gap < 1e-10


if __name__ == "__main__":
    main()
