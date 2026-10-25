"""Chapter 4: binary logistic regression from scratch and with scikit-learn."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def sigmoid(z: np.ndarray) -> np.ndarray:
    """Numerically stable enough for this lesson through clipping."""
    return 1.0 / (1.0 + np.exp(-np.clip(z, -40.0, 40.0)))


def loss_and_gradients(
    x: np.ndarray,
    y: np.ndarray,
    w: np.ndarray,
    b: float,
    l2: float = 0.0,
) -> tuple[float, np.ndarray, float]:
    """Binary cross-entropy from logits plus L2, with analytic gradients."""
    logits = x @ w + b
    loss = float(np.mean(np.logaddexp(0.0, logits) - y * logits))
    loss += float(l2 * (w @ w) / (2.0 * len(x)))
    error = sigmoid(logits) - y
    dw = x.T @ error / len(x) + l2 * w / len(x)
    db = float(np.mean(error))
    return loss, dw, db


def numerical_gradient(
    x: np.ndarray, y: np.ndarray, w: np.ndarray, b: float, eps: float = 1e-6
) -> tuple[np.ndarray, float]:
    dw = np.zeros_like(w)
    for j in range(len(w)):
        plus, minus = w.copy(), w.copy()
        plus[j] += eps
        minus[j] -= eps
        dw[j] = (
            loss_and_gradients(x, y, plus, b)[0]
            - loss_and_gradients(x, y, minus, b)[0]
        ) / (2.0 * eps)
    db = (
        loss_and_gradients(x, y, w, b + eps)[0]
        - loss_and_gradients(x, y, w, b - eps)[0]
    ) / (2.0 * eps)
    return dw, float(db)


def make_data(seed: int = 17) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    negative = rng.normal(loc=(-1.2, -0.8), scale=(1.0, 0.9), size=(160, 2))
    positive = rng.normal(loc=(1.2, 1.0), scale=(1.0, 0.9), size=(160, 2))
    x = np.vstack([negative, positive])
    y = np.concatenate([np.zeros(len(negative)), np.ones(len(positive))])
    return x, y


def save_figures(
    x: np.ndarray,
    y: np.ndarray,
    w: np.ndarray,
    b: float,
    loss_history: list[float],
    output: Path,
) -> None:
    output.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    for label, color, name in [(0, "#3478c0", "class 0"), (1, "#f28e2b", "class 1")]:
        selected = y == label
        ax.scatter(x[selected, 0], x[selected, 1], alpha=0.62, color=color, label=name, edgecolor="white")
    x_line = np.linspace(x[:, 0].min(), x[:, 0].max(), 200)
    y_line = -(w[0] * x_line + b) / w[1]
    ax.plot(x_line, y_line, color="#d94b4b", lw=2.5, label="p(y=1|x)=0.5 boundary")
    ax.set(xlabel="standardized feature 1", ylabel="standardized feature 2", title="A probability model creates a decision boundary")
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(alpha=0.15)
    fig.tight_layout()
    fig.savefig(output / "logistic-decision-boundary.png", dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    probability = sigmoid(x @ w + b)
    thresholds = np.linspace(0.05, 0.95, 37)
    precision, recall = [], []
    for threshold in thresholds:
        prediction = probability >= threshold
        precision.append(precision_score(y, prediction, zero_division=0))
        recall.append(recall_score(y, prediction, zero_division=0))
    fig, ax = plt.subplots(figsize=(7.4, 4.5))
    ax.plot(thresholds, precision, color="#3478c0", lw=2.5, label="precision")
    ax.plot(thresholds, recall, color="#f28e2b", lw=2.5, label="recall")
    ax.axvline(0.5, color="#3a9d5d", linestyle="--", label="default threshold 0.5")
    ax.set(xlabel="decision threshold", ylabel="metric", ylim=(0, 1.03), title="Threshold changes decisions, not learned probabilities")
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(alpha=0.15)
    fig.tight_layout()
    fig.savefig(output / "classification-threshold-tradeoff.png", dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.4, 4.5))
    steps = np.arange(len(loss_history)) * 50
    ax.plot(steps, loss_history, color="#6f4dbf", lw=2.5)
    ax.set(
        xlabel="gradient-descent step",
        ylabel="training binary cross-entropy",
        title="Training converges when the loss flattens",
    )
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(alpha=0.15)
    fig.tight_layout()
    fig.savefig(output / "logistic-training-loss.png", dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    x, y = make_data()
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.25, random_state=42, stratify=y
    )
    scaler = StandardScaler().fit(x_train)
    x_train_scaled = scaler.transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    w = np.zeros(x.shape[1])
    b = 0.0
    analytic = loss_and_gradients(x_train_scaled, y_train, w, b)[1:]
    numeric = numerical_gradient(x_train_scaled, y_train, w, b)
    gradient_error = max(
        float(np.max(np.abs(analytic[0] - numeric[0]))), abs(analytic[1] - numeric[1])
    )

    loss_history: list[float] = []
    for step in range(10_000):
        loss, dw, db = loss_and_gradients(x_train_scaled, y_train, w, b, l2=0.0)
        if step % 50 == 0:
            loss_history.append(loss)
        w -= 0.08 * dw
        b -= 0.08 * db
    loss_history.append(loss_and_gradients(x_train_scaled, y_train, w, b)[0])

    probability = sigmoid(x_test_scaled @ w + b)
    prediction = probability >= 0.5
    framework = LogisticRegression(C=1e6).fit(x_train_scaled, y_train)
    framework_probability = framework.predict_proba(x_test_scaled)[:, 1]

    print(f"gradient_check_max_error={gradient_error:.3e}")
    print(f"final_train_loss={loss:.4f}")
    print(f"test_accuracy={accuracy_score(y_test, prediction):.3f}")
    print(f"test_precision={precision_score(y_test, prediction):.3f}")
    print(f"test_recall={recall_score(y_test, prediction):.3f}")
    print(f"sklearn_probability_max_gap={np.max(np.abs(probability-framework_probability)):.4f}")

    output = Path(__file__).resolve().parents[2] / "docs" / "assets"
    all_x_scaled = scaler.transform(x)
    save_figures(all_x_scaled, y, w, b, loss_history, output)
    assert gradient_error < 1e-6
    assert accuracy_score(y_test, prediction) > 0.8
    assert np.max(np.abs(probability - framework_probability)) < 0.03


if __name__ == "__main__":
    main()
