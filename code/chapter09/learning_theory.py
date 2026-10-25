"""Chapter 9: model capacity, cross-validation, and learning curves."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import KFold, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "docs" / "assets"
BLUE, ORANGE, GREEN, RED, NAVY = "#3478c0", "#f28e2b", "#3a9d5d", "#d94b4b", "#19324d"


def model(degree: int, alpha: float = 1e-6):
    return make_pipeline(PolynomialFeatures(degree, include_bias=False), StandardScaler(), Ridge(alpha=alpha))


def make_data(seed: int = 19):
    rng = np.random.default_rng(seed)
    x = np.sort(rng.uniform(-3, 3, 120))[:, None]
    y = np.sin(1.35 * x[:, 0]) + 0.22 * x[:, 0] + rng.normal(0, 0.28, len(x))
    return train_test_split(x, y, test_size=.25, random_state=7)


def finish(fig, name):
    fig.tight_layout()
    fig.savefig(OUTPUT / name, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    x_train, x_test, y_train, y_test = make_data()
    grid = np.linspace(-3.1, 3.1, 500)[:, None]

    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), sharey=True)
    for axis, degree, title, color in zip(
        axes, [1, 5, 18], ["Too simple", "Useful capacity", "Too flexible"], [ORANGE, GREEN, RED]
    ):
        fitted = model(degree).fit(x_train, y_train)
        axis.scatter(x_train, y_train, s=24, alpha=.65, color=BLUE, edgecolor="white", label="training data")
        axis.plot(grid, fitted.predict(grid), lw=2.4, color=color, label=f"degree {degree}")
        axis.set(title=title, xlabel="x", ylim=(-2.2, 2.2))
        axis.grid(alpha=.15); axis.legend(frameon=False, fontsize=8)
    axes[0].set_ylabel("target y")
    finish(fig, "capacity-three-regimes.png")

    degrees = np.arange(1, 19)
    train_error, cv_mean, cv_std = [], [], []
    folds = KFold(n_splits=5, shuffle=True, random_state=11)
    for degree in degrees:
        estimator = model(int(degree))
        estimator.fit(x_train, y_train)
        train_error.append(mean_squared_error(y_train, estimator.predict(x_train)))
        fold_errors = []
        for fit_index, val_index in folds.split(x_train):
            candidate = model(int(degree)).fit(x_train[fit_index], y_train[fit_index])
            fold_errors.append(mean_squared_error(y_train[val_index], candidate.predict(x_train[val_index])))
        cv_mean.append(np.mean(fold_errors)); cv_std.append(np.std(fold_errors))
    cv_mean, cv_std = np.array(cv_mean), np.array(cv_std)
    best_degree = int(degrees[np.argmin(cv_mean)])
    fig, ax = plt.subplots(figsize=(8.2, 4.4))
    ax.plot(degrees, train_error, "o-", color=BLUE, label="training MSE")
    ax.plot(degrees, cv_mean, "o-", color=ORANGE, label="5-fold validation MSE")
    ax.fill_between(degrees, cv_mean-cv_std, cv_mean+cv_std, color=ORANGE, alpha=.16, label="fold variation")
    ax.axvline(best_degree, color=GREEN, ls="--", label=f"selected degree = {best_degree}")
    ax.set(title="Capacity is selected by validation, not training fit", xlabel="polynomial degree", ylabel="MSE")
    ax.set_yscale("log"); ax.grid(alpha=.16); ax.legend(frameon=False)
    finish(fig, "capacity-validation-curve.png")

    rng = np.random.default_rng(29)
    sizes = np.array([12, 20, 32, 48, 64, 84])
    fit_scores, validation_scores = [], []
    for size in sizes:
        fit_repeat, val_repeat = [], []
        for _ in range(20):
            index = rng.choice(len(x_train), size=size, replace=False)
            remaining = np.setdiff1d(np.arange(len(x_train)), index)
            candidate = model(best_degree, alpha=1e-3).fit(x_train[index], y_train[index])
            fit_repeat.append(mean_squared_error(y_train[index], candidate.predict(x_train[index])))
            val_repeat.append(mean_squared_error(y_train[remaining], candidate.predict(x_train[remaining])))
        fit_scores.append(np.mean(fit_repeat)); validation_scores.append(np.mean(val_repeat))
    fig, ax = plt.subplots(figsize=(8.2, 4.4))
    ax.plot(sizes, fit_scores, "o-", color=BLUE, lw=2.2, label="training error")
    ax.plot(sizes, validation_scores, "o-", color=ORANGE, lw=2.2, label="held-out error")
    ax.set(title="A learning curve asks whether more data is likely to help", xlabel="number of training examples", ylabel="MSE")
    ax.grid(alpha=.16); ax.legend(frameon=False)
    finish(fig, "learning-curve-data-size.png")

    fig, ax = plt.subplots(figsize=(9.2, 3.3))
    colors = [BLUE, ORANGE, GREEN, RED, "#8f63b8"]
    for fold in range(5):
        for block in range(5):
            ax.add_patch(plt.Rectangle((block, 4-fold), .92, .7,
                         color=colors[block] if block == fold else "#dbe7f3", ec="white"))
            ax.text(block+.46, 4.35-fold, "validate" if block == fold else "train", ha="center", va="center", fontsize=8)
    ax.set(xlim=(-.05, 5), ylim=(-.1, 5.1), xticks=np.arange(5)+.46,
           xticklabels=[f"block {i}" for i in range(1,6)], yticks=np.arange(5)+.35,
           yticklabels=[f"run {i}" for i in range(5,0,-1)])
    ax.set_title("Five-fold cross-validation rotates the validation block")
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    finish(fig, "cross-validation-rotation.png")

    fig, ax = plt.subplots(figsize=(10.2, 3.1))
    blocks = [(0.2, 5.4, "development set\n(train + validation)", "#dbe7f3"),
              (5.8, 2.0, "sealed test set\n(use once)", "#f8d7da")]
    for x0, width, label, color in blocks:
        ax.add_patch(plt.Rectangle((x0, 0.8), width, 1.2, color=color, ec=NAVY, lw=1.2))
        ax.text(x0 + width / 2, 1.4, label, ha="center", va="center", fontsize=11)
    ax.annotate("cross-validation chooses hyperparameters", xy=(2.9, .75), xytext=(2.9, .2),
                ha="center", arrowprops=dict(arrowstyle="->", color=GREEN), color=GREEN)
    ax.annotate("final evaluation only", xy=(6.8, 2.05), xytext=(6.8, 2.65),
                ha="center", arrowprops=dict(arrowstyle="->", color=RED), color=RED)
    ax.set(xlim=(0, 8), ylim=(0, 3), title="Separate model development from final evaluation")
    ax.axis("off")
    finish(fig, "data-split-roles.png")

    final_model = model(best_degree, alpha=1e-3).fit(x_train, y_train)
    test_mse = mean_squared_error(y_test, final_model.predict(x_test))
    print(f"selected_degree={best_degree}")
    print(f"cross_validation_mse={cv_mean.min():.4f}")
    print(f"final_test_mse={test_mse:.4f}")
    assert 2 <= best_degree <= 9 and test_mse < .35


if __name__ == "__main__":
    main()
