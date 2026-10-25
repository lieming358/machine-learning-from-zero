"""Chapter 10: from-scratch k-means and one-dimensional Gaussian-mixture EM."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.metrics import silhouette_score


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "docs" / "assets"
COLORS = np.array(["#3478c0", "#f28e2b", "#3a9d5d", "#d94b4b", "#8f63b8"])
NAVY = "#19324d"


def assign(x, centers):
    distance2 = ((x[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
    return distance2.argmin(axis=1)


def update(x, labels, k, old_centers):
    return np.vstack([x[labels == cluster].mean(axis=0) if np.any(labels == cluster)
                      else old_centers[cluster] for cluster in range(k)])


def scratch_kmeans(x, k=3, seed=4, max_iter=100):
    rng = np.random.default_rng(seed)
    centers = x[rng.choice(len(x), k, replace=False)].copy()
    history = [centers.copy()]
    for _ in range(max_iter):
        labels = assign(x, centers)
        new_centers = update(x, labels, k, centers)
        history.append(new_centers.copy())
        if np.allclose(new_centers, centers):
            break
        centers = new_centers
    return centers, assign(x, centers), history


def normal_pdf(x, mean, variance):
    return np.exp(-.5 * (x - mean) ** 2 / variance) / np.sqrt(2 * np.pi * variance)


def gmm_em(x, k=2, iterations=60):
    means = np.quantile(x, [.3, .7]).astype(float)
    variances = np.full(k, np.var(x))
    weights = np.full(k, 1 / k)
    log_likelihood = []
    responsibilities = None
    for _ in range(iterations):
        joint = np.column_stack([weights[j] * normal_pdf(x, means[j], variances[j]) for j in range(k)])
        total = joint.sum(axis=1, keepdims=True) + 1e-12
        responsibilities = joint / total
        log_likelihood.append(np.log(total[:, 0]).sum())
        counts = responsibilities.sum(axis=0)
        weights = counts / len(x)
        means = (responsibilities * x[:, None]).sum(axis=0) / counts
        variances = (responsibilities * (x[:, None] - means) ** 2).sum(axis=0) / counts + 1e-6
    return weights, means, variances, responsibilities, np.array(log_likelihood)


def finish(fig, name):
    fig.tight_layout(); fig.savefig(OUTPUT / name, dpi=180, bbox_inches="tight", facecolor="white"); plt.close(fig)


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    x, _ = make_blobs(n_samples=330, centers=[(-2.3,-1.2), (.3,2.2), (2.5,-.4)],
                      cluster_std=[.65,.72,.58], random_state=12)
    centers, labels, history = scratch_kmeans(x)
    snapshots = [0, min(1, len(history)-1), len(history)-1]
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.8), sharex=True, sharey=True)
    for axis, step, title in zip(axes, snapshots, ["Initialization", "After one update", "Converged"]):
        current = history[step]
        current_labels = assign(x, current)
        axis.scatter(x[:,0], x[:,1], c=COLORS[current_labels], s=20, alpha=.55, edgecolor="none")
        axis.scatter(current[:,0], current[:,1], marker="X", s=170, color=NAVY, edgecolor="white", label="centroids")
        axis.set(title=title, xlabel="feature 1"); axis.grid(alpha=.13)
    axes[0].set_ylabel("feature 2"); axes[-1].legend(frameon=False)
    finish(fig, "kmeans-iterations.png")

    candidates = np.arange(2, 7); inertia, silhouette = [], []
    for k in candidates:
        fitted = KMeans(n_clusters=int(k), n_init=10, random_state=4).fit(x)
        inertia.append(fitted.inertia_); silhouette.append(silhouette_score(x, fitted.labels_))
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(candidates, inertia, "o-", color=COLORS[0]); axes[0].set(title="Inertia always decreases", xlabel="k", ylabel="within-cluster squared distance")
    axes[1].plot(candidates, silhouette, "o-", color=COLORS[1]); axes[1].axvline(candidates[np.argmax(silhouette)], ls="--", color=COLORS[2]); axes[1].set(title="Silhouette balances cohesion and separation", xlabel="k", ylabel="silhouette score")
    for axis in axes: axis.grid(alpha=.15)
    finish(fig, "clustering-model-selection.png")

    rng = np.random.default_rng(8)
    one_d = np.r_[rng.normal(-1.6, .65, 190), rng.normal(1.35, .9, 260)]
    weights, means, variances, resp, likelihood = gmm_em(one_d)
    order = np.argsort(means); weights, means, variances, resp = weights[order], means[order], variances[order], resp[:, order]
    grid = np.linspace(-4.5, 4.5, 500)
    fig, ax = plt.subplots(figsize=(9, 4.4))
    ax.hist(one_d, bins=30, density=True, color="#dbe7f3", edgecolor="white", label="observed data")
    mixture = np.zeros_like(grid)
    for component in range(2):
        density = weights[component] * normal_pdf(grid, means[component], variances[component])
        mixture += density
        ax.plot(grid, density, lw=2, color=COLORS[component], label=f"component {component+1}")
    ax.plot(grid, mixture, lw=2.7, color=NAVY, label="mixture density")
    uncertain = np.argsort(np.abs(resp[:,0]-.5))[:20]
    ax.scatter(one_d[uncertain], np.zeros(len(uncertain))+.008, c=resp[uncertain,0], cmap="coolwarm", marker="|", s=120, label="uncertain assignments")
    ax.set(title="A GMM assigns soft responsibilities instead of hard cluster labels", xlabel="x", ylabel="density")
    ax.legend(frameon=False, ncol=2); ax.grid(alpha=.12)
    finish(fig, "gmm-soft-responsibilities.png")

    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.plot(np.arange(1, len(likelihood)+1), likelihood, color=COLORS[2], lw=2.5)
    ax.set(title="EM improves the observed-data log likelihood", xlabel="EM iteration", ylabel="log likelihood")
    ax.grid(alpha=.15); finish(fig, "em-log-likelihood.png")

    sklearn_centers = KMeans(n_clusters=3, n_init=20, random_state=4).fit(x).cluster_centers_
    center_error = np.mean(np.min(((centers[:,None,:]-sklearn_centers[None,:,:])**2).sum(axis=2), axis=1) ** .5)
    print(f"kmeans_iterations={len(history)-1}")
    print(f"center_match_error={center_error:.4f}")
    print(f"gmm_means={means.round(3).tolist()}")
    print(f"em_log_likelihood_gain={likelihood[-1]-likelihood[0]:.3f}")
    assert center_error < .12 and np.all(np.diff(likelihood) >= -1e-7)


if __name__ == "__main__":
    main()
