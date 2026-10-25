"""Chapter 11: PCA from scratch, reconstruction, and ICA separation."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_digits
from sklearn.decomposition import FastICA, PCA


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "docs" / "assets"
BLUE, ORANGE, GREEN, RED, NAVY = "#3478c0", "#f28e2b", "#3a9d5d", "#d94b4b", "#19324d"


def scratch_pca(x, components):
    mean = x.mean(axis=0)
    centered = x - mean
    _, singular_values, vt = np.linalg.svd(centered, full_matrices=False)
    directions = vt[:components]
    scores = centered @ directions.T
    return mean, directions, scores, singular_values


def finish(fig, name):
    fig.tight_layout(); fig.savefig(OUTPUT / name, dpi=180, bbox_inches="tight", facecolor="white"); plt.close(fig)


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(14)
    latent = rng.normal(0, 1.25, 180)
    x2 = np.column_stack([latent + rng.normal(0,.25,len(latent)),
                          .72*latent + rng.normal(0,.32,len(latent))])
    mean, directions, scores, singular = scratch_pca(x2, 1)
    reconstructed = scores @ directions + mean
    line_t = np.linspace(scores.min(), scores.max(), 200)
    line = line_t[:,None] @ directions + mean
    fig, ax = plt.subplots(figsize=(7, 5.2))
    ax.scatter(x2[:,0], x2[:,1], s=25, alpha=.55, color=BLUE, label="observations")
    ax.plot(line[:,0], line[:,1], color=RED, lw=2.5, label="first principal direction")
    for index in np.arange(0, len(x2), 12):
        ax.plot([x2[index,0], reconstructed[index,0]], [x2[index,1], reconstructed[index,1]],
                color="#777777", ls="--", lw=.8)
    ax.set(title="PCA keeps the direction with the largest projected variation", xlabel="feature 1", ylabel="feature 2")
    ax.set_aspect("equal"); ax.grid(alpha=.15); ax.legend(frameon=False)
    finish(fig, "pca-projection-reconstruction.png")

    digits = load_digits()
    x = digits.data / 16.0
    pca = PCA().fit(x)
    cumulative = np.cumsum(pca.explained_variance_ratio_)
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4))
    axes[0].bar(np.arange(1,21), pca.explained_variance_ratio_[:20], color=BLUE)
    axes[0].set(title="Scree plot", xlabel="principal component", ylabel="explained variance ratio")
    axes[1].plot(np.arange(1,len(cumulative)+1), cumulative, color=ORANGE, lw=2.4)
    axes[1].axhline(.9, color=GREEN, ls="--", label="90% variance")
    axes[1].set(title="Cumulative retained variation", xlabel="number of components", ylabel="cumulative ratio")
    axes[1].legend(frameon=False)
    for axis in axes: axis.grid(alpha=.14)
    finish(fig, "pca-explained-variance.png")

    sample_indices = [0, 10, 25, 42]
    component_counts = [8, 20]
    fig, axes = plt.subplots(3, len(sample_indices), figsize=(8, 6))
    for col, index in enumerate(sample_indices):
        axes[0,col].imshow(x[index].reshape(8,8), cmap="gray_r", vmin=0, vmax=1)
        axes[0,col].set_title(f"digit {digits.target[index]}")
        for row, count in enumerate(component_counts, start=1):
            reduced = PCA(n_components=count, random_state=0).fit(x)
            restored = reduced.inverse_transform(reduced.transform(x[index:index+1]))[0]
            axes[row,col].imshow(restored.reshape(8,8), cmap="gray_r", vmin=0, vmax=1)
            if col == 0: axes[row,col].set_ylabel(f"{count} PCs")
        for row in range(3): axes[row,col].set_xticks([]); axes[row,col].set_yticks([])
    axes[0,0].set_ylabel("original")
    fig.suptitle("Compression removes low-variance detail", color=NAVY, fontweight="bold")
    finish(fig, "pca-digit-reconstruction.png")

    time = np.linspace(0, 8, 1600)
    sources = np.column_stack([np.sin(2*time), np.sign(np.sin(3.1*time))])
    sources += .03 * rng.normal(size=sources.shape)
    mixing = np.array([[1.0,.65],[.45,1.15]])
    mixtures = sources @ mixing.T
    recovered = FastICA(n_components=2, random_state=3, whiten="unit-variance", max_iter=1000).fit_transform(mixtures)
    fig, axes = plt.subplots(3, 1, figsize=(10, 6), sharex=True)
    for j in range(2): axes[0].plot(time[:500], sources[:500,j], lw=1.2, label=f"source {j+1}")
    for j in range(2): axes[1].plot(time[:500], mixtures[:500,j], lw=1.2, label=f"mixture {j+1}")
    for j in range(2): axes[2].plot(time[:500], recovered[:500,j], lw=1.2, label=f"recovered {j+1}")
    for axis, title in zip(axes, ["Independent sources", "Observed mixtures", "ICA recovered signals"]):
        axis.set_title(title); axis.legend(frameon=False, ncol=2); axis.grid(alpha=.12)
    axes[2].set_xlabel("time")
    finish(fig, "ica-source-separation.png")

    scratch_mean, scratch_dirs, _, _ = scratch_pca(x, 10)
    framework = PCA(n_components=10).fit(x)
    projection_gap = np.linalg.norm(scratch_dirs.T@scratch_dirs - framework.components_.T@framework.components_)
    reconstruction = (x-scratch_mean)@scratch_dirs.T@scratch_dirs + scratch_mean
    mse = np.mean((x-reconstruction)**2)
    print(f"pca_subspace_gap={projection_gap:.8f}")
    print(f"digits_10_component_reconstruction_mse={mse:.5f}")
    print(f"components_for_90_percent={int(np.searchsorted(cumulative,.9)+1)}")
    assert projection_gap < 1e-8 and mse < .045


if __name__ == "__main__":
    main()
