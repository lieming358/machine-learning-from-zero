"""Chapter 14 compact preview; use expanded_models.py for the full course lab."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "docs" / "assets"
BLUE, ORANGE, GREEN, RED, NAVY = "#3478c0", "#f28e2b", "#3a9d5d", "#d94b4b", "#19324d"


def softmax(x, axis=-1):
    shifted = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(shifted)
    return e / e.sum(axis=axis, keepdims=True)


def finish(fig, name):
    fig.tight_layout()
    fig.savefig(OUTPUT / name, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)

    signal = np.array([0, 0, 1, 1, 1, 0, 0, -1, -1, 0, 0], dtype=float)
    kernels = {"smooth": np.ones(3) / 3, "edge": np.array([-1, 0, 1.0])}
    fig, axes = plt.subplots(2, 1, figsize=(9, 5.8), sharex=True)
    axes[0].stem(np.arange(len(signal)), signal, basefmt=" ", linefmt=BLUE, markerfmt="o")
    axes[0].set(title="One signal contains levels and transitions", ylabel="input")
    for label, kernel, color in [("smoothing kernel", kernels["smooth"], GREEN), ("edge kernel", kernels["edge"], RED)]:
        out = np.convolve(signal, kernel[::-1], mode="valid")
        axes[1].plot(np.arange(1, len(signal)-1), out, "o-", color=color, label=label)
    axes[1].set(title="The same local operation slides across every position", xlabel="position", ylabel="convolution")
    axes[1].legend(frameon=False); [ax.grid(alpha=.15) for ax in axes]
    finish(fig, "cnn-shared-kernel.png")

    steps = np.arange(0, 61)
    fig, ax = plt.subplots(figsize=(8.3, 4.3))
    for factor, color in [(.75, RED), (.95, ORANGE), (1.0, GREEN), (1.05, BLUE)]:
        ax.semilogy(steps, np.maximum(factor ** steps, 1e-4), color=color, lw=2.2, label=f"recurrent factor={factor}")
    ax.set(title="Repeated Jacobians can erase or amplify long-range influence", xlabel="time steps", ylabel="relative gradient magnitude")
    ax.grid(alpha=.15); ax.legend(frameon=False)
    finish(fig, "rnn-gradient-memory.png")

    tokens = ["the", "animal", "did", "not", "cross", "because", "it", "was", "tired"]
    rng = np.random.default_rng(4)
    x = rng.normal(size=(len(tokens), 6))
    x[1] = np.array([2, 0, 1, 0, 0, 0]); x[6] = np.array([1.9, .1, 1.1, 0, 0, 0])
    q, k, v = x, x, rng.normal(size=(len(tokens), 5))
    scores = q @ k.T / np.sqrt(k.shape[1])
    weights = softmax(scores, axis=1)
    context = weights @ v
    fig, ax = plt.subplots(figsize=(8.2, 6.3))
    image = ax.imshow(weights, cmap="Blues", vmin=0, vmax=weights.max())
    ax.set(xticks=np.arange(len(tokens)), yticks=np.arange(len(tokens)), xticklabels=tokens,
           yticklabels=tokens, xlabel="key token", ylabel="query token", title="Self-attention builds a data-dependent weighted context")
    plt.setp(ax.get_xticklabels(), rotation=40, ha="right")
    fig.colorbar(image, ax=ax, label="attention weight")
    finish(fig, "self-attention-weights.png")

    lengths = np.arange(16, 513, 16)
    fig, ax = plt.subplots(figsize=(8.2, 4.3))
    ax.plot(lengths, lengths, color=GREEN, lw=2.3, label="recurrent states: O(n)")
    ax.plot(lengths, lengths ** 2 / 64, color=RED, lw=2.3, label="full attention scores: O(n²)")
    ax.set(title="Sequence length changes the dominant memory pattern", xlabel="sequence length n", ylabel="relative stored interactions")
    ax.grid(alpha=.15); ax.legend(frameon=False)
    finish(fig, "sequence-complexity.png")

    assert np.allclose(weights.sum(axis=1), 1.0)
    assert context.shape == (len(tokens), 5)
    print(f"attention_row_sum_error={np.max(np.abs(weights.sum(axis=1)-1)):.2e}")
    print(f"context_shape={context.shape}")


if __name__ == "__main__":
    main()
