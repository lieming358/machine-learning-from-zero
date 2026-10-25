"""Chapter 15: causal attention, Transformer cost, and mixture-of-experts routing."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "docs" / "assets"
BLUE, ORANGE, GREEN, RED, NAVY = "#3478c0", "#f28e2b", "#3a9d5d", "#d94b4b", "#19324d"


def softmax(x, axis=-1):
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


def finish(fig, name):
    fig.tight_layout(); fig.savefig(OUTPUT / name, dpi=180, bbox_inches="tight", facecolor="white"); plt.close(fig)


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(9)
    n, d = 10, 8
    x = rng.normal(size=(n, d)); q, k, v = x @ rng.normal(size=(d, d)), x @ rng.normal(size=(d, d)), x @ rng.normal(size=(d, d))
    logits = q @ k.T / np.sqrt(d)
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    logits[mask] = -np.inf
    attention = softmax(logits)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.3))
    axes[0].imshow(mask, cmap="Greys", vmin=0, vmax=1); axes[0].set(title="Causal mask", xlabel="key position", ylabel="query position")
    image = axes[1].imshow(attention, cmap="magma", vmin=0, vmax=attention.max()); axes[1].set(title="Masked attention weights", xlabel="key position", ylabel="query position")
    fig.colorbar(image, ax=axes[1], label="weight")
    finish(fig, "transformer-causal-mask.png")

    fig, ax = plt.subplots(figsize=(10, 3.4)); ax.axis("off")
    boxes = [("tokens", .03, BLUE), ("embedding + position", .20, ORANGE), ("masked attention", .43, GREEN),
             ("residual + norm", .63, NAVY), ("MLP", .80, RED)]
    for label, x0, color in boxes:
        ax.add_patch(plt.Rectangle((x0, .34), .15, .32, facecolor=color, alpha=.85, edgecolor="white"))
        ax.text(x0+.075, .5, label, ha="center", va="center", color="white", fontsize=10)
        if x0 < .80: ax.annotate("", xy=(x0+.18,.5), xytext=(x0+.15,.5), arrowprops=dict(arrowstyle="->", color=NAVY, lw=1.8))
    ax.set(title="A decoder block alternates communication and per-token computation")
    finish(fig, "transformer-block-flow.png")

    lengths = np.array([128, 256, 512, 1024, 2048, 4096])
    score_memory = lengths[:, None] ** 2 * np.array([8, 16, 32])[None, :] * 2 / 1024**2
    fig, ax = plt.subplots(figsize=(8.4, 4.5))
    for j, heads in enumerate([8,16,32]): ax.plot(lengths, score_memory[:,j], "o-", lw=2.1, label=f"{heads} heads")
    ax.set(title="Attention-score memory grows quadratically with context", xlabel="tokens", ylabel="score memory proxy (MiB, fp16)")
    ax.grid(alpha=.15); ax.legend(frameon=False)
    finish(fig, "transformer-context-cost.png")

    tokens = 8000; experts = 8
    balanced = rng.multinomial(tokens, np.ones(experts)/experts)
    biased_probability = np.array([.38,.20,.12,.09,.07,.06,.05,.03]); biased = rng.multinomial(tokens, biased_probability)
    fig, axes = plt.subplots(1,2,figsize=(10,4),sharey=True)
    for ax, counts, title in zip(axes,[balanced,biased],["Balanced routing","Collapsed routing"]):
        ax.bar(np.arange(experts),counts,color=BLUE if title.startswith("Balanced") else RED); ax.axhline(tokens/experts,color=NAVY,ls="--")
        ax.set(title=title,xlabel="expert",ylabel="assigned tokens"); ax.grid(axis="y",alpha=.15)
    finish(fig,"moe-routing-balance.png")

    assert np.all(attention[mask] == 0)
    assert np.allclose(attention.sum(axis=1), 1)
    print(f"max_future_attention={attention[mask].max():.2e}")
    print(f"biased_load_ratio={biased.max()/biased.mean():.2f}")


if __name__ == "__main__":
    main()
