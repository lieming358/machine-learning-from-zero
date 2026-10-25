"""Chapter 15 lab: shifted targets, causal attention, and a tiny language model.

The trainable model is deliberately a bigram model.  It lets a beginner see the
complete language-model loop without hiding the causal-attention calculation in
a framework.  The attention block is implemented separately with NumPy.
"""

from __future__ import annotations

import numpy as np


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    shifted = x - np.max(x, axis=axis, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=axis, keepdims=True)


def shifted_pairs(sequence: list[int]) -> tuple[np.ndarray, np.ndarray]:
    """Teacher forcing: prefix tokens are inputs and the next tokens are labels."""
    return np.asarray(sequence[:-1]), np.asarray(sequence[1:])


def causal_self_attention(
    hidden: np.ndarray, wq: np.ndarray, wk: np.ndarray, wv: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Single-head causal self-attention for one sequence of shape [T, d]."""
    q, k, v = hidden @ wq, hidden @ wk, hidden @ wv
    logits = q @ k.T / np.sqrt(q.shape[-1])
    future = np.triu(np.ones_like(logits, dtype=bool), k=1)
    logits[future] = -np.inf
    weights = softmax(logits, axis=1)
    return weights @ v, weights


class BigramLanguageModel:
    """A complete, transparent next-token model with trainable logits."""

    def __init__(self, vocabulary_size: int, seed: int = 7):
        rng = np.random.default_rng(seed)
        self.logits = rng.normal(0.0, 0.01, (vocabulary_size, vocabulary_size))

    def loss_and_gradient(self, inputs: np.ndarray, targets: np.ndarray):
        probabilities = softmax(self.logits[inputs], axis=1)
        loss = -np.log(probabilities[np.arange(len(targets)), targets] + 1e-12).mean()
        gradient_rows = probabilities
        gradient_rows[np.arange(len(targets)), targets] -= 1.0
        gradient_rows /= len(targets)
        gradient = np.zeros_like(self.logits)
        np.add.at(gradient, inputs, gradient_rows)
        return float(loss), gradient

    def fit(self, inputs: np.ndarray, targets: np.ndarray, steps: int = 500, lr: float = 1.2):
        history = []
        for step in range(steps):
            loss, gradient = self.loss_and_gradient(inputs, targets)
            self.logits -= lr * gradient
            if step % 50 == 0 or step == steps - 1:
                history.append(loss)
        return history

    def next_probabilities(self, token: int, temperature: float = 1.0) -> np.ndarray:
        return softmax(self.logits[token] / temperature)


def top_p_sample(probabilities: np.ndarray, p: float, rng: np.random.Generator) -> int:
    order = np.argsort(probabilities)[::-1]
    cumulative = np.cumsum(probabilities[order])
    keep_count = int(np.searchsorted(cumulative, p, side="left") + 1)
    kept = order[:keep_count]
    kept_probability = probabilities[kept] / probabilities[kept].sum()
    return int(rng.choice(kept, p=kept_probability))


def main() -> None:
    vocabulary = ["<BOS>", "machine", "learning", "is", "useful", "fun", "<EOS>"]
    token_to_id = {token: index for index, token in enumerate(vocabulary)}
    sentences = [
        ["<BOS>", "machine", "learning", "is", "useful", "<EOS>"],
        ["<BOS>", "machine", "learning", "is", "fun", "<EOS>"],
    ] * 20
    encoded = [[token_to_id[token] for token in sentence] for sentence in sentences]
    pairs = [shifted_pairs(sentence) for sentence in encoded]
    inputs = np.concatenate([pair[0] for pair in pairs])
    targets = np.concatenate([pair[1] for pair in pairs])

    model = BigramLanguageModel(len(vocabulary))
    history = model.fit(inputs, targets)

    rng = np.random.default_rng(11)
    generated = [token_to_id["<BOS>"]]
    for _ in range(8):
        generated.append(top_p_sample(model.next_probabilities(generated[-1], 0.8), 0.9, rng))
        if generated[-1] == token_to_id["<EOS>"]:
            break

    hidden = rng.normal(size=(5, 4))
    wq, wk, wv = (rng.normal(size=(4, 4)) for _ in range(3))
    output, attention = causal_self_attention(hidden, wq, wk, wv)
    future = np.triu(np.ones_like(attention, dtype=bool), k=1)

    dense_parameters = 512 * 512
    rank = 8
    lora_parameters = rank * (512 + 512)

    print(f"shifted_pair={vocabulary[inputs[0]]}->{vocabulary[targets[0]]}")
    print(f"bigram_loss={history[0]:.4f}->{history[-1]:.4f}")
    print("generated=" + " ".join(vocabulary[index] for index in generated))
    print(f"attention_shape={attention.shape}, output_shape={output.shape}")
    print(f"lora_parameter_ratio={lora_parameters / dense_parameters:.4f}")

    assert history[-1] < history[0] * 0.35
    assert np.all(attention[future] == 0.0)
    assert np.allclose(attention.sum(axis=1), 1.0)
    assert output.shape == hidden.shape


if __name__ == "__main__":
    main()
