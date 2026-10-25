"""Chapter 19 lab: decision costs, paired uncertainty, drift, and release gates."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def sigmoid(x: np.ndarray) -> np.ndarray:
    return np.exp(-np.logaddexp(0.0, -x))


def decision_cost(y: np.ndarray, probability: np.ndarray, threshold: float, fp_cost=1.0, fn_cost=8.0):
    prediction = probability >= threshold
    false_positive = np.sum(prediction & (y == 0))
    false_negative = np.sum(~prediction & (y == 1))
    return float(fp_cost * false_positive + fn_cost * false_negative)


def choose_threshold(y: np.ndarray, probability: np.ndarray) -> tuple[float, float]:
    candidates = np.linspace(0.05, 0.95, 91)
    costs = np.array([decision_cost(y, probability, threshold) for threshold in candidates])
    best = int(np.argmin(costs))
    return float(candidates[best]), float(costs[best])


def paired_bootstrap_interval(differences: np.ndarray, seed: int = 5, repeats: int = 2000):
    rng = np.random.default_rng(seed)
    means = np.empty(repeats)
    for repeat in range(repeats):
        index = rng.integers(0, len(differences), len(differences))
        means[repeat] = differences[index].mean()
    return tuple(np.quantile(means, [0.025, 0.975]))


def population_stability_index(reference: np.ndarray, current: np.ndarray, bins: int = 10) -> float:
    boundaries = np.quantile(reference, np.linspace(0, 1, bins + 1))
    boundaries[0], boundaries[-1] = -np.inf, np.inf
    reference_count = np.histogram(reference, boundaries)[0] / len(reference)
    current_count = np.histogram(current, boundaries)[0] / len(current)
    reference_count = np.clip(reference_count, 1e-6, None)
    current_count = np.clip(current_count, 1e-6, None)
    return float(np.sum((current_count - reference_count) * np.log(current_count / reference_count)))


@dataclass
class ReleaseGate:
    minimum_quality: float
    maximum_p95_ms: float
    maximum_error_rate: float

    def decide(self, quality: float, p95_ms: float, error_rate: float) -> str:
        if quality < self.minimum_quality:
            return "rollback: quality"
        if p95_ms > self.maximum_p95_ms:
            return "rollback: latency"
        if error_rate > self.maximum_error_rate:
            return "rollback: errors"
        return "promote"


def main() -> None:
    rng = np.random.default_rng(31)
    latent = rng.normal(size=5000)
    true_probability = sigmoid(latent - 1.0)
    y = (rng.random(len(latent)) < true_probability).astype(int)
    model_a = np.clip(true_probability + rng.normal(0, 0.025, len(y)), 0.001, 0.999)
    model_b = np.clip(sigmoid(0.65 * latent - 0.7), 0.001, 0.999)

    validation = np.arange(0, 2500)
    test = np.arange(2500, 5000)
    threshold, validation_cost = choose_threshold(y[validation], model_a[validation])
    default_test_cost = decision_cost(y[test], model_a[test], 0.5)
    selected_test_cost = decision_cost(y[test], model_a[test], threshold)

    loss_a = (model_a[test] - y[test]) ** 2
    loss_b = (model_b[test] - y[test]) ** 2
    interval = paired_bootstrap_interval(loss_a - loss_b)

    reference = rng.normal(0, 1, 3000)
    current = rng.normal(0.55, 1.15, 3000)
    psi = population_stability_index(reference, current)
    gate = ReleaseGate(minimum_quality=0.82, maximum_p95_ms=180, maximum_error_rate=0.01)

    print(f"selected_threshold={threshold:.2f}, validation_cost={validation_cost:.0f}")
    print(f"test_cost_default={default_test_cost:.0f}, selected={selected_test_cost:.0f}")
    print(f"paired_brier_difference_ci={interval}")
    print(f"psi={psi:.3f}")
    print("release_good=" + gate.decide(0.85, 145, 0.004))
    print("release_slow=" + gate.decide(0.85, 230, 0.004))

    assert selected_test_cost < default_test_cost
    assert interval[1] < 0.0
    assert psi > 0.1
    assert gate.decide(0.85, 145, 0.004) == "promote"
    assert gate.decide(0.85, 230, 0.004).startswith("rollback")


if __name__ == "__main__":
    main()
