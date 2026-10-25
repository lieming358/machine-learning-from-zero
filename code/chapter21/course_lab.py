"""Chapter 21 lab: turn repeated results into a defensible claim-evidence record."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Experiment:
    name: str
    scores: np.ndarray
    latency_ms: float
    memory_mb: float


def paired_interval(a: np.ndarray, b: np.ndarray, repeats: int = 4000, seed: int = 8):
    difference = a - b
    rng = np.random.default_rng(seed)
    means = np.empty(repeats)
    for repeat in range(repeats):
        index = rng.integers(0, len(difference), len(difference))
        means[repeat] = difference[index].mean()
    return float(difference.mean()), tuple(np.quantile(means, [0.025, 0.975]))


def validate_problem_card(card: dict[str, str]) -> list[str]:
    required = ["decision", "sample", "label_time", "metric", "baseline", "split", "stop_rule"]
    return [key for key in required if not card.get(key, "").strip()]


def validate_claims(claims: list[dict[str, str]]) -> list[str]:
    return [claim["claim"] for claim in claims if not claim.get("evidence") or not claim.get("boundary")]


def main() -> None:
    rng = np.random.default_rng(55)
    seeds = np.arange(10)
    baseline = Experiment("baseline", 0.710 + rng.normal(0, 0.008, len(seeds)), 82, 340)
    proposal = Experiment("proposal", baseline.scores + 0.028 + rng.normal(0, 0.004, len(seeds)), 96, 390)
    gain, interval = paired_interval(proposal.scores, baseline.scores)

    ablation = {
        "full": proposal.scores.mean(),
        "without_reranking": proposal.scores.mean() - 0.017,
        "without_hard_negatives": proposal.scores.mean() - 0.009,
    }
    error_counts = {"not_retrieved": 18, "wrong_evidence": 7, "generation_error": 5}
    problem_card = {
        "decision": "answer a course-policy question with a citation",
        "sample": "one user question",
        "label_time": "annotated before model development",
        "metric": "citation-supported answer accuracy",
        "baseline": "embedding top-5 retrieval",
        "split": "question-template grouped split",
        "stop_rule": "stop if the 95% gain interval crosses zero",
    }
    claims = [
        {
            "claim": "reranking improves the synthetic evaluation score",
            "evidence": f"paired gain {gain:.3f}, 95% interval {interval}",
            "boundary": "ten fixed seeds under the chapter protocol",
        },
        {
            "claim": "the proposal remains inside the latency budget",
            "evidence": f"P95 proxy {proposal.latency_ms} ms",
            "boundary": "chapter workload and machine only",
        },
    ]

    print(f"mean_gain={gain:.4f}, interval={interval}")
    print(f"ablation={ablation}")
    print(f"error_taxonomy={error_counts}")
    print(f"quality_latency_tradeoff={proposal.scores.mean():.3f}@{proposal.latency_ms:.0f}ms")

    assert not validate_problem_card(problem_card)
    assert not validate_claims(claims)
    assert interval[0] > 0.0
    assert ablation["full"] > ablation["without_reranking"]
    assert sum(error_counts.values()) == 30


if __name__ == "__main__":
    main()
