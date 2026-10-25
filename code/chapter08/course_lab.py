"""Chapter 8 lab: impurity, bagging votes, and one boosting correction."""

from __future__ import annotations

import numpy as np


def gini(labels: np.ndarray) -> float:
    if len(labels) == 0:
        return 0.0
    probability = np.bincount(labels, minlength=2) / len(labels)
    return float(1.0 - probability @ probability)


def weighted_split_gini(
    labels: np.ndarray,
    left_mask: np.ndarray,
) -> float:
    left = labels[left_mask]
    right = labels[~left_mask]
    return (len(left) * gini(left) + len(right) * gini(right)) / len(labels)


def bagging_vote(tree_predictions: np.ndarray) -> np.ndarray:
    # Rows are trees and columns are queries.
    positive_votes = tree_predictions.mean(axis=0)
    return (positive_votes >= 0.5).astype(int)


def draw_bootstrap_indices(
    number_of_rows: int,
    number_of_trees: int,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, number_of_rows, size=(number_of_trees, number_of_rows))


def one_boosting_round(
    x: np.ndarray,
    y: np.ndarray,
    split: float,
    learning_rate: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    initial_prediction = np.full_like(y, y.mean(), dtype=float)
    residual = y - initial_prediction
    left = x <= split
    stump_prediction = np.where(left, residual[left].mean(), residual[~left].mean())
    updated_prediction = initial_prediction + learning_rate * stump_prediction
    return residual, stump_prediction, updated_prediction


def main() -> None:
    labels = np.array([0, 0, 1, 1, 1, 0])
    feature = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    split_score = weighted_split_gini(labels, feature <= 2.5)
    print(f"root_gini={gini(labels):.3f}, split_gini={split_score:.3f}")

    bootstrap_indices = draw_bootstrap_indices(number_of_rows=6, number_of_trees=3, seed=4)
    print(f"bootstrap_indices={bootstrap_indices.tolist()}")
    tree_predictions = np.array(
        [
            [0, 1, 1, 0],
            [0, 1, 0, 0],
            [1, 1, 1, 0],
        ]
    )
    forest_prediction = bagging_vote(tree_predictions)
    print(f"tree_votes={tree_predictions.sum(axis=0).tolist()}")
    print(f"forest_prediction={forest_prediction.tolist()}")

    regression_x = np.array([0.0, 1.0, 2.0, 3.0])
    regression_y = np.array([2.0, 2.0, 8.0, 10.0])
    residual, correction, updated = one_boosting_round(
        regression_x,
        regression_y,
        split=1.5,
        learning_rate=0.5,
    )
    initial = np.full_like(regression_y, regression_y.mean())
    mse_before = np.mean((regression_y - initial) ** 2)
    mse_after = np.mean((regression_y - updated) ** 2)
    print(f"residual={residual.tolist()}")
    print(f"stump_correction={correction.tolist()}")
    print(f"boosting_mse={mse_before:.3f}->{mse_after:.3f}")

    assert split_score < gini(labels)
    assert bootstrap_indices.shape == (3, 6)
    assert len(np.unique(bootstrap_indices[0])) < 6
    assert np.array_equal(forest_prediction, np.array([0, 1, 1, 0]))
    assert mse_after < mse_before


if __name__ == "__main__":
    main()
