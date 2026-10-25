"""Chapter 6 lab: trace one kNN prediction and one RBF-kernel prediction."""

from __future__ import annotations

import numpy as np


def squared_distance(query: np.ndarray, training: np.ndarray) -> np.ndarray:
    difference = training - query
    return np.sum(difference * difference, axis=1)


def knn_regression(
    query: np.ndarray,
    training_x: np.ndarray,
    training_y: np.ndarray,
    k: int,
) -> tuple[float, np.ndarray]:
    distance = squared_distance(query, training_x)
    neighbor_index = np.argsort(distance)[:k]
    prediction = float(np.mean(training_y[neighbor_index]))
    return prediction, neighbor_index


def rbf_kernel(distance_squared: np.ndarray, gamma: float) -> np.ndarray:
    return np.exp(-gamma * distance_squared)


def kernel_regression(
    query: np.ndarray,
    training_x: np.ndarray,
    training_y: np.ndarray,
    gamma: float,
) -> tuple[float, np.ndarray]:
    distance = squared_distance(query, training_x)
    raw_weight = rbf_kernel(distance, gamma)
    normalized_weight = raw_weight / raw_weight.sum()
    prediction = float(normalized_weight @ training_y)
    return prediction, normalized_weight


def select_gamma(
    training_x: np.ndarray,
    training_y: np.ndarray,
    validation_x: np.ndarray,
    validation_y: np.ndarray,
    candidates: list[float],
) -> tuple[float, dict[float, float]]:
    losses: dict[float, float] = {}
    for gamma in candidates:
        predictions = [
            kernel_regression(query, training_x, training_y, gamma)[0]
            for query in validation_x
        ]
        losses[gamma] = float(np.mean((np.asarray(predictions) - validation_y) ** 2))
    best_gamma = min(losses, key=losses.get)
    return best_gamma, losses


def main() -> None:
    # Each row is [area, distance_to_subway]; y is monthly rent in hundred yuan.
    training_x = np.array([[30.0, 1.0], [45.0, 0.8], [60.0, 1.2], [80.0, 3.0]])
    training_y = np.array([20.0, 26.0, 31.0, 35.0])
    query = np.array([50.0, 1.0])

    # Scale with training-set constants so area does not dominate the distance.
    scale = np.array([20.0, 1.0])
    scaled_training = training_x / scale
    scaled_query = query / scale

    knn_prediction, neighbor_index = knn_regression(
        scaled_query,
        scaled_training,
        training_y,
        k=2,
    )
    kernel_prediction, weight = kernel_regression(
        scaled_query,
        scaled_training,
        training_y,
        gamma=1.0,
    )
    print(f"knn_neighbors={neighbor_index.tolist()}, prediction={knn_prediction:.2f}")
    print(f"kernel_weights={np.round(weight, 3)}, prediction={kernel_prediction:.2f}")

    validation_x = np.array([[37.0, 0.9], [70.0, 2.0]]) / scale
    validation_y = np.array([23.0, 34.0])
    best_gamma, losses = select_gamma(
        scaled_training,
        training_y,
        validation_x,
        validation_y,
        candidates=[0.1, 1.0, 10.0],
    )
    rounded_losses = {gamma: round(loss, 3) for gamma, loss in losses.items()}
    print(f"validation_mse={rounded_losses}, selected_gamma={best_gamma}")

    assert np.isclose(weight.sum(), 1.0)
    assert 20.0 < kernel_prediction < 35.0
    assert best_gamma in losses


if __name__ == "__main__":
    main()
