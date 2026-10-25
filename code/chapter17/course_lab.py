"""Chapter 17 lab: neighbor CF, BPR matrix factorization, and ranking metrics."""

from __future__ import annotations

import numpy as np


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    observed = ~np.isnan(a) & ~np.isnan(b)
    if observed.sum() < 2:
        return 0.0
    a_centered = a[observed] - np.mean(a[observed])
    b_centered = b[observed] - np.mean(b[observed])
    denominator = np.linalg.norm(a_centered) * np.linalg.norm(b_centered)
    return 0.0 if denominator == 0 else float(a_centered @ b_centered / denominator)


def user_based_prediction(ratings: np.ndarray, user: int, item: int) -> float:
    """Mean-centered prediction from users who rated the target item."""
    user_mean = np.nanmean(ratings[user])
    numerator = denominator = 0.0
    for neighbor in range(len(ratings)):
        if neighbor == user or np.isnan(ratings[neighbor, item]):
            continue
        similarity = cosine_similarity(ratings[user], ratings[neighbor])
        numerator += similarity * (ratings[neighbor, item] - np.nanmean(ratings[neighbor]))
        denominator += abs(similarity)
    return float(user_mean if denominator == 0 else user_mean + numerator / denominator)


def sigmoid(x: float) -> float:
    return float(np.exp(-np.logaddexp(0.0, -x)))


def bpr_loss(users: np.ndarray, items: np.ndarray, triples: np.ndarray) -> float:
    positive = np.sum(users[triples[:, 0]] * items[triples[:, 1]], axis=1)
    negative = np.sum(users[triples[:, 0]] * items[triples[:, 2]], axis=1)
    return float(np.mean(np.logaddexp(0.0, -(positive - negative))))


def train_bpr(train_positive: list[set[int]], item_count: int, rank: int = 6, steps: int = 18000):
    rng = np.random.default_rng(13)
    user_vectors = rng.normal(0, 0.1, (len(train_positive), rank))
    item_vectors = rng.normal(0, 0.1, (item_count, rank))
    triples = []
    for user, positives in enumerate(train_positive):
        for positive in sorted(positives):
            negative = next(item for item in range(item_count) if item not in positives)
            triples.append((user, positive, negative))
    fixed_triples = np.asarray(triples, dtype=int)
    initial_loss = bpr_loss(user_vectors, item_vectors, fixed_triples)

    learning_rate, regularization = 0.035, 0.002
    for _ in range(steps):
        user = int(rng.integers(len(train_positive)))
        positive = int(rng.choice(sorted(train_positive[user])))
        while True:
            negative = int(rng.integers(item_count))
            if negative not in train_positive[user]:
                break
        old_user = user_vectors[user].copy()
        old_positive = item_vectors[positive].copy()
        old_negative = item_vectors[negative].copy()
        difference = old_user @ (old_positive - old_negative)
        strength = sigmoid(-difference)
        user_vectors[user] += learning_rate * (
            strength * (old_positive - old_negative) - regularization * old_user
        )
        item_vectors[positive] += learning_rate * (
            strength * old_user - regularization * old_positive
        )
        item_vectors[negative] += learning_rate * (
            -strength * old_user - regularization * old_negative
        )
    return user_vectors, item_vectors, initial_loss, bpr_loss(user_vectors, item_vectors, fixed_triples)


def ranking_metrics(scores: np.ndarray, train_positive: list[set[int]], held_out: np.ndarray, k: int = 5):
    hits, ndcg = [], []
    for user in range(len(scores)):
        candidate_scores = scores[user].copy()
        candidate_scores[list(train_positive[user])] = -np.inf
        top = np.argsort(candidate_scores)[::-1][:k]
        match = np.flatnonzero(top == held_out[user])
        hits.append(float(len(match) > 0))
        ndcg.append(0.0 if len(match) == 0 else 1.0 / np.log2(match[0] + 2.0))
    return float(np.mean(hits)), float(np.mean(ndcg))


def main() -> None:
    ratings = np.array(
        [[5, 4, np.nan, 1], [4, 5, 2, 1], [1, 1, 5, 4], [2, 1, 4, 5]],
        dtype=float,
    )
    neighbor_prediction = user_based_prediction(ratings, user=0, item=2)

    rng = np.random.default_rng(21)
    user_count, item_count, rank = 30, 50, 5
    true_users = rng.normal(size=(user_count, rank))
    true_items = rng.normal(size=(item_count, rank))
    preference = true_users @ true_items.T
    best = np.argsort(preference, axis=1)[:, -7:]
    held_out = best[:, -1]
    train_positive = [set(row[:-1]) for row in best]

    users, items, initial_loss, final_loss = train_bpr(train_positive, item_count)
    recall, ndcg = ranking_metrics(users @ items.T, train_positive, held_out)
    random_recall = 5 / (item_count - len(train_positive[0]))

    print(f"neighbor_rating_prediction={neighbor_prediction:.3f}")
    print(f"bpr_loss={initial_loss:.4f}->{final_loss:.4f}")
    print(f"recall@5={recall:.3f}, ndcg@5={ndcg:.3f}, random_recall={random_recall:.3f}")
    print(f"user_tower_shape={users.shape}, item_tower_shape={items.shape}")

    assert 1.0 <= neighbor_prediction <= 5.0
    assert final_loss < initial_loss * 0.7
    assert recall > random_recall
    assert 0.0 <= ndcg <= recall <= 1.0


if __name__ == "__main__":
    main()
