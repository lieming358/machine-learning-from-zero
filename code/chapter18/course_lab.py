"""Chapter 18 lab: compare value iteration, SARSA, and Q-learning."""

from __future__ import annotations

import numpy as np


ACTIONS = np.array([[-1, 0], [1, 0], [0, -1], [0, 1]])


class GridWorld:
    def __init__(self, rows: int = 4, columns: int = 5):
        self.rows, self.columns = rows, columns
        self.start = (rows - 1, 0)
        self.goal = (0, columns - 1)
        self.trap = (1, 3)

    @property
    def state_count(self) -> int:
        return self.rows * self.columns

    def encode(self, position: tuple[int, int]) -> int:
        return position[0] * self.columns + position[1]

    def decode(self, state: int) -> tuple[int, int]:
        return divmod(state, self.columns)

    def step(self, state: int, action: int):
        row, column = self.decode(state)
        next_row = int(np.clip(row + ACTIONS[action, 0], 0, self.rows - 1))
        next_column = int(np.clip(column + ACTIONS[action, 1], 0, self.columns - 1))
        next_position = (next_row, next_column)
        terminated = next_position in (self.goal, self.trap)
        reward = 8.0 if next_position == self.goal else (-8.0 if next_position == self.trap else -0.2)
        return self.encode(next_position), reward, terminated


def epsilon_greedy(q: np.ndarray, state: int, epsilon: float, rng: np.random.Generator) -> int:
    if rng.random() < epsilon:
        return int(rng.integers(q.shape[1]))
    best = np.flatnonzero(q[state] == q[state].max())
    return int(rng.choice(best))


def train_control(method: str, episodes: int = 1800, seed: int = 0):
    environment = GridWorld()
    rng = np.random.default_rng(seed)
    q = np.zeros((environment.state_count, len(ACTIONS)))
    alpha, gamma, epsilon = 0.18, 0.95, 0.18
    returns = []
    for _ in range(episodes):
        state = environment.encode(environment.start)
        action = epsilon_greedy(q, state, epsilon, rng)
        total = 0.0
        for _ in range(80):
            next_state, reward, terminated = environment.step(state, action)
            total += reward
            next_action = epsilon_greedy(q, next_state, epsilon, rng)
            if terminated:
                target = reward
            elif method == "sarsa":
                target = reward + gamma * q[next_state, next_action]
            else:
                target = reward + gamma * np.max(q[next_state])
            q[state, action] += alpha * (target - q[state, action])
            if terminated:
                break
            state, action = next_state, next_action
        returns.append(total)
        epsilon = max(0.03, epsilon * 0.998)
    return q, np.asarray(returns)


def evaluate(q: np.ndarray, episodes: int = 100) -> float:
    environment = GridWorld()
    successes = 0
    for _ in range(episodes):
        state = environment.encode(environment.start)
        for _ in range(40):
            action = int(np.argmax(q[state]))
            state, reward, terminated = environment.step(state, action)
            if terminated:
                successes += int(reward > 0)
                break
    return successes / episodes


def trajectory_targets(rewards: list[float], values: list[float], gamma: float = 0.9):
    monte_carlo = []
    running = 0.0
    for reward in reversed(rewards):
        running = reward + gamma * running
        monte_carlo.append(running)
    monte_carlo.reverse()
    td = [rewards[t] + gamma * values[t + 1] for t in range(len(rewards))]
    return np.asarray(monte_carlo), np.asarray(td)


def ppo_clipped_term(ratio: np.ndarray, advantage: np.ndarray, epsilon: float = 0.2):
    unclipped = ratio * advantage
    clipped = np.clip(ratio, 1 - epsilon, 1 + epsilon) * advantage
    return np.minimum(unclipped, clipped)


def main() -> None:
    sarsa_q, sarsa_return = train_control("sarsa", seed=4)
    q_learning_q, q_return = train_control("q_learning", seed=4)
    sarsa_success = evaluate(sarsa_q)
    q_success = evaluate(q_learning_q)

    mc_target, td_target = trajectory_targets(
        rewards=[0.0, 0.0, 1.0], values=[0.2, 0.4, 0.7, 0.0]
    )
    ratio = np.array([0.7, 1.0, 1.3])
    clipped = ppo_clipped_term(ratio, np.ones_like(ratio))

    print(f"mc_targets={np.round(mc_target, 3)}")
    print(f"td_targets={np.round(td_target, 3)}")
    print(f"sarsa_success={sarsa_success:.2f}, q_learning_success={q_success:.2f}")
    print(f"late_return_sarsa={sarsa_return[-100:].mean():.3f}")
    print(f"late_return_q_learning={q_return[-100:].mean():.3f}")
    print(f"ppo_positive_advantage_terms={clipped}")

    assert sarsa_success > 0.9 and q_success > 0.9
    assert sarsa_return[-100:].mean() > sarsa_return[:100].mean()
    assert q_return[-100:].mean() > q_return[:100].mean()
    assert np.isclose(clipped[-1], 1.2)


if __name__ == "__main__":
    main()
