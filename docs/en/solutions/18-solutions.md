# Chapter 18 Solutions

[Back to Chapter 18](../18-reinforcement-learning.md) | [中文](../../zh-CN/solutions/18-solutions.md)

## 1–3. Bellman and TD control

Start at $V_0=0$, compute expected $R+\gamma V_k(s')$ for each action, then maximize. The expectation equation averages
actions under a fixed policy; optimality maximizes to seek a policy. SARSA targets $r+\gamma Q(s',a')$ from the actual
behavior action and is on-policy. Q-learning targets $r+\gamma\max_{a'}Q(s',a')$ and is off-policy. Compare return
distributions under common environment steps and seeds.

## 4. Exploration

$\epsilon$-greedy ignores confidence, UCB adds an uncertainty bonus, and Thompson sampling draws from a posterior.
Compare cumulative regret and time to discover the best arm across horizons and reward gaps; no fixed $\epsilon$ wins
universally.

## 5. REINFORCE baseline

Use $\nabla p_\theta(\tau)=p_\theta(\tau)\nabla\log p_\theta(\tau)$. For action-independent $b(s)$,
$\mathbb E_a[\nabla\log\pi(a\mid s)b(s)]=b(s)\nabla\sum_a\pi(a\mid s)=0$, preserving expectation while reducing variance.

## 6–7. PPO and rewards

Plot clipped and unclipped surrogates and track KL, clip fraction, entropy, value loss, returns, and seeds. Enumerate how
an agent could exploit each reward proxy; add hard constraints, trajectory review, independent outcomes, and rollback.
High return is not a substitute for intended-behavior evaluation.
