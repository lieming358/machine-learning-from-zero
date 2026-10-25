# 第 18 章参考答案

[返回第 18 章](../18-reinforcement-learning.md) | [English](../../en/solutions/18-solutions.md)

## 1–2. Bellman 与价值迭代

列出每个状态—动作的转移和奖励，从 $V_0=0$ 开始把
$R+\gamma V_k(s')$ 按概率求和，再对动作取最大。expectation equation 对给定 $\pi$ 加权动作；
optimality equation 对动作取最大，前者评估策略、后者寻找最优策略。

## 3. TD、SARSA 与 Q-learning

TD 学 $V$；SARSA target 为 $r+\gamma Q(s',a')$，依赖实际行为策略，是 on-policy；Q-learning target
为 $r+\gamma\max_{a'}Q(s',a')$，是 off-policy。固定环境步数与种子集合，报告回报分布而非最佳曲线。

## 4. 探索

$\epsilon$-greedy 不使用置信度；UCB 加不确定性 bonus；Thompson 从后验采样。比较累计 regret、发现
最佳臂时间和不同 reward gap/horizon，不存在对所有环境固定最优的 $\epsilon$。

## 5. REINFORCE 与 baseline

$\nabla p_\theta(\tau)=p_\theta(\tau)\nabla\log p_\theta(\tau)$ 得到 score-function 梯度。
若 $b(s)$ 与动作无关，$\mathbb E_{a\sim\pi}[\nabla\log\pi(a\mid s)b(s)]=b(s)\nabla\sum_a\pi(a\mid s)=0$，
所以不改期望但可减方差。

## 6–7. PPO 与奖励审计

实现 clipped/unclipped surrogate，联合监控 KL、clip fraction、entropy、value loss、回报和多种子。
奖励方案需列出代理可能被钻的漏洞、硬安全约束、行为回放、独立质量指标和停止/回滚条件；高 reward
不能替代真实目标验收。
