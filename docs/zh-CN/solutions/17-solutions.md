# 第 17 章参考答案

[返回第 17 章](../17-recommender-systems.md) | [English](../../en/solutions/17-solutions.md)

## 1. 矩阵分解更新

对误差 $e_{ui}=R_{ui}-p_u^Tq_i$，平方损失加 L2 的梯度为
$-2e_{ui}q_i+2\lambda p_u$ 与 $-2e_{ui}p_u+2\lambda q_i$。更新第二个向量时应明确使用旧值还是
同步副本，便于复现。

## 2. BPR 梯度

令 $\Delta=s^+-s^-$， $L=-\log\sigma(\Delta)$，则
$\partial L/\partial\Delta=\sigma(\Delta)-1$；故正分数梯度为该值，负分数梯度取反，再经链式法则传到 embedding。

## 3. 负采样

均匀采样覆盖长尾但负例较易；热门采样更难却强化头部。保持正样本、训练步和调参预算一致，比较全库
Recall/NDCG、长尾覆盖、校准和曝光集中度。

## 4. 时间与全库评测

按事件时间切分，所有特征只用截止时刻数据。以完整可推荐库或真实候选为分母计算指标，再和少量随机
负样本结果对照；后者通常显著乐观。

## 5–7. ANN、MMR 与实验

用精确 top-K 作 ANN 真值，扫搜索深度并报告 Recall、P95、吞吐和索引内存。MMR 扫 $\lambda$，
画相关性与 intra-list diversity。新物品 A/B 需随机化探索曝光，主指标含有效互动，护栏含投诉、留存、
延迟和头尾部生态；不能把未曝光当负反馈。
