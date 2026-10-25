# 第 20 章参考答案

[返回第 20 章](../20-machine-learning-systems.md) | [English](../../en/solutions/20-solutions.md)

## 1. Amdahl 上限

把 profile 中目标阶段占比记为 $p$，局部加速为 $s$，代入
$S=1/[(1-p)+p/s]$。即使 $s\to\infty$，总加速也不超过 $1/(1-p)$；因此先优化占比最大的可改善阶段。

## 2. Roofline

统计算子 FLOPs 与从主存/显存搬运字节，算 $I=FLOPs/bytes$；与机器平衡点
$P_{peak}/B_{memory}$ 比较。 $I$ 较小更可能带宽受限，但最终仍需 profiler 验证缓存和融合。

## 3. 服务压测

预热后按真实长度分布扫 batch、并发和最大等待，报告吞吐、P50/P95/P99、峰值显存、错误率与功耗。
只在共同质量门槛内连接 Pareto 前沿，不能用 OOM 或超时点冒充高吞吐。

## 4. 量化

使用代表性校准集分别做 per-tensor/per-channel int8/int4，记录层级误差、下游质量、真实 kernel 延迟、
吞吐和内存。若硬件回退到 fp16，文件变小并不代表服务加速。

## 5–6. 内存与并行

参数字节为参数量乘精度；Adam 还含梯度、主权重和两组状态；激活与 batch/序列有关；KV 见第 15 章
公式。8 卡方案应把每层 all-reduce/all-gather/send-recv 字节列出，再用互联带宽估算通信并报告 scaling efficiency。

## 7. 容量与回退

由峰值 QPS、长度和服务时间估卡数并留余量；设置限流、优先级、超时、continuous batching、缓存、
小模型/短上下文降级、幂等重试和上一版本回滚。成本按成功任务统计，并定期做过载演练。
