# 第 15 章参考答案

[返回第 15 章](../15-transformers-llms-and-finetuning.md) | [English](../../en/solutions/15-solutions.md)

## 1. 自回归目标

链式法则给 $p(x_{1:T})=\prod_t p(x_t\mid x_{<t})$。取负对数得到 token 交叉熵之和；teacher
forcing 用真实前缀并行构造所有训练位置，推理时则只能使用已生成前缀。

## 2. 多头与位置

各头把 $d_{model}$ 投影到较小 $d_k,d_v$，独立做 attention，拼接后再投影。若无位置信息，纯
self-attention 对 token 排列近似置换等变，无法区分“狗咬人”和“人咬狗”。

## 3. 参数与 KV 内存

逐层统计 embedding、Q/K/V/O、FFN、Norm。KV 字节数近似
$2L_{layers}Tn_{kv}d_h\times bytes\times concurrency$；GQA/MQA 通过减少 $n_{kv}$ 降内存。

## 4. SFT、偏好与 LoRA

SFT 学示范输出；RLHF/DPO 用偏好区分更好回答。LoRA 令权重增量 $\Delta W=BA$，以低秩参数适配。
公平比较需相同数据、基座、token 预算与评测，并监控遗忘、KL 漂移和安全切片。

## 5–7. 解码、MoE 与评估

温度降低分布熵，top-$k$/top-$p$ 截断候选；任务需确定性时不应只追求多样。MoE 扩参数而每 token
只激活少数专家，但必须用负载均衡和容量约束防热点。LLM 评测同时覆盖事实、引用、安全、长上下文、
TTFT、TPOT、成本和污染，不以 perplexity 单独验收。
