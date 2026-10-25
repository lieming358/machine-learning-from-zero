# Chapter 15 Solutions

[Back to Chapter 15](../15-transformers-llms-and-finetuning.md) | [中文](../../zh-CN/solutions/15-solutions.md)

## 1–2. Objective and architecture

The chain rule gives $p(x_{1:T})=\prod_t p(x_t\mid x_{<t})$; negative log-likelihood is summed token cross-entropy.
Teacher forcing creates all shifted training targets in parallel, while generation only has its sampled prefix. Multi-head
attention projects into several subspaces and concatenates outputs. Without positional information, self-attention cannot
distinguish reordered token roles.

## 3–4. Memory and adaptation

Count embeddings, Q/K/V/O, FFN, and normalization per layer. KV bytes are approximately
$2L_{layers}Tn_{kv}d_h\times bytes\times concurrency$; GQA/MQA reduce $n_{kv}$. SFT learns demonstrations; preference
optimization separates chosen from rejected behavior. LoRA uses $\Delta W=BA$ to adapt with low-rank parameters.

## 5–7. Decoding, MoE, and evaluation

Temperature changes entropy and top-$k$/top-$p$ truncate support. MoE expands parameters while activating few experts,
requiring load balancing and capacity controls. Evaluate factuality, citation, safety, long context, TTFT, TPOT, cost,
and contamination; perplexity alone cannot approve a model.
