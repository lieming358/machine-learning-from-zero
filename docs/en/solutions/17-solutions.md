# Chapter 17 Solutions

[Back to Chapter 17](../17-recommender-systems.md) | [中文](../../zh-CN/solutions/17-solutions.md)

## 1–2. Factorization and BPR

For $e_{ui}=R_{ui}-p_u^Tq_i$, squared-loss gradients are
$-2e_{ui}q_i+2\lambda p_u$ and $-2e_{ui}p_u+2\lambda q_i$. For BPR let $\Delta=s^+-s^-$;
$\partial[-\log\sigma(\Delta)]/\partial\Delta=\sigma(\Delta)-1$, with opposite score gradients.

## 3–4. Sampling and evaluation

Uniform negatives cover the tail but are easy; popularity negatives are harder but can reinforce the head. Compare under
equal positives and budget using full-catalog Recall/NDCG, calibration, tail coverage, and exposure concentration. Split
by event time and build every feature only from past information; sampled-negative evaluation is usually optimistic.

## 5–7. Retrieval and experiments

Use exact top-K as ANN reference and sweep search effort, reporting Recall, P95, throughput, and memory. Sweep MMR
$\lambda$ for relevance versus intra-list diversity. A new-item A/B test should randomize exploration exposure and guard
retention, complaints, latency, and catalog ecology; never label unexposed items negative.
