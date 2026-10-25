# Chapter 13 Solutions

[Back to Chapter 13](../13-representation-learning.md) | [中文](../../zh-CN/solutions/13-solutions.md)

## 1–2. InfoNCE

Divide a similarity row by $\tau$, subtract its maximum for stability, exponentiate, and normalize. If the diagonal is
correct, loss is $-\log P_{11}$. Logit gradient is $P_j-1[j=i]$: the positive rises, while negatives fall in proportion
to their probability, so the hardest current negative gets the strongest push.

## 3–5. Temperature, false negatives, and towers

Small temperature sharpens probabilities and gradients; large temperature weakens them. Compare under equal batch,
steps, and tuning budget. Semantic duplicates treated as negatives should be masked or included as multiple positives.
Shared towers suit similar modalities and reduce parameters; separate towers suit asymmetric modalities or statistics.

## 6–7. Retrieval and RAG

Use exact search as reference and sweep ANN search effort, plotting Recall against P95 latency, throughput, and memory.
For RAG, annotate gold evidence and acceptable answers, then separately measure retrieval, reranking, generation given
gold evidence, and end-to-end output. Attribute failures to missing corpus evidence, retrieval miss, ranking, evidence use,
or reasoning.
