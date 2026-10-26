# Chapter 19 Solutions

[Back to Chapter 19](../19-evaluation-causality-and-mlops.md) | [中文](../../zh-CN/solutions/19-solutions.md)

## 1. Same AUC, different calibration

Apply a strictly monotone transform such as $p'=\sigma(c\mathrm{logit}p)$. Ranking and AUC remain, while $c>1$
creates overconfidence and changes Brier score. Fit calibration separately from final reliability evaluation.

## 2. Cluster bootstrap

Sample patients with replacement and include every record belonging to a sampled patient, then recompute the metric or
model difference. Row-wise bootstrap falsely treats within-patient records as independent and often narrows intervals.

## 3–5. Causality, leakage, and drift

Draw pre-treatment interest affecting both exposure and click and adjust only a valid sufficient set—not mediators or
colliders. Record event, processing, and availability time per feature and replay historical joins. Combine drift tests
with feature relevance and quality slices. A large shift in an unused noise feature demonstrates that drift need not harm
quality.

## 6–7. Release and generative evaluation

A model card documents purpose, data, slices, and limits; a release gate covers quality, calibration, safety, P99, and
cost; canary policy specifies traffic steps, alert owner, and rollback. Layer RAG/agent evaluation into retrieval,
citation, generation, tool trajectory, and safety, using private rolling sets and human calibration of model judges.
