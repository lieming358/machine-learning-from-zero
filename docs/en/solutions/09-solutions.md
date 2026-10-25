# Chapter 9 Solutions

[Back to Chapter 9](../09-learning-theory-and-model-selection.md) | [中文](../../zh-CN/solutions/09-solutions.md)

## 1. Fold losses

For $[0.42,0.38,0.51,0.40,0.44]$, the mean is $0.43$ and sample standard deviation is about $0.050$.
This describes variation among correlated folds, not a strict confidence interval from independent experiments.

## 2. Leakage audit

Isolate a grouped/temporal test set first. Within training data, fit scaling separately in each CV training fold and
select by mean validation performance. Refit after selection and evaluate the test set once. Do not tune after viewing it.

## 3. Learning curves

High bias gives high, close train and validation errors: improve features/capacity or weaken regularization. High
variance gives low training error and a large validation gap: more data, stronger regularization, or less capacity may help.

## 4. Finite-class bound

Hoeffding gives $2e^{-2m\epsilon^2}$ for one fixed hypothesis. Union bound over $|\mathcal H|$ events gives
$2|\mathcal H|e^{-2m\epsilon^2}$. Set this to $\delta$ and solve for
$\epsilon=\sqrt{\log(2|\mathcal H|/\delta)/(2m)}$.

## 5–7. Experiments and design

Noise raises attainable error and selection variance; more data usually stabilizes selection; more candidates increase
validation selection bias. Group patients by patient ID, choose grouped or temporal user splits according to deployment,
and roll sales validation forward in time. For RAG, separately score retrieval Recall@$k$, reranking, grounded generation,
and end-to-end answers; label the stage where each failure began.
