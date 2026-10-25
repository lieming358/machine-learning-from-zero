# Chapter 11 Solutions

[Back to Chapter 11](../11-dimensionality-reduction.md) | [中文](../../zh-CN/solutions/11-solutions.md)

## 1–3. PCA derivation and shapes

Center columns, compute $S=X^\top X/m$, and take the normalized eigenvector of the largest eigenvalue; its sign is
arbitrary. Because projection and residual are orthogonal,
$\|x\|^2=(w^\top x)^2+\|x-ww^\top x\|^2$, proving the objective equivalence.
For thin SVD with $r=\min(m,d)$: $U:m\times r$, $\Sigma:r\times r$, $V:d\times r$, and
$Z=X V_{:,1:k}:m\times k$.

## 4–7. Experiments

Multiplying one feature by 100 multiplies its variance by $10^4$, dominating centered PCA; standardization restores
scale comparability when that assumption is appropriate. Fit PCA inside every training fold. Evaluate dimension by
downstream quality and time, not only variance. ICA degrades with correlated, Gaussian, or noisy sources; match recovered
components up to order/sign/scale. For embedding compression, jointly report index size, P50/P95 latency, Recall@1/10,
and reconstruction error.
