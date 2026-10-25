# Chapter 6 — Hints and Answers

[← Chapter 6](../06-neighbors-and-kernels.md) | [中文](../../zh-CN/solutions/06-solutions.md)

Sort explicit distances before voting. A unit change that changes neighbors exposes a scale problem.
Noise dimensions make nearest and farthest distances less distinguishable. For RBF,
$\gamma=1/(2\sigma^2)$, so larger $\sigma$ means a wider neighborhood. Compare retrieval metrics on
identical chunks and candidates; normalized dot product and cosine have identical rankings.
