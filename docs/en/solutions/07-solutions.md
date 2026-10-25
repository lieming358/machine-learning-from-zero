# Chapter 7 — Hints and Answers

[← Chapter 7](../07-support-vector-machines.md) | [中文](../../zh-CN/solutions/07-solutions.md)

Point-to-plane distance is $|w^Tx+b|/\|w\|$. Normalize the closest functional margin to one.
Tune $C$ and $\gamma$ inside leakage-safe folds. Removing zero-dual-weight points usually preserves the
solution; removing support vectors may not. Calibrate decision scores on separate validation data and
evaluate Brier score, log loss, and reliability—not accuracy alone.
