# Chapter 8 — Hints and Answers

[← Chapter 8](../08-trees-and-ensembles.md) | [中文](../../zh-CN/solutions/08-solutions.md)

Weight child impurity by sample counts. A recursive tree needs stopping and empty-child guards. In a
bootstrap ablation, hold depth and feature sampling fixed and report inter-tree error correlation.
Smaller boosting learning rates usually need more trees. A unique ID can earn high impurity importance
by memorization; subject/time splits expose the leakage.
