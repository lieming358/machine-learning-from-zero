# Chapters 0–3: Hints and Solution Guide

[← Home](../../../README.en.md) | [中文](../../zh-CN/solutions/01-03-solutions.md)

Use this only after attempting the exercises. Strong answers define assumptions, control comparisons,
report variation, and distinguish observations from explanations.

Key checks:

- A meal-demand task uses historical context as experience and should report waste/stockout costs, not
  error alone.
- Split repeated medical records by patient, not row, to prevent identity leakage.
- $[32,128][128,64]\to[32,64]$ costs about $32\cdot128\cdot64$ multiply-accumulates.
- For $J=m^{-1}\|Xw-y\|^2+\lambda\|w\|^2$,
  $\nabla J=2X^T(Xw-y)/m+2\lambda w$.
- Bernoulli negative log-likelihood is binary cross-entropy.
- An $[8,2048,4096]$ FP16 tensor occupies 128 MiB, excluding all other state.
- Collinear features destabilize coefficients; Ridge improves conditioning.
- Linear decodability from an embedding does not prove causal use by the original model.
