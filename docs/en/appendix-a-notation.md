# Appendix A — Course-wide Notation and Defaults

[← Home](../../README.en.md) | [中文](../zh-CN/appendix-a-notation.md) | [Chapter 1](01-what-is-ml.md)

This page is the single notation standard for the course. An imported convention must state its mapping.

| Symbol | Fixed meaning |
|---|---|
| $m,d,K$ | number of examples, features, and classes |
| $x$ | one unspecified feature vector |
| $x^{(i)}$ | feature vector of example $i$; parentheses mean an index, not a power |
| $x_j^{(i)}=X_{ij}$ | feature $j$ of example $i$ |
| $X\in\mathbb R^{m\times d}$ | data matrix; examples are rows, features are columns |
| $y^{(i)},\hat y^{(i)}$ | true and predicted target for example $i$ |
| $f_\theta,\theta$ | model and all learned parameters |
| $\ell,J$ | per-example loss and aggregate objective |
| $\eta,\lambda$ | learning rate and regularization strength |
| $\nabla_\theta J$ | gradient of the objective with respect to parameters |

The course does **not** use $x_i$ for example $i$, because that conflicts with feature indexing.
Mathematics uses one-based indices; Python uses zero-based indices, so $X_{32}$ is `X[2, 1]`.

Binary labels default to $\{0,1\}$; margin methods may explicitly switch to $\{-1,+1\}$. LLM hidden
states use shape $[B,T,D]$: batch, sequence length, and hidden dimension.

By default, preprocessing is fitted on training data only; validation selects models; test data is used
for final acceptance; random seeds and variation are reported; matrix shapes and assumptions are stated.

For every formula ask: What is each symbol and shape? What does the index traverse? What is known or
learned? Is the equality exact, definitional, approximate, or optimal? Which assumptions make it valid?
