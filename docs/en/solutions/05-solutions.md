# Chapter 5 — Hints and Reference Answers

[← Chapter 5](../05-generative-classifiers.md) | [中文答案](../../zh-CN/solutions/05-solutions.md)

## 1. Medical-test posterior

$p(D\mid+)=0.9(0.01)/[0.9(0.01)+0.05(0.99)]\approx0.154$. The posterior is not sensitivity;
the low disease base rate creates many more false positives than intuition first suggests.

## 2. Shared covariance

The two $-x^T\Sigma^{-1}x/2$ terms cancel in the Gaussian log-likelihood ratio. Linear terms, mean
constants, and the log prior remain. Class-specific covariances prevent that cancellation.

## 3. Covariance experiment

Duplicate features push the smallest eigenvalue toward zero and increase the condition number. Report
conditioning and predictions as well as failure. Jitter can stabilize arithmetic but cannot create
missing information.

## 4. Smoothing

$\alpha=0$ can produce $\log0$; a small positive value prevents zero probabilities; a huge value makes
word probabilities nearly uniform. Select it on validation data.

## 5. Multimodal counterexample

One Gaussian per class may put high density between two real clusters. QDA changes covariance but still
cannot represent several modes per class. Plot density and boundaries rather than accuracy alone.

## 6. Intent-classification rubric

Use identical leakage-safe splits; report majority baselines, macro-F1, log loss/calibration, latency,
model size, and multiple seeds. Separate predictive quality from system cost.
