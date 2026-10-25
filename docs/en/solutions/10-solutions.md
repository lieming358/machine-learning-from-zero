# Chapter 10 Solutions

[Back to Chapter 10](../10-clustering-em-and-elbo.md) | [中文](../../zh-CN/solutions/10-solutions.md)

## 1–2. k-means

For each round, list squared distances, assignments, new coordinate-wise means, and objective $J$; $J$ must not rise.
For $J(\mu)=\sum_{i\in C}\|x^{(i)}-\mu\|^2$, setting
$\nabla J=2|C|\mu-2\sum_i x^{(i)}=0$ gives the mean, and Hessian $2|C|I$ proves it is a minimum.

## 3–4. GMM and EM

Compute unnormalized $a_k=\pi_k\mathcal N(x;\mu_k,\sigma_k^2)$ and normalize
$\gamma_k=a_k/\sum_j a_j$; use log-sum-exp numerically. Maximizing the responsibility-weighted complete log-likelihood
under $\sum_k\pi_k=1$ gives $\pi_k=N_k/m$ and the responsibility-weighted mean.

## 5–7. Evaluation

Moons violate elliptical clusters, unequal densities challenge one DBSCAN radius, and outliers drag means/covariances.
Combine inertia elbows, silhouette, resampling stability, and domain utility; “no reliable discrete clusters” is a valid
conclusion. For embedding clustering, hold the corpus fixed and audit encoder, normalization, distance, stability,
human topic coherence, and downstream value.
