# Chapter 16 Solutions

[Back to Chapter 16](../16-vae-and-diffusion.md) | [中文](../../zh-CN/solutions/16-solutions.md)

## 1–2. ELBO and Gaussian KL

Insert $q(z\mid x)$ and apply Jensen:
$\log p(x)\ge\mathbb E_q\log p(x\mid z)-D_{KL}(q\|p)$. The gap is
$D_{KL}(q(z\mid x)\|p(z\mid x))$. Taking the expected Gaussian log-density ratio gives
$\frac12\sum_j(\mu_j^2+\sigma_j^2-\log\sigma_j^2-1)$.

## 3–4. Reparameterization and diagnosis

Use $z=\mu+\exp(0.5\log\sigma^2)\epsilon$ and hold $\epsilon$ fixed for finite-difference gradient checks. Plot
reconstruction and KL separately, prior samples, and latent interventions. Near-zero KL plus output insensitivity to $z$
supports posterior-collapse diagnosis.

## 5–7. Diffusion and audit

Recursively substitute $x_t=\sqrt{\alpha_t}x_{t-1}+\sqrt{1-\alpha_t}\epsilon_t$; Gaussian closure produces signal
$\sqrt{\bar\alpha_t}$ and noise variance $1-\bar\alpha_t$. Sweep schedule, sampler, steps, and guidance under a fixed
model, reporting quality, coverage, alignment, and P95 latency. Add nearest-training-example, duplication, group coverage,
and prompt-sensitivity audits rather than relying on FID alone.
