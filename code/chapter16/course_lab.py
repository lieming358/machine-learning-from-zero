"""Chapter 16 lab: train a tiny linear VAE and a one-dimensional denoiser."""

from __future__ import annotations

import numpy as np


class LinearVAE:
    """A 2D-to-1D VAE whose gradients are short enough to inspect by hand."""

    def __init__(self, seed: int = 3):
        rng = np.random.default_rng(seed)
        self.w_mu = rng.normal(0, 0.1, 2)
        self.b_mu = 0.0
        self.w_logvar = rng.normal(0, 0.05, 2)
        self.b_logvar = -0.2
        self.w_decoder = rng.normal(0, 0.1, 2)
        self.b_decoder = np.zeros(2)

    def loss_and_gradients(self, x: np.ndarray, epsilon: np.ndarray, beta: float = 0.1):
        n = len(x)
        mu = x @ self.w_mu + self.b_mu
        logvar = np.clip(x @ self.w_logvar + self.b_logvar, -6.0, 4.0)
        sigma = np.exp(0.5 * logvar)
        z = mu + sigma * epsilon
        reconstruction = z[:, None] * self.w_decoder[None, :] + self.b_decoder

        reconstruction_loss = 0.5 * np.mean(np.sum((reconstruction - x) ** 2, axis=1))
        kl = 0.5 * np.mean(mu**2 + np.exp(logvar) - logvar - 1.0)
        loss = reconstruction_loss + beta * kl

        d_reconstruction = (reconstruction - x) / n
        grad_w_decoder = np.sum(d_reconstruction * z[:, None], axis=0)
        grad_b_decoder = d_reconstruction.sum(axis=0)
        d_z = d_reconstruction @ self.w_decoder
        d_mu = d_z + beta * mu / n
        d_logvar = d_z * epsilon * 0.5 * sigma + beta * 0.5 * (np.exp(logvar) - 1.0) / n
        gradients = (
            x.T @ d_mu,
            d_mu.sum(),
            x.T @ d_logvar,
            d_logvar.sum(),
            grad_w_decoder,
            grad_b_decoder,
        )
        return float(loss), float(reconstruction_loss), float(kl), gradients

    def fit(self, x: np.ndarray, steps: int = 1800, lr: float = 0.035, beta: float = 0.1):
        rng = np.random.default_rng(19)
        history = []
        for step in range(steps):
            epsilon = rng.normal(size=len(x))
            loss, reconstruction, kl, gradients = self.loss_and_gradients(x, epsilon, beta)
            parameters = (
                "w_mu", "b_mu", "w_logvar", "b_logvar", "w_decoder", "b_decoder"
            )
            for name, gradient in zip(parameters, gradients):
                setattr(self, name, getattr(self, name) - lr * gradient)
            if step % 100 == 0 or step == steps - 1:
                history.append((loss, reconstruction, kl))
        return history

    def reconstruct_mean(self, x: np.ndarray) -> np.ndarray:
        mu = x @ self.w_mu + self.b_mu
        return mu[:, None] * self.w_decoder[None, :] + self.b_decoder

    def sample(self, count: int, seed: int = 0) -> np.ndarray:
        z = np.random.default_rng(seed).normal(size=count)
        return z[:, None] * self.w_decoder[None, :] + self.b_decoder


def diffusion_training_set(x0: np.ndarray, alpha_bar: np.ndarray, seed: int = 5):
    rng = np.random.default_rng(seed)
    time_index = rng.integers(1, len(alpha_bar), size=len(x0))
    epsilon = rng.normal(size=len(x0))
    retained = alpha_bar[time_index]
    xt = np.sqrt(retained) * x0 + np.sqrt(1.0 - retained) * epsilon
    tau = time_index / (len(alpha_bar) - 1)
    features = np.column_stack([np.ones(len(x0)), xt, tau, xt * tau, xt**3])
    return features, epsilon


def main() -> None:
    rng = np.random.default_rng(7)
    latent = rng.normal(size=600)
    x = np.column_stack([latent + 0.15 * rng.normal(size=600), 0.7 * latent + 0.15 * rng.normal(size=600)])
    x = (x - x.mean(axis=0)) / x.std(axis=0)

    vae = LinearVAE()
    history = vae.fit(x)
    reconstruction_mse = np.mean((vae.reconstruct_mean(x) - x) ** 2)
    prior_samples = vae.sample(5)

    steps = 30
    beta = np.linspace(0.002, 0.08, steps)
    alpha_bar = np.r_[1.0, np.cumprod(1.0 - beta)]
    clean = np.r_[rng.normal(-2.0, 0.45, 3000), rng.normal(2.0, 0.45, 3000)]
    features, epsilon = diffusion_training_set(clean, alpha_bar)
    regularizer = 1e-3 * np.eye(features.shape[1])
    denoiser = np.linalg.solve(features.T @ features + regularizer, features.T @ epsilon)
    predicted_noise = features @ denoiser
    denoising_mse = np.mean((epsilon - predicted_noise) ** 2)

    print(f"vae_loss={history[0][0]:.4f}->{history[-1][0]:.4f}")
    print(f"vae_reconstruction_mse={reconstruction_mse:.4f}")
    print(f"prior_sample_shape={prior_samples.shape}")
    print(f"noise_prediction_mse={denoising_mse:.4f}")

    assert history[-1][0] < history[0][0] * 0.7
    assert reconstruction_mse < 0.4
    assert prior_samples.shape == (5, 2)
    assert denoising_mse < np.mean(epsilon**2)


if __name__ == "__main__":
    main()
