"""Chapter 16: visual demonstrations for VAE objectives and diffusion noising."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "docs" / "assets"
BLUE, ORANGE, GREEN, RED, NAVY = "#3478c0", "#f28e2b", "#3a9d5d", "#d94b4b", "#19324d"


def finish(fig, name):
    fig.tight_layout(); fig.savefig(OUTPUT / name, dpi=180, bbox_inches="tight", facecolor="white"); plt.close(fig)


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(11)
    mu, log_var = 1.2, np.log(.35**2)
    epsilon = rng.normal(size=5000); z = mu + np.exp(.5*log_var)*epsilon
    fig, ax = plt.subplots(figsize=(8,4.3))
    ax.hist(epsilon,bins=45,density=True,alpha=.55,color=BLUE,label=r"noise $\epsilon\sim N(0,1)$")
    ax.hist(z,bins=45,density=True,alpha=.55,color=ORANGE,label=r"latent $z=\mu+\sigma\epsilon$")
    ax.axvline(mu,color=NAVY,ls="--",label="encoder mean")
    ax.set(title="Reparameterization moves randomness outside the encoder",xlabel="value",ylabel="density"); ax.legend(frameon=False); ax.grid(alpha=.12)
    finish(fig,"vae-reparameterization.png")

    beta = np.logspace(-2,1,80)
    reconstruction = .12 + .34 * beta**.65
    kl = 1.6/(1+2.2*beta)
    fig,ax=plt.subplots(figsize=(8.2,4.3)); ax.semilogx(beta,reconstruction,color=RED,lw=2.3,label="reconstruction error"); ax.semilogx(beta,kl,color=BLUE,lw=2.3,label="KL to prior")
    ax.set(title="The VAE objective trades reconstruction against latent regularity",xlabel=r"KL weight $\beta$",ylabel="illustrative term value"); ax.grid(alpha=.15); ax.legend(frameon=False)
    finish(fig,"vae-elbo-tradeoff.png")

    clean = np.r_[rng.normal(-2,.45,2500),rng.normal(2,.55,2500)]
    alphas = [1.0,.75,.35,.05]
    fig,axes=plt.subplots(1,4,figsize=(12,3.4),sharex=True,sharey=True)
    for ax,alpha in zip(axes,alphas):
        noisy=np.sqrt(alpha)*clean+np.sqrt(1-alpha)*rng.normal(size=len(clean))
        ax.hist(noisy,bins=45,density=True,color=BLUE,alpha=.72); ax.set(title=f"signal fraction={alpha:.2f}",xlabel=r"$x_t$"); ax.grid(alpha=.1)
    axes[0].set_ylabel("density"); finish(fig,"diffusion-forward-noising.png")

    t=np.linspace(0,1,200); linear=1-t; cosine=np.cos(t*np.pi/2)**2
    fig,ax=plt.subplots(figsize=(8.2,4.3)); ax.plot(t,linear,color=ORANGE,lw=2.3,label="linear schedule"); ax.plot(t,cosine,color=GREEN,lw=2.3,label="cosine schedule")
    ax.set(title="A noise schedule controls how quickly information disappears",xlabel="normalized diffusion time",ylabel="cumulative signal fraction"); ax.grid(alpha=.15); ax.legend(frameon=False)
    finish(fig,"diffusion-noise-schedule.png")

    empirical_mean, empirical_std = z.mean(), z.std()
    assert abs(empirical_mean-mu)<.03 and abs(empirical_std-.35)<.03
    print(f"reparameterized_mean={empirical_mean:.3f}")
    print(f"reparameterized_std={empirical_std:.3f}")


if __name__ == "__main__":
    main()
