"""Chapter 17: implicit matrix factorization and ranking diagnostics."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "docs" / "assets"
BLUE, ORANGE, GREEN, RED, NAVY = "#3478c0", "#f28e2b", "#3a9d5d", "#d94b4b", "#19324d"


def finish(fig,name):
    fig.tight_layout(); fig.savefig(OUTPUT/name,dpi=180,bbox_inches="tight",facecolor="white"); plt.close(fig)


def main():
    OUTPUT.mkdir(parents=True,exist_ok=True); rng=np.random.default_rng(13)
    users,items,rank=42,60,4
    true_u=rng.normal(size=(users,rank)); true_v=rng.normal(size=(items,rank))
    score=true_u@true_v.T; probability=1/(1+np.exp(-score))
    observed=rng.random((users,items))<(.13+.25*probability); clicks=(rng.random((users,items))<probability)&observed
    shown=np.where(observed,clicks.astype(float),np.nan)
    fig,ax=plt.subplots(figsize=(9,4.8)); image=ax.imshow(shown,aspect="auto",cmap="YlGnBu",vmin=0,vmax=1)
    ax.set(title="Recommendation data is a sparse exposure-and-feedback matrix",xlabel="item",ylabel="user"); fig.colorbar(image,ax=ax,ticks=[0,1],label="observed feedback")
    finish(fig,"recommender-sparse-matrix.png")

    u=rng.normal(0,.15,(users,rank)); v=rng.normal(0,.15,(items,rank)); losses=[]; lr=.8; reg=.001
    pairs=np.argwhere(observed); labels=clicks[observed].astype(float)
    for epoch in range(1200):
        logits=np.sum(u[pairs[:,0]]*v[pairs[:,1]],axis=1); p=1/(1+np.exp(-np.clip(logits,-30,30)))
        error=p-labels; gu=np.zeros_like(u); gv=np.zeros_like(v)
        np.add.at(gu,pairs[:,0],error[:,None]*v[pairs[:,1]]); np.add.at(gv,pairs[:,1],error[:,None]*u[pairs[:,0]])
        gu=gu/len(pairs)+reg*u; gv=gv/len(pairs)+reg*v; u-=lr*gu; v-=lr*gv
        if epoch%10==0: losses.append(np.mean(np.logaddexp(0,logits)-labels*logits)+reg*(np.mean(u*u)+np.mean(v*v)))
    fig,axes=plt.subplots(1,2,figsize=(10,4.1)); axes[0].plot(np.arange(len(losses))*10,losses,color=BLUE,lw=2.3); axes[0].set(title="Observed-feedback objective",xlabel="epoch",ylabel="BCE + regularization")
    axes[1].scatter(v[:,0],v[:,1],c=np.arange(items),cmap="viridis",s=35); axes[1].set(title="Learned item embedding",xlabel="latent dimension 1",ylabel="latent dimension 2")
    [ax.grid(alpha=.15) for ax in axes]; finish(fig,"matrix-factorization-training.png")

    positions=np.arange(1,21); discount=1/np.log2(positions+1); binary=np.r_[1,0,1,0,0,1,np.zeros(14)]; graded=np.r_[3,0,2,0,1,2,np.zeros(14)]
    fig,ax=plt.subplots(figsize=(8.5,4.2)); ax.bar(positions,discount,color=BLUE,alpha=.65,label="rank discount"); ax.scatter(positions[binary.astype(bool)],discount[binary.astype(bool)],color=RED,s=55,label="relevant item")
    ax.set(title="Ranking metrics value relevant items more near the top",xlabel="rank",ylabel="discount weight"); ax.grid(axis="y",alpha=.15); ax.legend(frameon=False)
    finish(fig,"ranking-metric-discount.png")

    popularity=np.arange(1,21)**-1.05; popularity/=popularity.sum(); uniform=np.ones(20)/20
    fig,ax=plt.subplots(figsize=(8.5,4.2)); ax.plot(np.arange(1,21),popularity,"o-",color=RED,label="logged exposure"); ax.plot(np.arange(1,21),uniform,"--",color=GREEN,label="uniform reference")
    ax.set(title="Logged feedback overrepresents already-popular items",xlabel="item popularity rank",ylabel="exposure probability"); ax.grid(alpha=.15); ax.legend(frameon=False)
    finish(fig,"recommendation-exposure-bias.png")

    assert losses[-1] < losses[0]
    print(f"observed_fraction={observed.mean():.3f}")
    print(f"loss_reduction={losses[0]-losses[-1]:.4f}")


if __name__ == "__main__":
    main()
