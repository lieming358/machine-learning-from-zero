"""Chapter 19: calibration, confounding, drift, and monitoring demonstrations."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss, roc_auc_score


ROOT=Path(__file__).resolve().parents[2]; OUTPUT=ROOT/"docs"/"assets"
BLUE,ORANGE,GREEN,RED,NAVY="#3478c0","#f28e2b","#3a9d5d","#d94b4b","#19324d"


def finish(fig,name):
    fig.tight_layout(); fig.savefig(OUTPUT/name,dpi=180,bbox_inches="tight",facecolor="white"); plt.close(fig)


def main():
    OUTPUT.mkdir(parents=True,exist_ok=True); rng=np.random.default_rng(31)
    n=6000; latent=rng.normal(size=n); true_p=1/(1+np.exp(-latent)); y=rng.random(n)<true_p
    confident=1/(1+np.exp(-2.2*latent)); calibrated=true_p
    fig,ax=plt.subplots(figsize=(6.2,5.2)); ax.plot([0,1],[0,1],"--",color=NAVY,label="perfect calibration")
    for prediction,color,label in [(calibrated,GREEN,"calibrated"),(confident,RED,"overconfident")]:
        frac,mean=calibration_curve(y,prediction,n_bins=10,strategy="quantile"); ax.plot(mean,frac,"o-",color=color,label=label)
    ax.set(title="Ranking quality and probability calibration are different",xlabel="mean predicted probability",ylabel="observed positive rate"); ax.grid(alpha=.15); ax.legend(frameon=False)
    finish(fig,"calibration-reliability.png")

    # Treatment helps within both groups, but is assigned mostly to the low-baseline group.
    # Aggregation therefore reverses the sign: a controlled Simpson's-paradox example.
    group=rng.integers(0,2,5000); treatment=np.where(group==0,rng.random(5000)<.2,rng.random(5000)<.8)
    base=np.where(group==0,.72,.25); outcome=rng.random(5000)<np.clip(base+.07*treatment,0,1)
    rates=np.zeros((2,2))
    for g in range(2):
        for t in range(2): rates[g,t]=outcome[(group==g)&(treatment==t)].mean()
    overall=[outcome[~treatment].mean(),outcome[treatment].mean()]
    fig,axes=plt.subplots(1,2,figsize=(10,4),sharey=True)
    axes[0].bar(["control","treated"],overall,color=[BLUE,ORANGE]); axes[0].set(title="Aggregated observation",ylabel="outcome rate")
    x=np.arange(2); axes[1].bar(x-.18,rates[:,0],.36,label="control",color=BLUE); axes[1].bar(x+.18,rates[:,1],.36,label="treated",color=ORANGE); axes[1].set(xticks=x,xticklabels=["group A","group B"],title="Conditioned on confounder"); axes[1].legend(frameon=False)
    [ax.grid(axis="y",alpha=.15) for ax in axes]; finish(fig,"confounding-simpson.png")

    train=rng.normal(0,1,5000); current=rng.normal(.7,1.25,5000)
    bins=np.linspace(-4,5,45); fig,ax=plt.subplots(figsize=(8.4,4.3)); ax.hist(train,bins=bins,density=True,alpha=.55,color=BLUE,label="training"); ax.hist(current,bins=bins,density=True,alpha=.55,color=ORANGE,label="current")
    ax.set(title="Input drift can be observed before labels arrive",xlabel="feature value",ylabel="density"); ax.grid(alpha=.12); ax.legend(frameon=False)
    finish(fig,"feature-drift.png")

    days=np.arange(60); data_quality=.995-.0004*days; latency=120+15*np.sin(days/7)+np.where(days>38,70,0); quality=.86-.001*np.maximum(days-30,0)+rng.normal(0,.006,len(days))
    fig,axes=plt.subplots(3,1,figsize=(9,6.8),sharex=True)
    for ax,series,color,label,threshold in [(axes[0],data_quality,BLUE,"valid record rate",.97),(axes[1],latency,ORANGE,"P95 latency (ms)",180),(axes[2],quality,GREEN,"delayed quality",.82)]:
        ax.plot(days,series,color=color,lw=2); ax.axhline(threshold,color=RED,ls="--",label="alert threshold"); ax.set(ylabel=label); ax.grid(alpha=.13)
    axes[0].legend(frameon=False); axes[-1].set(xlabel="day"); fig.suptitle("Production monitoring needs data, system, and outcome signals")
    finish(fig,"mlops-monitoring-signals.png")

    print(f"auc_calibrated={roc_auc_score(y,calibrated):.3f}")
    print(f"auc_overconfident={roc_auc_score(y,confident):.3f}")
    print(f"brier_calibrated={brier_score_loss(y,calibrated):.3f}")
    print(f"brier_overconfident={brier_score_loss(y,confident):.3f}")
    print(f"simpson_overall_control_treated={overall[0]:.3f},{overall[1]:.3f}")
    assert abs(roc_auc_score(y,calibrated)-roc_auc_score(y,confident))<1e-8
    assert brier_score_loss(y,calibrated)<brier_score_loss(y,confident)
    assert np.all(rates[:,1] > rates[:,0]) and overall[1] < overall[0]


if __name__=="__main__": main()
