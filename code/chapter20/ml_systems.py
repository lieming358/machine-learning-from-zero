"""Chapter 20: batching, roofline intuition, quantization, and latency budgets."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT=Path(__file__).resolve().parents[2]; OUTPUT=ROOT/"docs"/"assets"
BLUE,ORANGE,GREEN,RED,NAVY="#3478c0","#f28e2b","#3a9d5d","#d94b4b","#19324d"


def finish(fig,name):
    fig.tight_layout(); fig.savefig(OUTPUT/name,dpi=180,bbox_inches="tight",facecolor="white"); plt.close(fig)


def quantize(x,bits):
    qmax=2**(bits-1)-1; scale=np.max(np.abs(x))/qmax; q=np.clip(np.round(x/scale),-qmax,qmax); return q*scale


def main():
    OUTPUT.mkdir(parents=True,exist_ok=True); rng=np.random.default_rng(41)
    batch=np.array([1,2,4,8,16,32,64,128]); throughput=1800*batch/(batch+10); latency=18+1.1*batch+.018*batch**2
    fig,ax1=plt.subplots(figsize=(8.6,4.4)); ax2=ax1.twinx(); ax1.plot(batch,throughput,"o-",color=GREEN,lw=2.3,label="throughput"); ax2.plot(batch,latency,"s-",color=RED,lw=2.3,label="P95 latency")
    ax1.set(title="Batching improves utilization but increases waiting and tail latency",xlabel="batch size",ylabel="requests/s",xscale="log",xticks=batch); ax1.get_xaxis().set_major_formatter(plt.ScalarFormatter()); ax2.set(ylabel="P95 latency (ms)"); ax1.grid(alpha=.15)
    ax1.legend(loc="upper left",frameon=False); ax2.legend(loc="center right",frameon=False); finish(fig,"batching-throughput-latency.png")

    intensity=np.logspace(-2,3,300); bandwidth=900; peak=120000; attainable=np.minimum(bandwidth*intensity,peak)
    fig,ax=plt.subplots(figsize=(8.3,4.4)); ax.loglog(intensity,attainable,color=BLUE,lw=2.5); ax.axhline(peak,color=RED,ls="--",label="compute ceiling"); ax.loglog(intensity,bandwidth*intensity,color=GREEN,ls="--",label="bandwidth ceiling")
    ax.set(title="Roofline: arithmetic intensity decides the bottleneck",xlabel="operations per byte",ylabel="attainable work rate (relative)"); ax.grid(alpha=.15,which="both"); ax.legend(frameon=False)
    finish(fig,"roofline-model.png")

    weights=rng.normal(0,1,200000); bits=np.array([2,3,4,6,8,16]); error=np.array([np.mean((weights-quantize(weights,int(b)))**2) if b<16 else 0 for b in bits]); size=bits/16
    fig,axes=plt.subplots(1,2,figsize=(10,4.2)); axes[0].semilogy(bits,error+1e-9,"o-",color=RED); axes[0].set(title="Lower precision increases quantization error",xlabel="bits per weight",ylabel="weight MSE")
    axes[1].plot(bits,size,"o-",color=BLUE); axes[1].set(title="Model storage scales with bit width",xlabel="bits per weight",ylabel="size relative to fp16")
    [ax.grid(alpha=.15) for ax in axes]; finish(fig,"quantization-quality-size.png")

    stages=["queue","tokenize","prefill","decode","network"]; values=np.array([18,7,85,240,35]); colors=[BLUE,ORANGE,GREEN,RED,NAVY]
    fig,ax=plt.subplots(figsize=(8.5,4.3)); left=0
    for stage,value,color in zip(stages,values,colors): ax.barh([0],[value],left=left,color=color,label=f"{stage}: {value} ms"); left+=value
    ax.set(title="End-to-end latency is a budget across multiple stages",xlabel="milliseconds",yticks=[]); ax.legend(ncol=3,frameon=False,loc="upper center",bbox_to_anchor=(.5,-.12))
    finish(fig,"inference-latency-budget.png")

    kv_tokens=np.arange(128,8193,128); layers,heads,head_dim=32,32,128; kv_gib=2*layers*heads*head_dim*kv_tokens*2/1024**3
    fig,ax=plt.subplots(figsize=(8.3,4.2)); ax.plot(kv_tokens,kv_gib,color=ORANGE,lw=2.4); ax.set(title="KV-cache memory grows linearly with generated context",xlabel="cached tokens",ylabel="KV cache (GiB per sequence, fp16)"); ax.grid(alpha=.15)
    finish(fig,"kv-cache-growth.png")

    assert error[-2] < error[2]
    print(f"best_throughput_batch={batch[np.argmax(throughput)]:d}")
    print(f"int4_relative_size={4/16:.2f}")


if __name__=="__main__": main()
