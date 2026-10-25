"""Chapter 21: reproducible project evidence and interview mastery diagnostics."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT=Path(__file__).resolve().parents[2]; OUTPUT=ROOT/"docs"/"assets"
BLUE,ORANGE,GREEN,RED,NAVY="#3478c0","#f28e2b","#3a9d5d","#d94b4b","#19324d"


def finish(fig,name):
    fig.tight_layout(); fig.savefig(OUTPUT/name,dpi=180,bbox_inches="tight",facecolor="white"); plt.close(fig)


def main():
    OUTPUT.mkdir(parents=True,exist_ok=True); rng=np.random.default_rng(55)
    components=["data cleanup","representation","loss","augmentation","reranking"]
    baseline=.71; gains=np.array([.035,.052,.018,.024,.041]); noise=rng.normal(0,.006,(10,len(gains)))
    means=baseline+gains; std=noise.std(axis=0)
    fig,ax=plt.subplots(figsize=(9,4.5)); ax.bar(components,means-baseline,yerr=std,color=[BLUE,ORANGE,GREEN,RED,NAVY],capsize=4); ax.axhline(0,color="black",lw=.8)
    ax.set(title="An ablation asks which component causes the improvement",ylabel="metric gain over baseline"); plt.setp(ax.get_xticklabels(),rotation=20,ha="right"); ax.grid(axis="y",alpha=.15)
    finish(fig,"capstone-ablation.png")

    skills=["intuition","math","implementation","diagnosis","systems","communication"]; before=np.array([3,2,2,2,1,3]); after=np.array([4.3,4,4.2,4,3.8,4.2]); angle=np.linspace(0,2*np.pi,len(skills),endpoint=False); angle=np.r_[angle,angle[0]]
    fig=plt.figure(figsize=(6.2,5.5)); ax=fig.add_subplot(111,polar=True); ax.plot(angle,np.r_[before,before[0]],"o-",color=ORANGE,label="initial self-check"); ax.fill(angle,np.r_[before,before[0]],color=ORANGE,alpha=.12); ax.plot(angle,np.r_[after,after[0]],"o-",color=BLUE,label="capstone target"); ax.fill(angle,np.r_[after,after[0]],color=BLUE,alpha=.12); ax.set_xticks(angle[:-1]); ax.set_xticklabels(skills); ax.set_ylim(0,5); ax.set_title("Course mastery is multi-dimensional"); ax.legend(frameon=False,loc="lower right",bbox_to_anchor=(1.25,-.05))
    finish(fig,"course-mastery-radar.png")

    stages=["question","data card","baseline","method","ablation","error analysis","system profile","report"]
    fig,ax=plt.subplots(figsize=(11,3.2)); ax.axis("off")
    for i,label in enumerate(stages):
        x=.02+i*.12; color=GREEN if i in (0,2,4,6) else BLUE; ax.add_patch(plt.Rectangle((x,.36),.105,.28,facecolor=color,alpha=.85)); ax.text(x+.0525,.5,label,ha="center",va="center",color="white",fontsize=9)
        if i<len(stages)-1: ax.annotate("",xy=(x+.118,.5),xytext=(x+.105,.5),arrowprops=dict(arrowstyle="->",color=NAVY))
    ax.set(title="A defensible project is an evidence chain, not a final score")
    finish(fig,"research-project-evidence-chain.png")

    questions=np.arange(1,41); recall=1-np.exp(-questions/11); transfer=.25+.68*(1-np.exp(-questions/18));
    fig,ax=plt.subplots(figsize=(8.4,4.2)); ax.plot(questions,recall,color=BLUE,lw=2.3,label="definition recall"); ax.plot(questions,transfer,color=GREEN,lw=2.3,label="novel-case reasoning"); ax.set(title="Interview practice must move from recall to transfer",xlabel="deliberate practice sessions",ylabel="illustrative mastery"); ax.grid(alpha=.15); ax.legend(frameon=False)
    finish(fig,"interview-practice-transfer.png")

    assert np.all(means>baseline)
    print(f"largest_ablation_gain={gains.max():.3f}")
    print(f"target_average_mastery={after.mean():.2f}")


if __name__=="__main__": main()
