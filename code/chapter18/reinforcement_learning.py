"""Chapter 18: value iteration, temporal-difference learning, bandits, and PPO clipping."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT=Path(__file__).resolve().parents[2]; OUTPUT=ROOT/"docs"/"assets"
BLUE,ORANGE,GREEN,RED,NAVY="#3478c0","#f28e2b","#3a9d5d","#d94b4b","#19324d"
ACTIONS=np.array([[-1,0],[1,0],[0,-1],[0,1]])


def finish(fig,name):
    fig.tight_layout(); fig.savefig(OUTPUT/name,dpi=180,bbox_inches="tight",facecolor="white"); plt.close(fig)


def value_iteration(size=6,gamma=.95,iterations=300):
    goal=(0,size-1); trap=(2,3); value=np.zeros((size,size))
    for _ in range(iterations):
        new=value.copy()
        for r in range(size):
            for c in range(size):
                if (r,c) in (goal,trap): continue
                candidates=[]
                for dr,dc in ACTIONS:
                    nr,nc=np.clip(r+dr,0,size-1),np.clip(c+dc,0,size-1)
                    reward=10 if (nr,nc)==goal else (-8 if (nr,nc)==trap else -1)
                    candidates.append(reward+gamma*value[nr,nc])
                new[r,c]=max(candidates)
        if np.max(np.abs(new-value))<1e-10: break
        value=new
    policy=np.zeros((size,size),int)
    for r in range(size):
        for c in range(size):
            candidates=[]
            for dr,dc in ACTIONS:
                nr,nc=np.clip(r+dr,0,size-1),np.clip(c+dc,0,size-1)
                reward=10 if (nr,nc)==goal else (-8 if (nr,nc)==trap else -1)
                candidates.append(reward+gamma*value[nr,nc])
            policy[r,c]=int(np.argmax(candidates))
    return value,policy,goal,trap


def main():
    OUTPUT.mkdir(parents=True,exist_ok=True); rng=np.random.default_rng(21)
    value,policy,goal,trap=value_iteration(); fig,ax=plt.subplots(figsize=(6.2,5.5)); image=ax.imshow(value,cmap="RdYlGn")
    for r in range(value.shape[0]):
        for c in range(value.shape[1]):
            if (r,c)==goal: label="G"
            elif (r,c)==trap: label="X"
            else: label=["↑","↓","←","→"][policy[r,c]]
            ax.text(c,r,label,ha="center",va="center",fontsize=14,color=NAVY)
    ax.set(title="Value iteration propagates delayed reward into a policy",xlabel="column",ylabel="row"); fig.colorbar(image,ax=ax,label="optimal state value")
    finish(fig,"gridworld-value-policy.png")

    true=np.array([.15,.22,.3,.5,.54]); runs=150; horizon=900; epsilons=[0,.05,.15]
    fig,ax=plt.subplots(figsize=(8.5,4.4))
    for eps,color in zip(epsilons,[RED,GREEN,BLUE]):
        regret=np.zeros(horizon)
        for _ in range(runs):
            counts=np.zeros(len(true)); estimates=np.zeros(len(true))
            for t in range(horizon):
                action=rng.integers(len(true)) if rng.random()<eps else int(np.argmax(estimates+1e-9*rng.normal(size=len(true))))
                reward=rng.random()<true[action]; counts[action]+=1; estimates[action]+=(reward-estimates[action])/counts[action]
                regret[t]+=true.max()-true[action]
        ax.plot(np.cumsum(regret/runs),color=color,lw=2,label=f"epsilon={eps}")
    ax.set(title="Exploration changes cumulative regret",xlabel="interaction",ylabel="expected cumulative regret"); ax.grid(alpha=.15); ax.legend(frameon=False)
    finish(fig,"bandit-exploration-regret.png")

    alpha=.12; gamma=.92; episodes=300; values=np.zeros(7); errors=[]
    for episode in range(episodes):
        state=3; episode_error=[]
        while state not in (0,6):
            next_state=state+(1 if rng.random()<.5 else -1); reward=1.0 if next_state==6 else 0.0
            delta=reward+gamma*values[next_state]-values[state]; values[state]+=alpha*delta; episode_error.append(abs(delta)); state=next_state
        errors.append(np.mean(episode_error))
    fig,ax=plt.subplots(figsize=(8.3,4.2)); ax.plot(errors,color=ORANGE,alpha=.7); ax.plot(np.convolve(errors,np.ones(20)/20,mode="valid"),color=NAVY,lw=2.2,label="20-episode mean")
    ax.set(title="TD error shrinks as bootstrapped values improve",xlabel="episode",ylabel="mean absolute TD error"); ax.grid(alpha=.15); ax.legend(frameon=False)
    finish(fig,"td-learning-diagnostics.png")

    ratio=np.linspace(.5,1.5,500); advantage_pos=np.ones_like(ratio); advantage_neg=-np.ones_like(ratio); clip=.2
    positive=np.minimum(ratio*advantage_pos,np.clip(ratio,1-clip,1+clip)*advantage_pos)
    negative=np.minimum(ratio*advantage_neg,np.clip(ratio,1-clip,1+clip)*advantage_neg)
    fig,ax=plt.subplots(figsize=(8.2,4.2)); ax.plot(ratio,positive,color=GREEN,lw=2.3,label="positive advantage"); ax.plot(ratio,negative,color=RED,lw=2.3,label="negative advantage"); ax.axvspan(.8,1.2,color=BLUE,alpha=.08,label="unclipped region")
    ax.set(title="PPO clipping limits incentives for large policy-ratio changes",xlabel="new/old action probability ratio",ylabel="clipped surrogate contribution"); ax.grid(alpha=.15); ax.legend(frameon=False)
    finish(fig,"ppo-clipped-objective.png")

    assert value[5,0] < value[0,4]
    print(f"start_value={value[5,0]:.3f}")
    print(f"final_td_error={np.mean(errors[-20:]):.4f}")


if __name__=="__main__": main()
