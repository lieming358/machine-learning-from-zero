"""Chapter 13: a NumPy two-tower model trained with InfoNCE for paired retrieval."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "docs" / "assets"
BLUE, ORANGE, GREEN, RED, NAVY = "#3478c0", "#f28e2b", "#3a9d5d", "#d94b4b", "#19324d"


def normalize(z):
    norm = np.linalg.norm(z, axis=1, keepdims=True) + 1e-9
    return z / norm, norm


def softmax_rows(logits):
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


def retrieval_metrics(query_embedding, document_embedding):
    score = query_embedding @ document_embedding.T
    order = np.argsort(-score, axis=1)
    target = np.arange(len(score))
    ranks = np.array([np.flatnonzero(order[i] == target[i])[0] + 1 for i in range(len(score))])
    return np.mean(ranks == 1), np.mean(ranks <= 5), np.mean(1 / ranks), score


def train_two_tower(query, document, embedding_dim=8, temperature=.12, epochs=700, lr=.8, seed=7):
    rng = np.random.default_rng(seed)
    wq = rng.normal(0, .12, (query.shape[1], embedding_dim))
    wd = rng.normal(0, .12, (document.shape[1], embedding_dim))
    target = np.eye(len(query)); losses=[]; recall=[]
    for epoch in range(epochs):
        raw_q, raw_d = query @ wq, document @ wd
        eq, nq = normalize(raw_q); ed, nd = normalize(raw_d)
        logits = eq @ ed.T / temperature
        probability = softmax_rows(logits)
        loss = -np.mean(np.log(np.diag(probability) + 1e-12))
        dlogits = (probability - target) / len(query)
        deq = dlogits @ ed / temperature
        ded = dlogits.T @ eq / temperature
        dzq = (deq - eq * np.sum(deq * eq, axis=1, keepdims=True)) / nq
        dzd = (ded - ed * np.sum(ded * ed, axis=1, keepdims=True)) / nd
        wq -= lr * query.T @ dzq
        wd -= lr * document.T @ dzd
        if epoch % 10 == 0:
            losses.append(loss); recall.append(retrieval_metrics(eq,ed)[0])
    eq = normalize(query @ wq)[0]; ed = normalize(document @ wd)[0]
    return wq, wd, eq, ed, np.array(losses), np.array(recall)


def finish(fig, name):
    fig.tight_layout(); fig.savefig(OUTPUT / name, dpi=180, bbox_inches="tight", facecolor="white"); plt.close(fig)


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(42)
    n, latent_dim, input_dim = 80, 6, 18
    classes = np.repeat(np.arange(4), n//4)
    centers = rng.normal(0, 1.5, (4, latent_dim))
    latent = centers[classes] + rng.normal(0, .45, (n, latent_dim))
    aq, ad = rng.normal(size=(latent_dim,input_dim)), rng.normal(size=(latent_dim,input_dim))
    query = latent @ aq + rng.normal(0,.35,(n,input_dim))
    document = latent @ ad + rng.normal(0,.35,(n,input_dim))

    initial_q = normalize(query[:,:8])[0]
    initial_d = normalize(document[:,:8])[0]
    before = retrieval_metrics(initial_q,initial_d)[0]
    wq, wd, eq, ed, losses, recall = train_two_tower(query,document)
    after_r1, after_r5, mrr, score = retrieval_metrics(eq,ed)

    combined_before = np.vstack([initial_q,initial_d]); combined_after = np.vstack([eq,ed])
    before_2d = PCA(n_components=2).fit_transform(combined_before)
    after_2d = PCA(n_components=2).fit_transform(combined_after)
    fig, axes = plt.subplots(1,2,figsize=(11,4.4))
    for axis, points, title in zip(axes,[before_2d,after_2d],["Before contrastive training","After InfoNCE training"]):
        for i in range(n):
            axis.plot([points[i,0],points[n+i,0]],[points[i,1],points[n+i,1]],color="#bbbbbb",lw=.45,alpha=.45)
        axis.scatter(points[:n,0],points[:n,1],c=classes,cmap="tab10",marker="o",s=24,label="queries")
        axis.scatter(points[n:,0],points[n:,1],c=classes,cmap="tab10",marker="x",s=28,label="documents")
        axis.set_title(title); axis.grid(alpha=.12); axis.legend(frameon=False)
    finish(fig,"contrastive-representation-before-after.png")

    before_score = initial_q @ initial_d.T
    fig, axes = plt.subplots(1,2,figsize=(9.5,4.2))
    for axis,matrix,title in zip(axes,[before_score,score],["Before training","After training"]):
        image=axis.imshow(matrix,cmap="coolwarm",vmin=-1,vmax=1,aspect="auto")
        axis.set(title=title,xlabel="document index",ylabel="query index")
    fig.colorbar(image,ax=axes.ravel().tolist(),shrink=.8,label="cosine similarity")
    fig.subplots_adjust(wspace=.28,right=.88)
    fig.savefig(OUTPUT/"retrieval-similarity-matrix.png",dpi=180,bbox_inches="tight",facecolor="white"); plt.close(fig)

    steps=np.arange(len(losses))*10
    fig,axes=plt.subplots(1,2,figsize=(10.5,4))
    axes[0].plot(steps,losses,color=BLUE,lw=2.3); axes[0].set(title="InfoNCE optimization",xlabel="epoch",ylabel="loss")
    axes[1].plot(steps,recall,color=GREEN,lw=2.3); axes[1].axhline(before,color=RED,ls="--",label="untrained baseline"); axes[1].set(title="Retrieval quality during training",xlabel="epoch",ylabel="Recall@1",ylim=(0,1.03)); axes[1].legend(frameon=False)
    for axis in axes: axis.grid(alpha=.14)
    finish(fig,"contrastive-training-retrieval.png")

    similarities=np.linspace(-.2,.9,300)
    fig,ax=plt.subplots(figsize=(8,4.2))
    for temperature,color in [(.05,RED),(.12,ORANGE),(.4,BLUE)]:
        probability=np.exp(similarities/temperature)/(np.exp(similarities/temperature)+np.exp(.25/temperature))
        ax.plot(similarities,probability,color=color,lw=2.2,label=f"temperature={temperature}")
    ax.set(title="Temperature controls how sharply similarity changes softmax preference",xlabel="candidate cosine similarity",ylabel="two-candidate softmax probability")
    ax.grid(alpha=.14); ax.legend(frameon=False); finish(fig,"contrastive-temperature.png")

    index=NearestNeighbors(n_neighbors=5,metric="cosine").fit(ed)
    neighbors=index.kneighbors(eq[:3],return_distance=False)
    print(f"untrained_recall_at_1={before:.3f}")
    print(f"trained_recall_at_1={after_r1:.3f}")
    print(f"trained_recall_at_5={after_r5:.3f}")
    print(f"trained_mrr={mrr:.3f}")
    print(f"first_three_neighbor_indices={neighbors.tolist()}")
    assert after_r1 > .85 and after_r5 > .95 and after_r1 > before + .5


if __name__ == "__main__":
    main()
