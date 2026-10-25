"""Chapter 7: linear maximum-margin classifier from scratch and SVM figures."""

from __future__ import annotations
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_blobs, make_moons
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

OUTPUT = Path(__file__).resolve().parents[2] / "docs" / "assets"


class LinearSVM:
    def __init__(self, c: float = 1., learning_rate: float = .02, steps: int = 6000) -> None: self.c, self.learning_rate, self.steps = c, learning_rate, steps
    def fit(self, x: np.ndarray, y: np.ndarray) -> "LinearSVM":
        target = np.where(y == 1, 1., -1.); self.w = np.zeros(x.shape[1]); self.b = 0.; self.loss_history = []
        for step in range(self.steps):
            margin = target * (x @ self.w + self.b); active = margin < 1
            dw = self.w - self.c * (x[active].T @ target[active]) / len(x); db = -self.c * target[active].sum() / len(x)
            self.w -= self.learning_rate * dw; self.b -= self.learning_rate * db
            if step % 50 == 0: self.loss_history.append(.5 * self.w @ self.w + self.c * np.maximum(0, 1 - margin).mean())
        return self
    def decision_function(self, x: np.ndarray) -> np.ndarray: return x @ self.w + self.b
    def predict(self, x: np.ndarray) -> np.ndarray: return (self.decision_function(x) >= 0).astype(int)


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    x, y = make_blobs(n_samples=280, centers=[(-1.5, -1), (1.4, 1)], cluster_std=[.9, .85], random_state=12)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.3, stratify=y, random_state=4)
    scaler = StandardScaler().fit(x_train); train, test = scaler.transform(x_train), scaler.transform(x_test)
    scratch = LinearSVM(c=2.).fit(train, y_train); framework = SVC(kernel="linear", C=2.).fit(train, y_train)
    print(f"scratch_test_accuracy={accuracy_score(y_test, scratch.predict(test)):.3f}")
    print(f"sklearn_test_accuracy={accuracy_score(y_test, framework.predict(test)):.3f}")

    # Hinge loss.
    margin = np.linspace(-2, 3, 400); fig, ax = plt.subplots(figsize=(7.2, 4.3)); ax.plot(margin, np.maximum(0, 1-margin), lw=2.7, color="#d94b4b", label="hinge loss"); ax.axvline(0, ls="--", color="#777", label="wrong/right boundary"); ax.axvline(1, ls=":", color="#3a9d5d", label="unit margin")
    ax.set(xlabel="signed margin y f(x)", ylabel="loss", title="Correct is not enough: SVM also asks for margin"); ax.legend(frameon=False); ax.grid(alpha=.15); ax.spines[["top", "right"]].set_visible(False); fig.tight_layout(); fig.savefig(OUTPUT / "hinge-loss.png", dpi=180, bbox_inches="tight", facecolor="white"); plt.close(fig)

    # Margin and support vectors.
    xx, yy = np.meshgrid(np.linspace(-3, 3, 250), np.linspace(-3, 3, 250)); grid = np.c_[xx.ravel(), yy.ravel()]; score = framework.decision_function(grid).reshape(xx.shape)
    fig, ax = plt.subplots(figsize=(6.5, 5)); ax.contour(xx, yy, score, levels=[-1, 0, 1], colors=["#777", "#202020", "#777"], linestyles=["--", "-", "--"]); ax.scatter(train[y_train==0,0], train[y_train==0,1], color="#3478c0", edgecolor="white"); ax.scatter(train[y_train==1,0], train[y_train==1,1], color="#f28e2b", edgecolor="white"); ax.scatter(framework.support_vectors_[:,0], framework.support_vectors_[:,1], s=150, facecolors="none", edgecolors="#d94b4b", lw=2, label="support vectors")
    ax.set(title="Only points near the margin determine the boundary", xlabel="standardized feature 1", ylabel="standardized feature 2"); ax.legend(frameon=False); ax.grid(alpha=.12); ax.spines[["top","right"]].set_visible(False); fig.tight_layout(); fig.savefig(OUTPUT / "svm-margin-support-vectors.png", dpi=180, bbox_inches="tight", facecolor="white"); plt.close(fig)

    # C tradeoff.
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharex=True, sharey=True)
    for ax, c in zip(axes, [.05, 1, 100]):
        model=SVC(kernel="linear", C=c).fit(train,y_train); z=model.decision_function(grid).reshape(xx.shape); ax.contour(xx,yy,z,levels=[-1,0,1],colors=["#777","#202020","#777"],linestyles=["--","-","--"]); ax.scatter(train[:,0],train[:,1],c=y_train,cmap="coolwarm",edgecolor="white",alpha=.75); ax.set_title(f"C={c}"); ax.spines[["top","right"]].set_visible(False)
    fig.suptitle("C trades a wider margin against training violations"); fig.tight_layout(); fig.savefig(OUTPUT / "svm-c-tradeoff.png", dpi=180,bbox_inches="tight",facecolor="white"); plt.close(fig)

    # Kernel boundary.
    moons, labels=make_moons(n_samples=280,noise=.2,random_state=17); moon_scaler=StandardScaler().fit(moons); moons=moon_scaler.transform(moons); kernel=SVC(kernel="rbf",C=3,gamma=1).fit(moons,labels); xx2,yy2=np.meshgrid(np.linspace(-2.5,2.5,260),np.linspace(-2.5,2.5,260)); grid2=np.c_[xx2.ravel(),yy2.ravel()]; z=kernel.decision_function(grid2).reshape(xx2.shape)
    fig,ax=plt.subplots(figsize=(6.5,5)); ax.contourf(xx2,yy2,z,levels=15,cmap="RdBu_r",alpha=.25); ax.contour(xx2,yy2,z,levels=[0],colors="#222",linewidths=2); ax.scatter(moons[:,0],moons[:,1],c=labels,cmap="coolwarm",edgecolor="white"); ax.scatter(kernel.support_vectors_[:,0],kernel.support_vectors_[:,1],s=90,facecolors="none",edgecolors="#222",label="support vectors"); ax.set(title="RBF kernel: a linear separator in an implicit feature space",xlabel="feature 1",ylabel="feature 2"); ax.legend(frameon=False); ax.spines[["top","right"]].set_visible(False); fig.tight_layout(); fig.savefig(OUTPUT / "kernel-svm-boundary.png",dpi=180,bbox_inches="tight",facecolor="white"); plt.close(fig)
    assert accuracy_score(y_test, scratch.predict(test)) > .9


if __name__ == "__main__": main()
