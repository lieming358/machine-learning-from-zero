"""Chapter 8: decision-stump split from scratch and tree ensembles."""

from __future__ import annotations
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_moons
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree

OUTPUT = Path(__file__).resolve().parents[2] / "docs" / "assets"


def gini(y: np.ndarray) -> float:
    if len(y)==0: return 0.
    p=np.bincount(y,minlength=2)/len(y); return float(1-np.sum(p*p))


class DecisionStump:
    def fit(self,x:np.ndarray,y:np.ndarray)->"DecisionStump":
        best=(np.inf,None,None,None,None)
        for j in range(x.shape[1]):
            values=np.unique(x[:,j]); thresholds=(values[:-1]+values[1:])/2
            for threshold in thresholds:
                left=x[:,j]<=threshold; score=(left.mean()*gini(y[left])+(~left).mean()*gini(y[~left]))
                if score<best[0]:
                    lp=int(np.mean(y[left])>=.5); rp=int(np.mean(y[~left])>=.5); best=(score,j,threshold,lp,rp)
        self.impurity,self.feature,self.threshold,self.left_prediction,self.right_prediction=best; return self
    def predict(self,x:np.ndarray)->np.ndarray: return np.where(x[:,self.feature]<=self.threshold,self.left_prediction,self.right_prediction)


def boundary(axis,model,x,y,title):
    xx,yy=np.meshgrid(np.linspace(-1.8,2.8,240),np.linspace(-1.5,2,240)); grid=np.c_[xx.ravel(),yy.ravel()]; z=model.predict(grid).reshape(xx.shape); axis.contourf(xx,yy,z,levels=[-.5,.5,1.5],cmap="RdBu_r",alpha=.28); axis.scatter(x[:,0],x[:,1],c=y,cmap="coolwarm",edgecolor="white",alpha=.75); axis.set(title=title,xlabel="feature 1",ylabel="feature 2"); axis.spines[["top","right"]].set_visible(False)


def main()->None:
    OUTPUT.mkdir(parents=True,exist_ok=True); x,y=make_moons(n_samples=520,noise=.29,random_state=22); x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=.3,stratify=y,random_state=8)
    stump=DecisionStump().fit(x_train,y_train); library_stump=DecisionTreeClassifier(max_depth=1,random_state=0).fit(x_train,y_train)
    print(f"scratch_stump_accuracy={accuracy_score(y_test,stump.predict(x_test)):.3f}"); print(f"sklearn_stump_accuracy={accuracy_score(y_test,library_stump.predict(x_test)):.3f}"); print(f"best_feature={stump.feature} threshold={stump.threshold:.3f} weighted_gini={stump.impurity:.3f}")

    # Impurity intuition.
    p=np.linspace(0,1,300); safe=np.clip(p,1e-12,1-1e-12); fig,ax=plt.subplots(figsize=(7.2,4.3)); ax.plot(p,2*p*(1-p),lw=2.6,label="Gini",color="#3478c0"); entropy=-(p*np.log2(safe)+(1-p)*np.log2(1-safe)); ax.plot(p,entropy/2,lw=2.6,label="entropy / 2",color="#f28e2b"); ax.set(xlabel="positive-class proportion p",ylabel="impurity",title="A node is pure when one class dominates"); ax.legend(frameon=False); ax.grid(alpha=.15); ax.spines[["top","right"]].set_visible(False); fig.tight_layout(); fig.savefig(OUTPUT/"tree-impurity.png",dpi=180,bbox_inches="tight",facecolor="white"); plt.close(fig)

    # A small interpretable tree.
    small=DecisionTreeClassifier(max_depth=3,min_samples_leaf=12,random_state=0).fit(x_train,y_train); fig,ax=plt.subplots(figsize=(12,6)); plot_tree(small,feature_names=["feature 1","feature 2"],class_names=["class 0","class 1"],filled=True,rounded=True,ax=ax,proportion=True); fig.tight_layout(); fig.savefig(OUTPUT/"decision-tree-rules.png",dpi=180,bbox_inches="tight",facecolor="white"); plt.close(fig)

    # Depth: underfit to overfit.
    depths=np.arange(1,16); train_scores=[]; test_scores=[]
    for depth in depths:
        model=DecisionTreeClassifier(max_depth=int(depth),random_state=0).fit(x_train,y_train); train_scores.append(model.score(x_train,y_train)); test_scores.append(model.score(x_test,y_test))
    fig,ax=plt.subplots(figsize=(7.3,4.4)); ax.plot(depths,train_scores,marker="o",label="training",color="#3478c0"); ax.plot(depths,test_scores,marker="o",label="test (illustration only)",color="#f28e2b"); ax.set(xlabel="maximum depth",ylabel="accuracy",title="Deeper trees keep fitting; generalization may stop improving"); ax.legend(frameon=False); ax.grid(alpha=.15); ax.spines[["top","right"]].set_visible(False); fig.tight_layout(); fig.savefig(OUTPUT/"tree-depth-overfitting.png",dpi=180,bbox_inches="tight",facecolor="white"); plt.close(fig)

    tree=DecisionTreeClassifier(max_depth=None,min_samples_leaf=1,random_state=0).fit(x_train,y_train); forest=RandomForestClassifier(n_estimators=160,min_samples_leaf=2,max_features="sqrt",random_state=0,n_jobs=1).fit(x_train,y_train); boost=GradientBoostingClassifier(n_estimators=90,max_depth=2,learning_rate=.06,random_state=0).fit(x_train,y_train)
    fig,axes=plt.subplots(1,3,figsize=(13,4),sharex=True,sharey=True)
    for ax,(name,model) in zip(axes,[("one deep tree",tree),("random forest: average",forest),("gradient boosting: sequential corrections",boost)]): boundary(ax,model,x_train,y_train,name)
    fig.suptitle("Bagging reduces variance; boosting builds corrections sequentially"); fig.tight_layout(); fig.savefig(OUTPUT/"tree-ensemble-boundaries.png",dpi=180,bbox_inches="tight",facecolor="white"); plt.close(fig)
    for name,model in [("tree",tree),("random_forest",forest),("gradient_boosting",boost)]: print(f"{name}_test_accuracy={model.score(x_test,y_test):.3f}")
    assert abs(stump.predict(x_test).mean()-library_stump.predict(x_test).mean())<.1


if __name__=="__main__": main()
