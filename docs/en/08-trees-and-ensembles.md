# Chapter 8 — Trees and Ensembles

[← Chapter 7](07-support-vector-machines.md) | [中文](../zh-CN/08-trees-and-ensembles.md)

Loan or click-through rules often sound like questions: “any overdue payment?”, then “debt ratio above
45%?”, then “new user?”. Trees are useful when such thresholds and feature interactions are more natural
than one smooth global boundary. The learning problem is deciding which question should come first.

A candidate split minimizes weighted child impurity. Binary Gini is $2p(1-p)$; entropy is
$-p\log_2p-(1-p)\log_2(1-p)$.

![Gini and entropy](../assets/tree-impurity.png)

The horizontal axis is positive-class proportion. Both measures are zero at pure nodes ($p=0$ or $1$)
and largest near an even mixture. They measure class mixing, not model accuracy.

A stump searches every feature and midpoint threshold. A full tree repeats recursively and predicts
class proportions or regression means at leaves.

![A depth-three rule tree](../assets/decision-tree-rules.png)

Start at the root, follow the left branch when a condition is true and the right branch otherwise, and
stop at a leaf. `samples` counts training rows reaching a node, `value` counts classes, and darker color
means a purer node rather than guaranteed real-world confidence.

Deep trees can memorize training data and are unstable to sampling perturbations.

![Depth and overfitting](../assets/tree-depth-overfitting.png)

Depth and complexity increase to the right. Training accuracy keeps rising, while held-out performance
peaks and may fall; a widening gap signals overfitting. Choose depth with validation data, not repeated
test-set inspection.

Random forests bootstrap data and randomize feature candidates, then average many decorrelated trees.
Gradient boosting builds a serial additive model
$F_M=F_0+\eta\sum_mh_m$ whose next tree fits residuals or negative gradients.

For a concrete forest vote, first sample training-row indices **with replacement** for each tree; repeated and omitted
rows make the fitted trees different. Let three trees then predict four queries as $[0,1,1,0]$, $[0,1,0,0]$, and
$[1,1,1,0]$. Positive vote counts are $[1,3,2,0]$, so majority voting returns $[0,1,1,0]$. Averaging helps only
when individual trees have useful signal and their errors are not identical.

For one squared-error boosting round, take labels $[2,2,8,10]$. The initial mean prediction is $5.5$, leaving
residuals $[-3.5,-3.5,2.5,4.5]$. A stump separating the first two samples from the last two predicts residuals
$-3.5$ and $3.5$. With learning rate $0.5$, predictions become $[3.75,3.75,7.25,7.25]$ and MSE falls from
$12.75$ to about $3.56$. The next tree fits the new remaining error, not the original labels from scratch.

![Tree and ensemble boundaries](../assets/tree-ensemble-boundaries.png)

All panels use the same points; background color is the predicted class. A deep tree creates fragmented
axis-aligned regions, a forest averages randomized trees, and boosting adds shallow corrections. This is
an illustration on one dataset, not proof that one ensemble always wins.

```bash
python code/chapter08/trees_and_ensembles.py
python code/chapter08/course_lab.py
```

The course lab is the beginner-readable entry: it computes impurity, a three-tree vote, and one boosting correction.
The figure script compares the scratch stump and framework ensembles.
Impurity importance favors high-cardinality features and is not causal. Audit leakage, correlated
features, validation design, latency, and model size.

## Modern AI connections

Recommendation and advertising systems can combine embedding retrieval with GBDT ranking over counts,
price, time, and context. Trees provide auditable routing rules in risk systems, but SHAP explains model
behavior rather than causality. In LLM systems, a small tree can route requests among models, retrieval,
and caches using expected quality, latency, and cost; it is a system controller, not the Transformer's
token model. XGBoost, LightGBM, and CatBoost share additive-tree principles but differ in optimization
and systems design.

## Failure diagnosis

Near-perfect training with weak validation suggests deep leaves, leakage, or both. Highly correlated
forest trees reduce averaging gains. Boosting that worsens late in training needs early stopping or
smaller learning rate/depth. Unique identifiers can dominate importance without helping new entities.
Measure feature-computation cost, tree count/depth, model bytes, throughput, and p95 latency.

## Socratic mentor

<details><summary>Why does a greedy best split not guarantee the globally best tree?</summary>
A locally attractive root can block better later combinations; exhaustive tree search is combinatorial.
</details>

<details><summary>Why randomize both samples and candidate features in a forest?</summary>
Both sources diversify trees and reduce error correlation, making averaging more effective.
</details>

<details><summary>Why does boosting fit negative gradients?</summary>
They give the function-space direction that most rapidly reduces the current loss; for squared loss this
specializes to ordinary residuals.
</details>

## Exercises

Hand-compute two candidate splits; implement a two-level tree; plot depth overfitting; ablate bootstrap,
feature subsampling, learning rate, and tree count; audit ID/correlated-feature importance; design an LLM
router with quality and latency costs. [Hints and answers](solutions/08-solutions.md).

## Interview recap

- Trees greedily maximize impurity reduction and usually do not need scaling.
- Deep trees have high variance; leaf size, depth, and pruning control capacity.
- Bagging averages parallel estimators; boosting serially follows loss gradients.
- Random forests additionally subsample candidate features; OOB rows support internal evaluation.
- Feature importance is not causality; RF versus GBDT is an empirical quality–cost choice.

## Mastery checklist

- [ ] I can hand-compute Gini, entropy, weighted impurity, and information gain.
- [ ] I can implement split search and explain recursive stopping.
- [ ] I diagnose depth overfitting with validation rather than the test set.
- [ ] I can compare bagging, forests, and boosting through bias, variance, and dependencies.
- [ ] I can audit leakage, importance, extrapolation, latency, and modern routing uses.

### Further reading

- [scikit-learn Decision Trees guide](https://scikit-learn.org/stable/modules/tree.html)
- [scikit-learn Ensemble Methods guide](https://scikit-learn.org/stable/modules/ensemble.html)
- [Breiman, Random Forests](https://doi.org/10.1023/A:1010933404324)
- [XGBoost: A Scalable Tree Boosting System](https://arxiv.org/abs/1603.02754)

[Next: Learning Theory and Model Selection →](09-learning-theory-and-model-selection.md)
