# Chapter 6 — Neighbors and Kernels

[← Chapter 5](05-generative-classifiers.md) | [中文](../zh-CN/06-neighbors-and-kernels.md) | [Chapter 7 →](07-support-vector-machines.md)

Suppose one student wants to estimate the rent of an apartment. A single city-wide formula may miss
neighborhood effects, while nearby apartments with similar size and age are often informative. kNN is
needed when this **local similarity** assumption is more natural than one global formula: it stores past
examples and lets similar examples vote or average at prediction time.

For one query, the procedure is: measure its distance to every training point, sort the distances,
keep the nearest $k$, inspect their labels, and vote for classification or average for regression.

![One kNN prediction, step by step](../assets/knn-one-prediction.png)

Read the red star as the new query, the colored dots as labeled history, the black circles as the seven
selected neighbors, and the dashed segments as measured distances. The title reports the vote. The
background is deliberately absent: this figure explains one prediction before showing a full boundary.

$$d(x,z)=\|x-z\|_2,\qquad \hat y=\mathrm{mode}\{y^{(i)}:i\in\mathcal N_k(x)\}.$$

![k controls boundary complexity](../assets/knn-k-boundaries.png)

All panels reuse the same dots. Background color is the predicted class and the black curve is the
boundary. At $k=1$ tiny islands follow individual samples; an intermediate $k$ preserves the main shape;
a very large $k$ washes out local structure. Select $k$ on validation data, not the test set.

![Validation selection of k](../assets/knn-k-validation.png)

The horizontal axis lists candidate $k$ values and the vertical axis is accuracy. Training accuracy is
not the selection rule: choose the validation peak, then use the test set only for final evaluation.

Distance depends on units and representation. Fit scaling only on training data. Cosine is common for
embeddings; Euclidean, Manhattan, and Mahalanobis distances encode different notions of similarity.

![Feature scaling changes neighbors](../assets/distance-scaling-effect.png)

The bars show how much each feature contributes to squared distance before and after scaling. A feature
measured in large numerical units can dominate even when it is not more important.

Brute-force prediction costs $O(qmd)$ for $q$ queries, $m$ stored examples, and $d$ features. KD trees
help mainly in lower dimensions; large systems use ANN, quantization, and sharding.

The RBF kernel turns distance into soft similarity:

$$K(x,z)=\exp(-\gamma\|x-z\|^2).$$

![RBF similarity widths](../assets/rbf-similarity.png)

Distance increases from left to right and similarity is on the vertical axis. Larger $\gamma$ produces a
narrower neighborhood; smaller $\gamma$ lets a point influence farther locations.

The similarity can directly produce a soft local regression estimate:

$$\hat y(x)=\frac{\sum_iK(x,x^{(i)})y^{(i)}}{\sum_iK(x,x^{(i)})}.$$

For training inputs $[0,1,3]$, labels $[0,2,6]$, query $1.5$, and $\gamma=1$, squared distances are
$[2.25,0.25,2.25]$ and weights are approximately $[0.105,0.779,0.105]$. Normalizing the weights gives
$\hat y\approx2.21$. Small $\gamma$ can oversmooth; large $\gamma$ can copy one nearby noisy point, so select it on
validation data.

A kernel can stand for an inner product $K(x,z)=\phi(x)^T\phi(z)$ without explicitly building
$\phi$. A valid Gram matrix is symmetric positive semidefinite. The next chapter inserts kernels into
the SVM dual.

```bash
python code/chapter06/neighbors_and_kernels.py
python code/chapter06/course_lab.py
```

The figure script compares NumPy kNN with scikit-learn. The readable course lab prints selected neighbors, normalized
kernel weights, the prediction, and validation MSE for candidate $\gamma$ values.

## Modern AI connections

RAG retrieval follows the embedding → ANN → top-k → rerank → LLM skeleton. A nearby vector is not
automatically trustworthy evidence: audit corpus coverage, chunking, freshness, permissions, citations,
and whether the generator used the retrieved text. Recommenders similarly retrieve a few hundred items
from millions before ranking them. Contrastive and metric learning shape the embedding space in which
neighbor search operates. Kernels and neural networks both change representations, but a Transformer is
not thereby a kernel SVM.

## Failure diagnosis

Unscaled features can dominate distance; tiny $k$ follows noise; large $k$ erases local structure;
irrelevant high-dimensional features cause distance concentration; brute-force search can meet offline
quality yet fail latency. Validate representation, $k$, Recall@k, p95 latency, memory, and sliced errors.

## Socratic mentor

<details><summary>How can kNN overfit if it performs almost no optimization?</summary>
Model complexity belongs to the prediction rule, not training time. With $k=1$, every noisy sample can
create its own local island.
</details>

<details><summary>Is an RBF similarity of 0.9 a 90% same-class probability?</summary>
No. It is a distance-based similarity fixed by $\gamma$, not a calibrated class probability.
</details>

<details><summary>Why accept approximate neighbors in production?</summary>
ANN trades a measured amount of retrieval recall for large latency and memory gains. The trade is valid
only after end-to-end quality and cost evaluation.
</details>

## Exercises

Hand-compute one vote; plot boundaries and validation scores across $k$; add noise dimensions; derive
$\gamma=1/(2\sigma^2)$; compare cosine, Euclidean distance, and dot product; classify RAG failures.
[Hints and answers](solutions/06-solutions.md).

## Interview recap

- Small $k$: low bias, high variance; large $k$: smoother, higher bias.
- Scaling matters because numerical units change Euclidean neighbors.
- Brute-force query time is about $O(md)$ and storage is $O(md)$.
- A valid kernel produces symmetric positive-semidefinite Gram matrices.
- ANN selection is a recall–latency–memory decision, not an algorithm-name contest.

## Mastery checklist

- [ ] I can hand-compute classification and regression predictions.
- [ ] I can diagnose scaling, dimensionality, and representation failures.
- [ ] I choose $k$ on validation data and explain its bias–variance effect.
- [ ] I can explain RBF width, the kernel trick, and Gram matrices.
- [ ] I can map neighbors to recommendation/RAG without ignoring end-to-end failures.

### Further reading

- [scikit-learn Nearest Neighbors guide](https://scikit-learn.org/stable/modules/neighbors.html)
- [Faiss official repository](https://github.com/facebookresearch/faiss)
- [Retrieval-Augmented Generation](https://arxiv.org/abs/2005.11401)
- [Stanford CS229 materials](https://cs229.stanford.edu/materials.html-withcomments)
