# Chapter 9: Learning Theory and Model Selection

[← Chapter 8](08-trees-and-ensembles.md) | [中文](../zh-CN/09-learning-theory-and-model-selection.md) | [Notation](appendix-a-notation.md)

> **Prerequisites:** train/validation/test splits, underfitting and overfitting, regularization, and basic probability.

## From Chapter 8 to this chapter

Polynomial degree, regularization strength, neighbor count, and tree depth all control capacity. The setting with the
lowest training error is rarely the most trustworthy on new data. This chapter develops a common language for every
model: expected risk, bias and variance, cross-validation, learning curves, and a first generalization bound.

```mermaid
flowchart LR
  A[Candidate models] --> B[Fit on training data]
  B --> C[Cross-validation]
  C --> D[Choose capacity]
  D --> E[Refit]
  E --> F[Use test set once]
  F --> G[Monitor deployment shift]
```

## 9.1 Why training error is not enough

For $(x,y)\sim\mathcal D$, the desired expected risk and its observable empirical estimate are

$$R(f)=\mathbb E_{\mathcal D}[\ell(f(x),y)],\qquad
\hat R_m(f)=\frac1m\sum_{i=1}^{m}\ell(f(x^{(i)}),y^{(i)}).$$

A flexible model can memorize the sample, making $\hat R_m$ tiny while $R$ remains large. Scoring a learner on the
same examples used to fit it is like giving a student the exam in advance.

![Capacity moves from underfitting to a useful fit and then overfitting](../assets/capacity-three-regimes.png)

All panels use the same noisy rent data. The line on the left misses the curved trend, the middle curve follows the
stable pattern, and the high-degree curve oscillates in sparse regions while closely following training points. The
figure illustrates three regimes; it does not establish that degree five is universally best.

## 9.2 Bias, variance, and irreducible noise

Suppose $y=f^{\ast}(x)+\varepsilon$, with zero-mean noise of variance $\sigma^2$. Under squared loss,

$$\mathbb E[(y-\hat f(x))^2]=\sigma^2+
(\mathbb E\hat f(x)-f^{\ast}(x))^2+
\mathbb E[(\hat f(x)-\mathbb E\hat f(x))^2].$$

The terms are irreducible noise, squared bias, and variance. Add and subtract $\mathbb E\hat f(x)$, expand the
square, and note that the cross terms have expectation zero. High bias means systematic error across training sets;
high variance means sensitivity to which examples were sampled. The neat equality is specific to squared regression
loss, although the diagnostic ideas remain useful more broadly.

## 9.3 Cross-validation without leakage

In $K$-fold cross-validation, each fold is held out once:

$$CV_K(\lambda)=\frac1K\sum_{k=1}^{K}L_k(\lambda).$$

![Five-fold cross-validation rotates the validation block](../assets/cross-validation-rotation.png)

Each row is one fit. The orange block is validation data and the blue blocks are training data, so every observation
is evaluated once. Scaling, imputation, feature selection, and PCA must be fitted inside the training portion of each
row. Fitting them once on all data leaks information from validation examples.

Use stratified folds for imbalanced labels, group folds when one entity has repeated records, and forward-in-time
validation for temporal prediction. After repeated model selection, use an independent test set or nested CV for the
final estimate.

## 9.4 Validation curves and learning curves

![Training and validation MSE as polynomial degree increases](../assets/capacity-validation-curve.png)

Training error generally decreases with degree because the larger family contains the smaller one. Validation error
first improves and then rises: added capacity initially fixes underfitting, then begins to fit sample-specific noise.
The dashed line marks the degree selected in this experiment only.

![Training and validation error as the training set grows](../assets/learning-curve-data-size.png)

With few examples, the model can fit training data while failing on validation data. As sample size grows, the gap
narrows and validation error improves with diminishing returns. Two high, converged curves suggest bias; a persistent
large gap suggests variance. Curves being close is not sufficient—they may meet at a poor error level.

## 9.5 Regularization as capacity control

Structural risk minimization adds a penalty:

$$\min_\theta\hat R_m(\theta)+\lambda\Omega(\theta).$$

L2 shrinks weights smoothly and relates to a Gaussian-prior MAP estimate; L1 often produces sparse solutions and
relates to a Laplace prior. Tree-depth limits, early stopping, and valid data augmentation also restrict effective
capacity. The strength is a hyperparameter chosen on validation data, not by minimizing training loss.

## 9.6 A first generalization bound

For a finite hypothesis set $\mathcal H$ and loss in $[0,1]$, Hoeffding's inequality plus a union bound gives

$$P\left(\max_{h\in\mathcal H}|R(h)-\hat R(h)|>\epsilon\right)
\le2|\mathcal H|e^{-2m\epsilon^2}.$$

Thus, with probability at least $1-\delta$, every $h$ satisfies

$$|R(h)-\hat R(h)|\le
\sqrt{\frac{\log(2|\mathcal H|/\delta)}{2m}}.$$

More data tightens the bound; more candidates or more confidence loosens it. This is a conservative upper bound, not
an exact prediction of test error. Infinite classes require tools such as VC dimension or Rademacher complexity.

## 9.7 Reproduce the experiment

```bash
python code/chapter09/learning_theory.py
```

The script implements polynomial candidates and five-fold selection from scratch while preserving a separate test
set. With the fixed seed it chooses degree 5, obtains CV MSE about 0.065, and test MSE about 0.049. Repeatedly changing
the model after seeing that test result would invalidate it.

## 9.8 Practical selection checklist

1. Define the deployment-time features, sample unit, metric, and cost of errors.
2. Isolate the test set before exploration; respect time and entity groups.
3. Establish a simple baseline and a bounded candidate space.
4. Fit all preprocessing inside each fold and report variation, time, and model size.
5. Prefer the simpler, cheaper, more stable model when scores are practically tied.
6. Refit after selection, evaluate the test set once, then monitor drift in production.

Search cost is roughly $O(HKC)$ for $H$ configurations, $K$ folds, and training cost $C$. Common failures are
preprocessing leakage, unequal search budgets, cherry-picked seeds, and a metric unrelated to deployment cost.

## Worked reasoning checkpoint

Keep test data sealed while all preprocessing and hyperparameter search happen inside the development set. In a
five-example leave-one-out comparison, average loss alone is not enough: inspect fold variation and failed examples.
Repeated K-fold measures split sensitivity; Group K-fold protects entities; walk-forward validation protects time;
nested CV estimates the entire model-selection procedure.

Selection itself can overfit. Trying many equally capable candidates tends to select the luckiest validation score.
Record the search budget, compare paired per-example losses with bootstrap intervals, and diagnose in this order:
data boundary, metric implementation, baseline, optimization, capacity, and distribution shift.

## Modern AI connections

- LLM training loss measures next-token fit to the training distribution, not safety or downstream reliability.
- LoRA reduces trainable parameters, but small fine-tuning sets can still overfit.
- RAG should evaluate retrieval, reranking, grounding, and generation separately.
- Recommenders require temporal splits; random splits often leak future behavior.
- Double descent and scaling laws refine the simple U-shaped story but do not remove contamination, shift, or privacy risks.

## Socratic mentor

<details><summary>Why can zero training error still be a bad model?</summary>
It may memorize examples or exploit leakage; the target is risk on new draws from the deployment distribution.
</details>

<details><summary>Why not report the same CV score used to choose among thousands of configurations?</summary>
Selection exploits random fluctuations in those folds. Use a held-out test set or an outer CV loop.
</details>

<details><summary>Why does more data not always fix high validation error?</summary>
If both training and validation errors are high and close, the model or features may have high bias.
</details>

<details><summary>Why must scaling be refitted in every fold?</summary>
Global means and variances contain information about the held-out fold and change the learning procedure.
</details>

<details><summary>Does a good IID test score guarantee production quality?</summary>
No. Deployment can introduce subgroup gaps, feedback loops, latency constraints, and distribution shift.
</details>

## Exercises

See [Chapter 9 solutions](solutions/09-solutions.md).

1. Compute the mean and sample standard deviation of five supplied fold losses.
2. Repair a pipeline that scales all data, searches on the test set, and reports the best test run.
3. Sketch idealized high-bias and high-variance learning curves and recommend a next action.
4. Derive the finite-class bound from Hoeffding and the union bound.
5. Vary noise, sample size, and maximum degree in the provided experiment.
6. Design validation for repeated patients, repeated users, and daily sales.
7. Design separate retrieval, grounding, and generation evaluations for a RAG system.

## Interview recap

- **Train/validation/test?** Fit parameters, select hyperparameters, and estimate final generalization.
- **Bias versus variance?** Systematic average error versus sensitivity to sampled training data.
- **Is larger $K$ always better?** No; it costs more and correlated folds do not guarantee lower-variance estimates.
- **When use group folds?** When related records from one entity must not cross the split.
- **Why no random split for time series?** It trains on the future and evaluates on the past.
- **What is nested CV?** Inner selection and outer generalization estimation.
- **What do learning curves diagnose?** Whether more data, more capacity, or more regularization is promising.
- **What does the bound connect?** Sample size, hypothesis complexity, confidence, and generalization gap.

## Mastery checklist

- [ ] I can distinguish expected and empirical risk.
- [ ] I can derive and interpret the squared-loss bias-variance decomposition.
- [ ] I can choose stratified, grouped, temporal, or nested validation without leakage.
- [ ] I can use validation and learning curves to propose the next experiment.
- [ ] I can derive the finite-hypothesis generalization bound.
- [ ] I can design layered evaluation for an LLM, RAG, or recommender system.

### Further reading

- [Stanford CS229: Bias/Variance Analysis](https://cs229.stanford.edu/summer2020/BiasVarianceAnalysis.pdf)
- [Stanford CS229: Learning Theory](https://cs229.stanford.edu/notes_archive/cs229-notes4.pdf)
- [scikit-learn: Cross-validation](https://scikit-learn.org/stable/modules/cross_validation.html)
- [scikit-learn: Validation and learning curves](https://scikit-learn.org/stable/modules/learning_curve.html)

[Next: Clustering, GMM, EM, and ELBO →](10-clustering-em-and-elbo.md)
