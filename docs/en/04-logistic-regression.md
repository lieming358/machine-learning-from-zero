# Chapter 4 — Logistic Regression, Perceptron, and GLMs

[← Chapter 3](03-linear-regression.md) | [中文](../zh-CN/04-logistic-regression.md) | [Notation](appendix-a-notation.md)

Logistic regression is usually a **classifier** despite its name. Linear regression is unbounded and
its squared-error model does not match binary outcomes. Logistic regression maps a linear score to a
Bernoulli probability.

This continues Chapter 3 rather than starting over. “What is the rent?” has a continuous label;
“Will the rent exceed the budget?” uses the same housing features but a binary label. The linear score
$w^Tx+b$ remains, while the output distribution and loss must change together.

![Linear versus logistic models for binary outcomes](../assets/linear-vs-logistic-classification.png)

## Sigmoid and log-odds

$$z=w^Tx+b,\qquad p(y=1\mid x)=\sigma(z)=\frac1{1+e^{-z}}.$$

![The sigmoid function](../assets/sigmoid-function.png)

The inverse relationship is $\log(p/(1-p))=w^Tx+b$. Increasing feature $j$ by one multiplies the odds
by $e^{w_j}$, holding other features fixed; it does not add a constant amount to probability.

## Maximum likelihood to cross-entropy

For $p^{(i)}=\sigma(w^Tx^{(i)}+b)$ and $y^{(i)}\in\{0,1\}$,

$$p(y^{(i)}\mid x^{(i)})=(p^{(i)})^{y^{(i)}}(1-p^{(i)})^{1-y^{(i)}}.$$

Negative average log-likelihood is binary cross-entropy:

$$J=-\frac1m\sum_i[y^{(i)}\log p^{(i)}+(1-y^{(i)})\log(1-p^{(i)})].$$

The chain rule simplifies to

$$\frac{\partial\ell^{(i)}}{\partial z^{(i)}}=p^{(i)}-y^{(i)},\quad
\nabla_wJ=\frac1mX^T(p-y),\quad
\frac{\partial J}{\partial b}=\frac1m\sum_i(p^{(i)}-y^{(i)}).$$

Compute stable BCE from logits with `logaddexp(0, z) - y*z`, not `log(sigmoid(z))`.

For one numerical example, let $z=0.5$ and $y=1$. Then $p=\sigma(0.5)\approx0.622$, the loss is
$-\log(0.622)\approx0.474$, and the logit gradient is $p-y\approx-0.378$. The negative gradient tells gradient
descent to increase the logit for this positive sample. This connects score, probability, loss, and update in one row.

## Boundary, threshold, and metrics

Chapter 1 introduces the [confusion matrix, ROC/PR curves, probability metrics, and regression metrics](01-what-is-ml.md#evaluation-metrics-what-does-good-mean).
Here they are applied specifically to logistic regression: changing the probability threshold changes
TP, FP, FN, and TN without retraining the model.

![A two-feature logistic decision boundary](../assets/logistic-decision-boundary.png)

At threshold $0.5$, the boundary is $w^Tx+b=0$. Thresholds convert learned probabilities into actions;
they do not retrain the model.

![Precision and recall change with threshold](../assets/classification-threshold-tradeoff.png)

Accuracy is insufficient for imbalanced data. Choose precision, recall, PR-AUC, ROC-AUC, log loss,
and calibration based on error costs. A probability of 0.8 is meaningful only if calibration and
deployment distribution support that interpretation.

## Implementation

```bash
python code/chapter04/generate_figures.py
python code/chapter04/logistic_regression.py
```

The NumPy model uses a train-only scaler, stable loss, finite-difference gradient check, and a
scikit-learn probability comparison. Expected gradient error is near $10^{-11}$ and probability gaps
are a few thousandths.

![Training cross-entropy decreases and approaches a plateau](../assets/logistic-training-loss.png)

A falling training loss verifies optimization, not generalization. Oscillation or `nan` values suggest
checking the learning rate, feature scale, and numerical stability; test performance must be measured
separately.

## Perceptron and GLMs

The perceptron predicts $\mathrm{sign}(w^Tx+b)$ and updates a misclassified example by
$w\leftarrow w+\eta y^{(i)}x^{(i)}$. It gives no calibrated probability and may cycle on inseparable
data. SVM later adds a margin objective.

A generalized linear model combines an exponential-family outcome, a linear predictor, and a link:
Gaussian + identity gives linear regression; Bernoulli + logit gives logistic regression; Poisson +
log gives count regression. Multiclass logistic regression replaces sigmoid with softmax.

## Modern-AI bridge

LLMs apply softmax to token logits and train with cross-entropy; CTR models estimate click probability;
preference models often use a logistic comparison; linear probes classify frozen embeddings; lightweight
safety classifiers can gate larger models.

## Mastery check

- [ ] Derive BCE from Bernoulli likelihood and its $p-y$ logit gradient.
- [ ] Distinguish probability, threshold, class, and boundary.
- [ ] Implement stable NumPy training and compare with scikit-learn.
- [ ] Explain logistic regression versus linear regression, perceptron, SVM, and softmax.

Exercises and feedback: [Chapter 4 hints and reference answers](solutions/04-solutions.md).

[Continue to Chapter 5: model how each class generates its data →](05-generative-classifiers.md)
