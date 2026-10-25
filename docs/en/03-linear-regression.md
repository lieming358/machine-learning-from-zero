# Chapter 3 — Linear Regression: The First Complete Algorithm Chain

[← Math](02-math-foundations.md) | [中文](../zh-CN/03-linear-regression.md) | [Roadmap](00-course-roadmap.md)

Why start here? Linear regression is simple enough to expose the whole data → assumption → loss →
solution → evaluation → diagnosis loop, yet its matrix operation remains central to neural networks,
recommenders, probes, and LoRA.

Linear regression models $y=w^Tx+b+\epsilon$ with commonly Gaussian noise. It does not claim the
world is linear; it asks whether a line is a useful, interpretable approximation over the domain.

### Learning boundary

OLS, its loss/gradient, held-out evaluation, and residual diagnosis are the required core. Polynomial features,
Ridge, and Lasso are required as evidence-driven responses to curvature and instability. Weighted and locally
weighted regression are extensions: understand why they exist without treating them as default starting points.

![Rental observations and target distribution](../assets/rent-data-distribution.png)

Read the data before fitting: inspect center, spread, shape, outliers, and boundary coverage. The plot
supports a linear baseline but neither proves causation nor licenses extrapolation beyond observed areas.

## Derivation chain

Choose $f(x)=w^Tx+b$ → assume Gaussian observation noise → maximum likelihood yields squared error →
differentiate for gradient descent or set the gradient to zero → evaluate on unseen data → inspect
residuals for broken assumptions.

With a constant column folded into $X$,

$$J(\theta)=\frac1m\|X\theta-y\|_2^2,\qquad
\nabla J=\frac2mX^T(X\theta-y).$$

Setting the gradient to zero produces the normal equations
$X^TX\theta=X^Ty$. In code, prefer `lstsq`, QR, or SVD over explicitly computing an inverse.
Gradient descent instead applies

$$\theta\leftarrow\theta-\eta\frac2mX^T(X\theta-y).$$

Finish this unregularized OLS loop before adding extensions: fit, evaluate unseen data, and inspect
residuals. New methods should answer observed failures rather than arrive as an isolated vocabulary list.

Weighted least squares minimizes $\sum_i r^{(i)}(y^{(i)}-\theta^Tx^{(i)})^2$ when examples have
different reliability. Locally weighted regression sets
$r^{(i)}=\exp(-\|x^{(i)}-x_q\|^2/(2\tau^2))$ for each query. Small bandwidth can overfit; large
bandwidth approaches a global line; per-query fitting is costly.

## Failure-driven model evolution

A curved residual pattern first motivates polynomial features: $x\mapsto[x,x^2,x^3]$. The resulting
curve is still a linear combination of learned coefficients.

![Polynomial feature expansion before the linear weighted sum](../assets/polynomial-feature-map.png)

If increased capacity creates unstable coefficients or a train–validation gap, regularization charges
for complexity. Ridge adds $\lambda\|w\|_2^2$ and smoothly shrinks coefficients; Lasso adds
$\lambda\|w\|_1$ and can create exact zeros. Their strength must be selected on validation data.

![Ridge and Lasso coefficient paths](../assets/regularization-coefficient-paths.png)

Start with OLS. Add polynomial terms for evidenced curvature; prefer Ridge for correlated features and
stable prediction; consider Lasso when a defensible sparse-feature assumption matters. Keep OLS when
it already generalizes well.

```bash
python code/chapter03/regression_family_comparison.py
```

![OLS, polynomial, Ridge, and Lasso on the same split](../assets/regression-family-comparison.png)

Run both and compare:

```bash
python code/chapter03/linear_regression.py
```

The script compares the closed form, gradient descent, and scikit-learn predictions with numerical
assertions, then demonstrates a locally weighted prediction.

The second experiment uses scikit-learn's built-in 442-example, 10-feature diabetes regression
dataset and reports a mean baseline, RMSE, MAE, $R^2$, and a paired bootstrap interval:

```bash
python code/chapter03/real_dataset_baseline.py
```

![Predictions and residuals on the real diabetes dataset](../assets/diabetes-real-data-diagnostics.png)

This is a teaching benchmark, not a clinically validated model; random splitting does not simulate
hospital or temporal shift. See the [official dataset documentation](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_diabetes.html).

Use RMSE for error in target units and $R^2$ relative to a mean baseline. A negative test $R^2$ is
possible. Curved residual structure suggests missing nonlinearity; a funnel suggests heteroscedasticity;
isolated large residuals suggest outliers or data errors.

![Fitted rental line and residuals](../assets/rent-fit-residuals.png)

Why inspect residuals? A single metric compresses all errors into one number and can hide systematic
failure. Patternless residuals support the model; structure tells us which assumption to revisit.

![Polynomial underfitting, useful fitting, and overfitting](../assets/underfit-goodfit-overfit.png)

A degree-15 curve can nearly memorize training points yet oscillate wildly between them. Greater
fitting capacity is not the same as greater generalization. See the full diagnosis and prevention
guide in [Chapter 1](01-what-is-ml.md).

## Modern-AI bridge

Neural-network linear layers are $XW+b$; linear probes test information in frozen embeddings; LoRA
uses low-rank linear updates $\Delta W=BA$; and most LLM compute is matrix multiplication. Linear
regression remains a valuable baseline against which complexity must justify latency and cost.

## Socratic check

<details><summary>Can adding a feature raise training MSE?</summary>
Not with full optimization in a nested unregularized model: its coefficient can be zero. Test error can rise.
</details>

<details><summary>Why avoid explicitly inverting $X^TX$?</summary>
It is less stable and efficient, and forming $X^TX$ squares the condition number. QR/SVD-based least
squares is safer.
</details>

## Interview recap

- Multicollinearity destabilizes coefficient interpretation; prediction may still be acceptable.
- Squared error is sensitive to outliers.
- Ridge uses $L_2$ shrinkage; Lasso uses $L_1$ and can create exact zeros.
- Classical interval inference needs more assumptions than point prediction alone.
