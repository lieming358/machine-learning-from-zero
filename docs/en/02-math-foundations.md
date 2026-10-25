# Chapter 2 — Mathematical Foundations

[← Chapter 1](01-what-is-ml.md) | [中文](../zh-CN/02-math-foundations.md) | [Chapter 3 →](03-linear-regression.md)

The goal is not to finish several mathematics textbooks first. It is to read formulas, derive model
updates, and debug tensor shapes.

Linear algebra represents batched data and transformations; calculus explains local change;
probability models uncertainty and justifies losses; statistics separates signal from sampling noise;
optimization finds usable parameters. Without them, tensor, convergence, calibration, and evaluation
failures look like unexplained framework behavior.

## The chain

**Represent data with vectors → express models as matrix operations → use probability to justify an
objective → use calculus to obtain directions → optimize iteratively → use statistics to reason about
generalization.**

### Learning boundary

You must be able to track vector/matrix shapes, take a simple derivative, apply the chain rule, read conditional
probability, distinguish MLE from MAP, and execute a gradient update. Rank, SVD, Jacobians, Hessians, and hypothesis
tests are first-pass maps here; later chapters make each concrete. Do not postpone all ML until every advanced term is
mastered.

For $m$ examples with $d$ features, $X\in\mathbb R^{m\times d}$ and $w\in\mathbb R^d$ give
$\hat y=Xw+b\mathbf1$. Shape checking turns $(d\times m)(m\times1)$ into a $d$-vector gradient and
catches many implementation errors.

![Rows and columns of a dataset matrix](../assets/dataset-matrix-anatomy.png)

For one home, $x=[72,18,1]^T$ and $w=[38,-6,120]^T$ produce
$w^Tx=38(72)-6(18)+120(1)=2748$. Each feature contributes “value × weight.” Matrix multiplication
applies that same calculation to a whole batch.

Normalized dot products are cosine similarity. The same operation appears in two-tower recommenders,
RAG retrieval, and query-key attention. Norm penalties control capacity: $L_2$ shrinks weights while
$L_1$ often yields sparsity.

![Vector length, direction, angle, and dot product](../assets/vector-dot-product-geometry.png)

SVD, $X=U\Sigma V^T$, orders matrix directions by singular value. Keeping leading directions creates
a low-rank approximation—the idea behind PCA, matrix-factorization recommenders, compression, and
LoRA. Broadcasting must still obey shape rules; a legal broadcast can remain semantically wrong.

## Calculus and gradient descent

For $J(w,b)=m^{-1}\sum_i(wx^{(i)}+b-y^{(i)})^2$, let
$e^{(i)}=wx^{(i)}+b-y^{(i)}$. The chain rule gives

$$\frac{\partial J}{\partial w}=\frac2m\sum_i e^{(i)}x^{(i)},\qquad
\frac{\partial J}{\partial b}=\frac2m\sum_i e^{(i)}.$$

In vector form, $e=Xw+b\mathbf1-y$ and $\nabla_wJ=2X^Te/m$. Gradient descent updates
$w\leftarrow w-\eta\nabla_wJ$. Backpropagation is repeated chain rule through a computation graph.

For a one-example loss $J(w)=(2w-5)^2$, $J'(1)=-12$: increasing $w$ locally decreases the loss.
At $w=2.5$, prediction equals 5 and the derivative is zero. A derivative is a local slope, not a
single permanent property of the function.

![Gradient updates on a loss landscape](../assets/gradient-descent-landscape.png)

The approximation $J(w+\Delta)\approx J(w)+\nabla J^T\Delta$ explains why a sufficiently small
negative-gradient step descends. Jacobians organize vector-to-vector derivatives; Hessians describe
curvature. Full Hessians are too costly for large models, motivating diagonal, low-rank, or statistical
approximations.

## Probability explains the loss

Bayes' rule is $p(\theta\mid D)=p(D\mid\theta)p(\theta)/p(D)$. If regression noise is independent
Gaussian, maximizing log-likelihood is equivalent to minimizing squared error. Bernoulli and
categorical likelihoods similarly produce binary and multiclass cross-entropy. An autoregressive LLM
factorizes sequence probability as $\prod_t p(x_t\mid x_{<t})$.

For intuition, a disease with 1% prevalence, 90% sensitivity, and 5% false-positive rate yields only
$90/(90+495)\approx15.4\%$ true disease among positive tests in 10,000 people. Base rates matter.

For a tiny MLE–MAP comparison, observe 8 heads in 10 coin tosses. Bernoulli MLE gives $\hat p=8/10=0.8$.
With a $\mathrm{Beta}(2,2)$ prior, the MAP estimate is
$(8+2-1)/(10+2+2-2)=9/12=0.75$. The prior pulls a small-data estimate toward 0.5; as data grow, its relative effect
shrinks. This is the same logic by which regularization can stabilize model parameters.

Softmax converts logits into categorical probabilities. Paired with cross-entropy it gives the useful
gradient $\partial L/\partial z=p-y$. Stable implementations subtract the maximum logit before
exponentiation. Cross-entropy differs from entropy by a KL divergence; distillation, VAEs, and RLHF
reuse this relationship.

Expectation describes average behavior, variance spread, and covariance linear co-variation. Zero
covariance does not generally imply independence. PCA uses covariance structure; normalization layers
use activation statistics.

![Equal means with different variances](../assets/mean-and-variance.png)

## Optimization and numerical checks

Batch GD is stable but costly; SGD is noisy; mini-batch GD maps efficiently to accelerators. Feature
standardization often improves conditioning. Momentum accumulates velocity; Adam rescales with moving
first and second moments. A central finite difference checks an analytic gradient:

$$\frac{\partial J}{\partial w_j}\approx
\frac{J(w_j+\epsilon)-J(w_j-\epsilon)}{2\epsilon}.$$

Run the complete experiment:

```bash
python code/chapter02/gradient_descent_demo.py
```

The script generates noisy data, splits it before fitting preprocessing, trains linear regression,
reports test RMSE, and checks gradients. Computing normalization statistics on the test set would leak
test information.

## Systems bridge

Equivalent algebra can have different runtime costs. LLM memory includes parameters, activations, and
optimizer state; naive attention materializes a $T\times T$ score matrix; quantization trades precision
for memory and bandwidth; larger batches improve throughput but consume memory and may hurt latency.

## Socratic check

<details><summary>Why subtract the gradient?</summary>
The gradient points toward steepest local ascent; its negative is the steepest first-order descent direction.
</details>

<details><summary>What assumption hides behind MSE?</summary>
Under maximum likelihood: independent, constant-variance Gaussian noise. Outliers can dominate because
errors are squared.
</details>

<details><summary>Why take logs of likelihoods?</summary>
Log is strictly increasing, preserves the optimum, turns products into sums, and improves numerical stability.
</details>

## Interview recap

- Rank is the number of linearly independent rows/columns.
- MLE maximizes likelihood; MAP includes a prior and often corresponds to regularization.
- Zero covariance does not generally imply independence.
- A convex objective has no suboptimal local minimum.
- Stable softmax subtracts the maximum logit before exponentiation.

[Next: derive linear regression end to end →](03-linear-regression.md)
