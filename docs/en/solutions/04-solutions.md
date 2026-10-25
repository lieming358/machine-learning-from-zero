# Chapter 4 — Hints and Reference Answers

[← Chapter 4](../04-logistic-regression.md) | [中文答案](../../zh-CN/solutions/04-solutions.md)

## 1. Manual calculation

$z=1(2)-2(1)+0.5=0.5$, so $p=\sigma(0.5)\approx0.622$ and the class is 1 at threshold 0.5.

## 2. Bernoulli, BCE, and the gradient

The likelihood is $p^y(1-p)^{1-y}$. Its negative logarithm is BCE. With
$\partial p/\partial z=p(1-p)$, the factors cancel and $\partial\ell/\partial z=p-y$.

## 3. Threshold design

Do not choose a threshold from F1 alone. State an error cost, such as $20FN+FP$, minimize it on the
validation set, and use the test set once for final evaluation. Also report the confusion matrix and
probability calibration.

## 4. XOR

One line cannot separate XOR in the original plane. An interaction feature $x_1x_2$ can make it
linearly separable in an expanded representation.

## 5. Regularization

Increasing $\lambda$ usually shrinks $\lVert w\rVert_2$ and raises training loss. Validation loss may
fall and then rise, so select regularization with validation or cross-validation rather than training
accuracy.

## 6. Linear-probe rubric

Include majority-class and random-feature baselines, leakage-safe splits, several representation
layers, sample-size curves, multiple seeds, and the limitation that decodability does not prove the
original model causally uses that information.
