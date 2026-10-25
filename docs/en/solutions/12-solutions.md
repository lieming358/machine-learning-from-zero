# Chapter 12 Solutions

[Back to Chapter 12](../12-neural-networks-and-backprop.md) | [中文](../../zh-CN/solutions/12-solutions.md)

## 1–2. Shapes and BCE gradient

$X:(32,10)$, $W_1:(10,20)$, $b_1:(1,20)$, $Z_1,A_1:(32,20)$, $W_2:(20,1)$, and
$Z_2,p,y:(32,1)$. Every parameter gradient matches its parameter. For one example,
$\partial L/\partial p=-y/p+(1-y)/(1-p)$ and $\partial p/\partial z=p(1-p)$, which simplify to $p-y$;
divide by $m$ for a mean batch loss.

## 3–5. Implementation tests

ReLU backward multiplies by $1[Z>0]$; cache every $Z_l,A_l$ and traverse layers backward. Removing the tanh derivative
should produce a large numerical-gradient discrepancy even if training occasionally moves. Zero initialization preserves
symmetry, tiny/huge scales shrink/explode signals, and Xavier/He approximately preserve variance for tanh/ReLU.

## 6–7. Fair optimization and Transformers

Use equal data order, update/epoch budget, search budget, and stopping rule for SGD, Momentum, and AdamW; report
validation, test, time, and optimizer memory. In a width-$d$ Transformer, Q/K/V/O projections total about $4d^2$
parameters, while a two-layer MLP has about $2dd_{ff}$, often $8d^2$ when $d_{ff}=4d$.
