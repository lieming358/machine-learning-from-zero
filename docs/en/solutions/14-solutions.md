# Chapter 14 Solutions

[Back to Chapter 14](../14-sequence-vision-and-attention.md) | [中文](../../zh-CN/solutions/14-solutions.md)

## 1–2. Convolution conventions, size, and parameters

Deep-learning cross-correlation uses the kernel as stored; mathematical convolution reverses it. Show every window and
state the convention. Per-axis output is $\lfloor(n+2p-d(k-1)-1)/s\rfloor+1$; parameters are
$K_hK_wC_{in}C_{out}+C_{out}$ and do not multiply by spatial output size.

## 3–4. Multi-channel implementation and shortcut counterexample

Sum each local patch over kernel height, width, and input channels for every output channel. Compare fixed NumPy and
`Conv2d` weights under identical layout, stride, padding, and bias. For the watermark experiment, correlate a corner mark
with the training label, then remove or swap it at test time. A collapse demonstrates shortcut learning; intervention is
stronger evidence than a heatmap alone.

## 5–6. RNN expansion and BPTT

For three steps,
$\partial h_3/\partial h_0=\prod_{t=1}^{3}\phi'(a_t)w_h$ and becomes $w_h^3$ for a linear recurrence. BPTT accumulates
$\partial L/\partial W_h=\sum_t\delta_th_{t-1}^T$, with analogous $W_x,b$ sums. Center-finite-difference every parameter;
large errors often mean a time-index or shared-gradient accumulation bug.

## 7–8. LSTM calculation and gate diagnosis

Compute $c_t=f_tc_{t-1}+i_t\tilde c_t$ and $h_t=o_t\tanh(c_t)$. The direct cell derivative is approximately $f_t$;
the complete derivative also contains gate-dependence paths. Plot gates by time, sample, and hidden dimension. Saturation
near zero/one reduces gate gradients but may be task-appropriate, so combine it with gradient and intervention evidence.

## 9. Fair delayed-copy comparison

Fix data, seeds, optimization steps, search budget, and parameter/FLOP tiers. Report accuracy by delay plus training time,
throughput, and peak memory. Vanilla RNN is expected to degrade earlier, gating can extend memory, and full attention has
direct paths with quadratic interaction cost—but measurements decide the conclusion.

## 10–11. Attention and causal masking

Compute $S=QK^T/\sqrt{d_k}$, stable row softmax, then $AV$; row sums must equal one. Before softmax set $j>i$ logits to
$-\infty$. Test zero upper-triangle probabilities and the stronger property that changing future tokens cannot affect
earlier outputs.

## 12. Leakage-safe sequences

Group by patient/video/device and split chronologically at the prediction time. Fit preprocessing only on training history;
keep adjacent windows with their entity; forbid future aggregates and bidirectional encoders in online tasks. Report cold
entities, long sequences, and later-time slices.
