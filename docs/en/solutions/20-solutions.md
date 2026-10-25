# Chapter 20 Solutions

[Back to Chapter 20](../20-machine-learning-systems.md) | [中文](../../zh-CN/solutions/20-solutions.md)

## 1–2. Amdahl and roofline

Substitute profiled fraction $p$ and local speedup $s$ into $S=1/[(1-p)+p/s]$; even infinite local speedup is capped by
$1/(1-p)$. Estimate operator FLOPs and memory bytes, compare intensity with machine balance
$P_{peak}/B_{memory}$, and confirm cache/fusion effects with a profiler.

## 3–4. Serving and quantization

After warm-up, sweep batch, concurrency, wait time, and realistic lengths. Report throughput, P50/P95/P99, peak memory,
errors, and power, keeping only points above the quality floor. Compare representative-calibration per-tensor and per-
channel int8/int4 on task quality and actual target kernels; a smaller file may silently fall back to fp16.

## 5–6. Memory and parallelism

Parameter bytes equal parameter count times precision; Adam adds gradients, master weights, and two moments; activations
scale with batch/sequence; KV follows the Chapter 15 formula. For eight devices, list all-reduce/all-gather/send-receive
bytes per layer and estimate communication from interconnect bandwidth before reporting scaling efficiency.

## 7. Capacity

Estimate devices from peak QPS, length, and service time with headroom. Add rate limits, priorities, timeouts, continuous
batching, caches, small-model/short-context degradation, idempotent backoff, and previous-version rollback. Measure cost
per successful task and exercise overload behavior.
