# Chapter 21 Solutions

[Back to Chapter 21](../21-interview-and-capstone.md) | [中文](../../zh-CN/solutions/21-solutions.md)

## 1–2. Cards and whiteboards

Each algorithm card includes definition, motivation, input/output shapes, assumption/objective, one failure, validation,
and modern connection. On a whiteboard label shapes first, derive assumption → loss → gradient/update, then state
complexity and numerical checks. Mark uncertain constants instead of inventing them.

## 3. Minimal reproduction

Freeze paper version, data checksum, preprocessing, config, seeds, and environment. Reproduce a simple baseline and one
main-table cell first. Diagnose discrepancies in data, implementation, optimization, evaluation, then hardware, recording
failed attempts and accepted tolerance.

## 4. Problem and data cards

The problem card states user, decision time, falsifiable hypothesis, metrics/guardrails, and stopping rule. The data card
states unit, source, license, label, duplicates, temporal split, and limits. A peer should search for deployment-unavailable
features, post-outcome labels, and causal claims not identifiable from observations.

## 5–7. Evidence and defense

Use a common split and budget for the main experiment; include component/objective, data/negative, and capacity/system
ablations, mechanism counterexamples, predeclared seeds, intervals, latency, memory, and cost. Link each defense claim to
a table, figure, config, or sampled error. For system design proceed through goal/SLO, data, baseline, architecture,
evaluation, capacity, monitoring, rollback, and safety with order-of-magnitude estimates.
