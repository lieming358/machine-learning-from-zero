# Chapter 21: Course Synthesis — Interviews and a Reproducible Capstone

[← Chapter 20](20-machine-learning-systems.md) | [中文](../zh-CN/21-interview-and-capstone.md) | [Notation](appendix-a-notation.md) | [Roadmap](00-course-roadmap.md)

> **Prerequisites:** Chapters 1–20. This chapter adds no isolated model; it organizes the course into explainable,
> implementable, testable, and transferable competence.

The remaining risk is “I read every chapter, but do not know where to begin on a new problem.” Two products test real
mastery: an interview answer that moves from definition through assumptions, derivation, failure, and systems; and a
research project whose baselines, ablations, error analysis, and reproducibility support its claims.

## Learning route

```mermaid
flowchart LR
  A["New problem"] --> B["Task data metric"]
  B --> C["Simple baseline"]
  C --> D["Mechanism hypothesis"]
  D --> E["Implementation checks"]
  E --> F["Fair main experiment"]
  F --> G["Ablation counterexample errors"]
  G --> H["System cost limitations"]
  H --> I["Reproduction defense transfer"]
```

### Learning boundary

The capstone is not expected to produce research-level novelty. It must demonstrate a small, complete evidence chain:
a falsifiable problem, a fair baseline, paired uncertainty, one useful ablation, error analysis, and claims whose scope
matches the evidence.

## 21.1 Six layers of knowing a model

1. **Intuition:** why it exists and what fails without it.
2. **Representation:** input, output, parameter, shape, and assumption.
3. **Derivation:** from probability or geometry to objective and update.
4. **Implementation:** core from scratch, framework comparison, numerical checks.
5. **Diagnosis:** infer causes from curves, errors, and profiles.
6. **Transfer:** locate the idea in modern AI and state its boundary.

![Mastery is multi-dimensional rather than one score](../assets/course-mastery-radar.png)

A student may recall terms yet be weak in derivation and diagnosis; another may operate a library without explaining a
failure. Use code, derivations, exercises, and defense evidence to locate—not cosmetically maximize—the radar.

## 21.2 The interview backbone

For “explain this algorithm,” avoid a keyword dump:

1. define the task and central assumption in one sentence;
2. explain the earlier pain and why the method exists;
3. give the key equation, symbols, and solver;
4. give one counterexample and diagnostic signal;
5. propose a fair experiment, complexity report, and modern use.

For Ridge versus Lasso, do more than say L2 versus L1: discuss coefficient geometry, collinearity, sparse-selection
conditions, feature scaling, validation of $\lambda$, and unstable selection among correlated features.

## 21.3 Build a knowledge network

Connect data splits, leakage, bias–variance, regularization, thresholds, calibration, uncertainty, and drift to every
model family. Connect OLS/MLE/MAP, logistic/softmax, GDA/NB, margins/kernels, trees/ensembles, EM/mixtures, PCA,
backpropagation, attention, contrastive learning, VAE/diffusion, recommendation, and RL to systems complexity and
evidence.

Every node should have two-way links. Cross-entropy connects logistic regression, softmax, language modeling, InfoNCE,
and policy learning. Low rank connects PCA, matrix factorization, LoRA, and compression—but similar algebra does not
make their objectives identical.

## 21.4 Why flashcards alone do not transfer

![Practice should move from recall to reasoning on new cases](../assets/interview-practice-transfer.png)

Definition recall saturates quickly in this teaching illustration; transfer grows through counterexamples and diagnosis.
After every answer ask what changes if an assumption, metric, or system budget changes, and what experiment separates
two explanations. Practice in three rounds: a 60-second answer, whiteboard derivation/code, then open system design.

## 21.5 Write a one-page problem card

Before implementation, state:

- problem, user, and feature availability at prediction time;
- sample unit, label source, and time;
- primary metric, guardrails, and minimum baseline;
- falsifiable mechanism hypothesis;
- train/validation/test and contamination controls;
- compute, data, privacy, and time budgets;
- success, failure, and stopping rules.

If this page is unclear, adding a larger model rarely fixes the project.

## 21.6 A defensible evidence chain

![A research project connects the problem to reproducible evidence](../assets/research-project-evidence-chain.png)

The data card bounds evidence; baselines establish necessity; ablations test mechanism; error analysis finds boundaries;
systems profiling tests deployability. Each conclusion should trace backward to an experiment and artifact.

```text
project/
  README.md              # one-command reproduction and claims
  data_card.md           # source, license, splits, limits
  configs/               # experiment configurations
  src/                   # data, model, training, evaluation
  tests/                 # shape, leakage, numerical, regression
  reports/               # tables, plots, errors, model card
  environment/           # dependencies and hardware
```

Do not commit large raw data or secrets; preserve checksums and lawful access instructions.

## 21.7 Baselines and fair comparison

Use at least a constant/random/popularity rule to test the metric pipeline, a classical interpretable baseline, and the
strongest closely related method. Keep data, preprocessing, search budget, seed set, stopping, hardware, and metrics
aligned. If a proposal uses ten times the compute, report both equal-budget and best-available views.

## 21.8 Ablation tests mechanism

![Ablation compares component gains relative to baseline](../assets/capstone-ablation.png)

Each bar changes one component and error bars show run variation. A useful ablation follows a mechanism hypothesis and
controls capacity and budget. Test components, objectives, data/negatives, model capacity and FLOPs, stress cases such
as noise/drift/long tail, and system settings such as precision or cache.

## 21.9 Error analysis returns to examples

Define an interpretable error taxonomy, sample randomly rather than selecting dramatic cases, and double-label a subset
for agreement. Track category prevalence, confidence, groups, and version changes, then choose fixes that cover many
errors. For RAG, separate corpus absence, retrieval miss, ranking failure, ignored evidence, reasoning error, bad
citation, and safety-refusal error. Qualitative examples explain; they do not replace quantitative evidence.

## 21.10 Statistical reporting and honest claims

Report mean/standard deviation or intervals across predeclared seeds, the sampling unit, and effect size. Use the test
set after design freezes. If test results guide another change, acknowledge that it became validation and establish new
test evidence. A 2% gain on one dataset does not support universal superiority. Synthetic results test a mechanism, not
deployment. A reproducible negative result can still identify exactly where an assumption fails.

## 21.11 Reproduce the reporting skeleton

```bash
python code/chapter21/capstone_and_interviews.py
python code/chapter21/course_lab.py
```

The script builds ablation, mastery, evidence-chain, and practice-transfer figures and checks gain definitions. A real
project should expose one command that rebuilds key tables and plots from a fixed small sample.

The course lab validates a problem card, computes a paired bootstrap interval, summarizes an ablation and error
taxonomy, and checks that each claim names its evidence and boundary. It is a reusable submission checklist rather
than another model to memorize.

## 21.12 Capstone options and minimum delivery

Possible projects include induction-bias comparisons on small data; contrastive negatives and retrieval; layered RAG
ablations; recommendation retrieval/ranking/diversity; VAE/diffusion quality–coverage–steps; offline-RL coverage or PPO
stability; and quantization/batching/KV quality–latency–memory Pareto analysis.

Minimum delivery: problem card, data card, reproduction README, baselines, main results, at least three mechanistic
ablations, error analysis, system cost, limitations, model card, and a ten-minute defense.

## 21.13 Answering an ML system-design question

Proceed in order: clarify goal/SLO → data/labels → baseline → architecture/stages → training/evaluation → serving/
capacity → monitoring/rollback → safety/privacy → trade-offs. Estimate QPS, candidates, dimensions, model size,
latency, and update rate. Explain why each component exists and how it fails.

For RAG, cover answer/evidence quality, document parsing and permissions, versioning, embeddings/index, retrieval and
re-ranking, prompt/citations, layered offline evaluation, cache/latency/cost, injection and authorization, no-answer
fallback, and online monitoring.

## 21.14 Scope and failure diagnosis

Start with a small end-to-end path before scaling. Maintain an experiment ledger with hypothesis, config, commit, data,
result, and conclusion—not only `final_v7_really_final.ipynb`.

| Symptom | Cause | Fix |
|---|---|---|
| result cannot reproduce | missing data/environment/randomness | lock versions and tolerance tests |
| all changes added together | attribution impossible | mechanistic and interaction ablations |
| only one seed wins | variance or selection | preset seeds and report distribution |
| test improves after every edit | test leakage | freeze a new test or use nested design |
| attractive figures, no argument | claims not mapped to evidence | claim–evidence table and pruning |
| demo fast, batch crashes | scale untested | length/concurrency/resource stress tests |

## Worked capstone example

Define a campus-policy RAG project as: answer real policy questions with supported citations under a two-second P95.
Use keyword retrieval and embedding retrieval as baselines, then test whether a cross-encoder reranker repairs a named
failure. Report evidence Recall@5, answer correctness, citation support, latency, and cost. Ablate reranking, chunk size,
and embeddings; classify failures as missing corpus, retrieval miss, reranking error, unused evidence, or bad citation.

The final claim should read: under dataset, protocol, and budget X, method A changes metric Y relative to baseline B
at cost Z. Deliver a problem card, data card, one-command reproduction, fair baselines, multi-seed main results,
mechanism-driven ablations, random-sample error analysis, a quality-cost Pareto curve, limitations, and a model card.

## Modern AI connections

- Paper reproduction aligns protocol and baseline before changing a model.
- LLM applications need layered evaluation, prompt/version management, safety, and cost—not one demo.
- Open-source contribution benefits from a minimal reproduction, tests, and a small reviewable change.
- Research novelty is a testable problem, insight, mechanism, or evidence—not merely more modules.
- Interviews often follow an algorithm into data, failure, and systems; a knowledge network is more robust than scripts.

## Socratic mentor

<details><summary>Why keep a simple baseline after a large gain?</summary>
It validates data and metrics and tests whether complexity is necessary; many gains disappear after leakage is removed.
</details>

<details><summary>If removing a module hurts, is its proposed mechanism proven?</summary>
No. Parameter count, compute, or regularization also changed. Match those factors and test counterexamples.
</details>

<details><summary>Why randomly sample errors?</summary>
Selecting memorable failures cannot estimate prevalence or expected benefit from a fix.
</details>

<details><summary>What if I forget a formula in an interview?</summary>
Reconstruct from assumptions, shapes, and objective; clearly flag an uncertain constant rather than confidently inventing it.
</details>

## Exercises

Compare with the [Chapter 21 solutions](solutions/21-solutions.md).

1. Write 60-second definition–assumption–failure–validation cards for 20 algorithms.
2. Randomly choose three chapters and derive objective-to-update with shapes on a whiteboard.
3. Minimally reproduce one public result and log every discrepancy.
4. Write a problem and data card and ask a peer to find leakage or non-identification.
5. Design a main experiment, three ablation types, counterexamples, intervals, and system profile.
6. Deliver a ten-minute defense and audit every unsupported claim.
7. Design RAG, recommendation, or edge inference with capacity, monitoring, and rollback.

## Interview recap

- **How choose a model?** Start from data structure, sample size, metric, and system constraints; validate independently.
- **How detect overfitting?** Training improves while validation worsens; inspect learning curves and leakage.
- **Fair comparison?** Same data, preprocessing, budget, seeds, stopping, hardware, and metrics.
- **Why ablate?** Attribute gains and test mechanism, not merely show the full system works.
- **Error analysis?** Taxonomy, random sample, prevalence/slices, then high-coverage fixes.
- **Statistical versus practical significance?** Intervals express uncertainty; effect and cost decide usefulness.
- **First system-design questions?** User goal, scale, SLO, data availability, and failure cost.
- **Research novelty?** A new question, insight, mechanism, or evidence—not a pile of names.

## Mastery checklist

- [ ] I can explain core methods from Chapters 1–20 at all six layers.
- [ ] I can derive an objective and update while checking shapes.
- [ ] I can turn an open task into a falsifiable, leakage-safe experiment.
- [ ] I can design fair baselines, ablations, counterexamples, intervals, and error analysis.
- [ ] I can report a quality–latency–memory–cost Pareto frontier and limits.
- [ ] I can deliver reproduction, data/model cards, ledger, and rollback documentation.
- [ ] I separate fact, assumption, evidence, and unknown in a defense.

### Further reading

- [Pineau et al., Improving Reproducibility in ML Research](https://www.jmlr.org/papers/v22/20-303.html)
- [NeurIPS Paper Checklist](https://neurips.cc/public/guides/PaperChecklist)
- [Stanford CS229](https://cs229.stanford.edu/)
- [Harvard CS249r](https://github.com/harvard-edge/cs249r_book)

The Chapters 0–21 pathway is complete. Select one capstone and turn course comprehension into independent evidence.
