# Course Design: 28-Week Beginner Track / 24-Week Fast Track

[← Home](../../README.en.md) | [中文](../zh-CN/00-course-roadmap.md) | [Chapter 0 →](00-prerequisites.md)

The course combines the approachable progression of Andrew Ng's ML curriculum, the mathematical
coverage of CS229, and the systems discipline of CS249r.

CS229 assumes Python, probability, multivariable calculus, and linear algebra. This course teaches
those assumptions explicitly in Chapters 0–2; beginner-friendliness does not mean omitting rigor.

## Outcomes

You will learn to formulate a real problem as data, a model, an objective, and a metric; derive and
implement core algorithms; diagnose leakage, overfitting, and drift; connect classical ideas to
LLMs and recommenders; and complete a reproducible research project with ablations.

## 28-week beginner sequence

| Weeks | Topic | Modern-AI bridge |
|---:|---|---|
| 0–4 | Python/NumPy, ML concepts, math foundations | tensors, embeddings, softmax, Adam |
| 5–10 | linear/local regression, logistic/GLM, GDA/NB, kernels/SVM, trees | probes, RAG, ranking |
| 11–15 | generalization, learning bounds, K-means, EM/ELBO, PCA/ICA | double descent, MoE, VAE |
| 16–20 | neural nets, backprop, contrastive learning, CNN/RNN/attention | retrieval, CLIP, VLMs |
| 21–23 | Transformers/LLMs, diffusion, recommenders | SFT, LoRA, generation, two-tower models |
| 24–25 | MDPs, value methods, policy gradients/PPO | RLHF and online recommendation |
| 26–27 | evaluation/MLOps and ML systems | Agent evaluation, quantization, KV cache |
| 28 | interviews and research project | reproduction and ablations |

Each algorithm chapter follows: intuition → formulation → derivation → NumPy → framework →
diagnosis → modern-AI bridge → Socratic check → interview questions.

## Completed chapters

- [0. Beginner preparation](00-prerequisites.md), [1. ML concepts](01-what-is-ml.md), and
  [2. mathematical foundations](02-math-foundations.md)
- [3. Regression](03-linear-regression.md), [4. Logistic regression and GLMs](04-logistic-regression.md),
  [5. generative classifiers](05-generative-classifiers.md)
- [6. Neighbors and kernels](06-neighbors-and-kernels.md), [7. SVMs](07-support-vector-machines.md),
  [8. trees and ensembles](08-trees-and-ensembles.md)
- [9. Learning theory and model selection](09-learning-theory-and-model-selection.md)
- [10. Clustering, GMM, EM, and ELBO](10-clustering-em-and-elbo.md)
- [11. PCA, factor analysis, and ICA](11-dimensionality-reduction.md)
- [12. Neural networks and backpropagation](12-neural-networks-and-backprop.md)
- [13. Representation learning](13-representation-learning.md)
- [14. Sequence, vision, and attention](14-sequence-vision-and-attention.md)
- [15. Transformers, LLMs, and fine-tuning](15-transformers-llms-and-finetuning.md)
- [16. VAE and diffusion](16-vae-and-diffusion.md)
- [17. Recommender systems](17-recommender-systems.md)
- [18. Reinforcement learning](18-reinforcement-learning.md)
- [19. Evaluation, causality, and MLOps](19-evaluation-causality-and-mlops.md)
- [20. Machine-learning systems](20-machine-learning-systems.md)
- [21. Interviews and capstone](21-interview-and-capstone.md)

## Assessment

- Derivations and self-checks: 20%
- Programming labs: 30%
- Midterm reproduction: 20%
- Final research project: 30%

The complete English edition is a compact counterpart to the more expansive Chinese primary edition; both versions
share the same concepts, figures, exercises, answers, and navigation.
