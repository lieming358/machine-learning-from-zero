# Chapter 15: Transformers, LLMs, Fine-Tuning, and MoE

[← Chapter 14](14-sequence-vision-and-attention.md) | [中文](../zh-CN/15-transformers-llms-and-finetuning.md) | [Notation](appendix-a-notation.md) | [Roadmap](00-course-roadmap.md)

> **Prerequisites:** attention, cross-entropy, backpropagation, and AdamW.

## From Chapter 14 to this chapter

Attention is only the communication operator. A language model also needs tokenization, positions, causal masking,
residual blocks, an autoregressive objective, decoding, and efficient state reuse. This chapter follows the complete
path from text to next-token prediction, then to SFT, preference optimization, LoRA, KV caching, and MoE.

```mermaid
flowchart LR
  A[Text] --> B[Tokenizer]
  B --> C[Token and position embeddings]
  C --> D[Causal multi-head attention]
  D --> E[Residual/norm/MLP]
  E --> F[Vocabulary logits]
  F --> G[Next-token cross-entropy]
  G --> H[Pretraining]
  H --> I[SFT/preference/LoRA]
  I --> J[KV cache/quantization/MoE]
```

### Learning boundary

After this chapter, you should be able to build shifted next-token targets, trace tensor shapes through causal
attention, explain the training–generation difference, and distinguish SFT, preference learning, and LoRA. Long-context
algorithms, distributed training, and full RLHF systems are included as a map rather than mastery requirements.

## 15.1 The autoregressive objective

$$p(x_{1:T})=\prod_{t=1}^{T}p(x_t\mid x_{<t}),$$

$$L=-\frac1N\sum_{t\in\text{valid tokens}}\log p_t[x_t].$$

The text supplies its own shifted targets. Maximum likelihood teaches statistical regularities, but does not by itself
guarantee truth, instruction following, safety, or current information.

Subword tokenizers trade vocabulary size against sequence length. Token counts vary by language, code, numbers, and
rare names, affecting cost and context capacity. Token IDs become dense embeddings, combined with absolute, relative,
or rotary positional information.

## 15.2 Inside a decoder block

![A decoder block alternates communication and per-token computation](../assets/transformer-block-flow.png)

Masked attention communicates across allowed prefix positions. Residual paths preserve a short information/gradient
route, LayerNorm stabilizes per-token feature scale, and the MLP performs nonlinear token-wise computation. A common
Pre-LN form is

$$H'=H+\operatorname{MHA}(\operatorname{LN}(H)),$$

$$H_{next}=H'+\operatorname{MLP}(\operatorname{LN}(H')).$$

If MLP width is about $4d$, its two matrices have roughly $8d^2$ parameters, often more than Q/K/V/O projections.

## 15.3 Causal masking and leakage

$$M_{ij}=\begin{cases}0,&j\le i,\\-\infty,&j>i.\end{cases}$$

Adding $M$ before softmax gives zero probability to future keys.

![A causal mask removes every future attention weight](../assets/transformer-causal-mask.png)

The upper triangle is forbidden on the left and exactly zero after softmax on the right. A reversed or shifted mask can
make training loss look excellent by exposing the answer. Prefix-invariance and input-target-shift tests are mandatory.

## 15.4 Training parallelism, decoding, and KV cache

Teacher-forced training knows the full sequence and evaluates all masked positions in parallel. Generation must wait
for each new token. KV caching stores historical keys and values so they are not recomputed:

$$M_{KV}=2LTH_{kv}d_hb.$$

![Attention score memory grows quadratically with context](../assets/transformer-context-cost.png)

Doubling context roughly quadruples a full attention-score matrix. FlashAttention reduces intermediate memory traffic
without making standard attention arithmetic linear; KV cache itself grows linearly with context and concurrency.

## 15.5 From pretraining to useful behavior

SFT continues token cross-entropy on instruction-answer examples, usually masking prompt loss. Preference methods use
chosen/rejected responses. RLHF fits a reward model and optimizes a policy; DPO directly changes relative log
probabilities against a reference. Preference data are a measurement process and can encode verbosity or cultural bias.

RAG retrieves external evidence, and tool use lets a model call calculators or services. Both require separate checks
for retrieval, permissions, arguments, grounding, and fallback.

## 15.6 LoRA

Freeze $W_0:d_{out}\times d_{in}$ and learn

$$W=W_0+\frac{\alpha}{r}BA,$$

with $A:r\times d_{in}$ and $B:d_{out}\times r$. Trainable parameters fall from $d_{out}d_{in}$ to
$r(d_{in}+d_{out})$. LoRA saves gradients and optimizer state; it does not automatically reduce base-model inference.

## 15.7 Sparse mixture of experts

$$g(x)=\operatorname{softmax}(W_gx),\qquad
y=\sum_{e\in TopK(g)}g_e(x)E_e(x).$$

Only selected experts process each token, increasing total capacity without activating all parameters.

![Balanced and collapsed MoE routing](../assets/moe-routing-balance.png)

Bars count assigned tokens and the dashed line is ideal mean load. Collapsed routing overloads a few experts, causing
capacity drops and communication hotspots. Auxiliary balancing helps but can interfere with specialization.

## 15.8 Decoding choices

Greedy decoding is deterministic; temperature rescales logits; top-k keeps a fixed number of candidates; top-p
keeps the smallest set reaching a probability mass; beam search retains several sequence hypotheses. Temperature is
not a factuality control. Compare quality, repetition, diversity, length, and latency under fixed prompts and seeds.

## 15.9 Reproduce and diagnose

```bash
python code/chapter15/transformers_and_llms.py
python code/chapter15/course_lab.py
```

The NumPy script asserts that future attention is zero and rows sum to one, then simulates context cost and MoE load.
Production tests should additionally check label shift, padding loss masks, checkpoint reproducibility, and train/eval
behavior.

The course lab prints shifted token pairs, runs causal attention with shape checks, trains a tiny bigram next-token
model, samples with top-p, and compares LoRA parameter counts. The bigram model is deliberately not a Transformer:
it exposes the language-model training loop, while the attention exercise isolates the Transformer operator.

Common failures are target leakage, repetitive decoding, inability to use distant evidence, catastrophic forgetting,
and MoE imbalance. Track loss by token group, contamination, long-context position, KL drift, routing load, TTFT, TPOT,
and KV memory rather than relying on one benchmark.

## Worked reasoning checkpoint

Turn one tokenized sequence into shifted input-target pairs; PAD tokens must be excluded from loss. Track shapes
from [B,T,d] through Q/K/V [B,H,T,dh], attention scores [B,H,T,T], and back to [B,T,d]. With feed-forward width 4d,
a decoder block has roughly 12d-squared weight parameters.

For example, with batch $B=2$, length $T=4$, model width $d=8$, and $H=2$ heads, each head has width $d_h=4$.
Thus $Q,K,V$ have shape $[2,2,4,4]$, attention scores have shape $[2,2,4,4]$, and concatenating the heads returns
$[2,4,8]$. This ledger catches many silent reshape and transpose errors.

A 32-layer model with 32 KV heads, head dimension 128, length 4096, and fp16 needs roughly 2 GiB of KV cache per
sequence; GQA with eight KV heads reduces this near 0.5 GiB. Decode separately reports first-token latency and
tokens per second. Sampling controls text selection, not model knowledge or factuality.

## Modern AI connections

This chapter is the direct core of modern LLMs. Decoder-only models generate, encoder-only models emphasize
representation, and encoder-decoder models condition generation. Multimodal systems add visual/audio tokens or cross
attention. Agents add state, tools, policy, verification, permissions, and monitoring outside the LLM.

## Socratic mentor

<details><summary>Why can next-token training produce downstream abilities?</summary>
Predicting diverse text rewards compressed models of syntax, semantics, facts, and task formats, but does not guarantee reasoning truth.
</details>

<details><summary>Why is masked training parallel while generation is sequential?</summary>
All true training tokens already exist; a generated future token does not.
</details>

<details><summary>Why does KV cache save compute but consume memory?</summary>
It stores per-layer historical K/V to avoid recomputation, growing with context and concurrent sequences.
</details>

<details><summary>Is larger LoRA rank always better?</summary>
It adds adaptation capacity and cost, while increasing overfitting risk; select by validation and resource limits.
</details>

<details><summary>How can MoE have huge parameter count but moderate FLOPs?</summary>
Each token activates only a few experts, although all expert weights and routing communication still exist.
</details>

<details><summary>Does lower temperature prevent hallucination?</summary>
No. It can make the model more consistently choose its highest-probability wrong token.
</details>

## Exercises

See [Chapter 15 solutions](solutions/15-solutions.md).

1. Construct shifted targets and a mask for six tokens.
2. Track every shape and parameter count in two-head attention.
3. Derive perplexity from average cross-entropy.
4. Implement a minimal decoder block and test prefix invariance.
5. Estimate KV memory for a specified model, context, precision, and concurrency.
6. Compare full fine-tuning and LoRA ranks by quality, trainable state, and peak memory.
7. Design MoE load, overflow, communication, and quality monitoring.
8. Design SFT, preference, RAG, and tool-permission evaluation for an assistant.

## Interview recap

- **LM objective?** Maximize conditional likelihood of every true next token.
- **Why causal mask?** Prevent access to future answers.
- **Residual and LayerNorm?** Short information/gradient paths and stable per-token scale.
- **Training versus generation?** Parallel teacher forcing versus autoregressive dependence.
- **KV cache?** Historical per-layer keys and values.
- **Perplexity?** Exponential of mean token negative log-likelihood.
- **What does LoRA save?** Trainable parameters, gradients, and optimizer state.
- **MoE tradeoff?** Sparse active compute with routing, storage, and communication complexity.
- **Top-k versus top-p?** Fixed candidate count versus adaptive cumulative mass.
- **SFT versus preference optimization?** Demonstration imitation versus relative behavior preference.

## Mastery checklist

- [ ] I can trace string, tokens, embeddings, positions, logits, and shifted targets.
- [ ] I can derive block shapes, causal masking, likelihood, and perplexity.
- [ ] I can explain training/decode parallelism and KV cache cost.
- [ ] I can derive LoRA parameter counts and diagnose adaptation failure.
- [ ] I can explain sparse MoE activation and routing balance.
- [ ] I can design layered evaluation for SFT, preference, RAG, and tools.

### Further reading

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- [Language Models are Few-Shot Learners](https://arxiv.org/abs/2005.14165)
- [LoRA](https://arxiv.org/abs/2106.09685)
- [Switch Transformers](https://arxiv.org/abs/2101.03961)
- [FlashAttention](https://arxiv.org/abs/2205.14135)

[Next: VAEs and Diffusion →](16-vae-and-diffusion.md)
