# Mixture of Experts: Many Specialists, Only a Few at a Time

> **The one idea in this guide:** a **mixture-of-experts (MoE)** layer contains many small networks (**experts**) and a **router**. For each token, the router picks just a **few** experts (often 1 or 2) to run. The model can then have a huge number of parameters in total, while each token only uses a small part of them.
>
> Everything below explains, demonstrates or deepens that idea.

## 1. In simple words

In a normal (**dense**) transformer, every token passes through **every** weight of every layer. Making the model bigger makes every token more expensive.

A **mixture of experts** replaces one big feed-forward network with several smaller ones, the **experts**, plus a **router** that sends each token to only a few of them:

```
                   ┌─▶ expert 1 ─┐
                   ├─▶ expert 2   │  (skipped)
token ─▶ router ───┼─▶ expert 3 ─┤──▶ weighted mix ─▶ output
                   └─▶ expert 4   │  (skipped)
              picks the top 2: experts 1 and 3
```

**Analogy: a hospital.** A hospital has many specialists, but you don't see all of them. A receptionist (the router) sends you to the two most relevant doctors. The hospital has lots of knowledge in total, but each patient only uses a little of the staff's time.

> **Quick check 1.** In an MoE layer with 8 experts and top-2 routing, how many experts process each token?
>
> *Answers to all checks are in Section 7.*

## 2. What you need to know first

- **[Attention Blocks](attention_blocks.md)**: a transformer block has attention followed by a feed-forward network (MLP). MoE usually replaces the MLP.
- **[Activation Functions](activation_functions.md)**: softmax, used by the router.
- **[LLM Inference](llm_inference.md)**: why memory and compute both matter.

## 3. The key terms

| Term | Meaning |
|---|---|
| Dense model | every token uses all the weights |
| Sparse model | each token uses only part of the weights |
| Expert | one of several small feed-forward networks in an MoE layer |
| Router (gate) | a small layer that scores the experts for each token |
| Top-k routing | send each token to its k highest-scoring experts |
| Total parameters | all weights in the model, including every expert |
| Active parameters | the weights actually used for one token |
| Load balancing | keeping the experts roughly equally busy |
| Capacity | the maximum number of tokens an expert can take in one batch |

## 4. How it works

### Where MoE sits in a transformer

A transformer block is: attention → feed-forward network (MLP). In an MoE model, the MLP is replaced by an **MoE layer**. Attention usually stays the same and is shared by all tokens.

```
Dense block:  attention → MLP
MoE block:    attention → router + experts
```

### The router, step by step

For each token's vector x:

1. **Score:** the router computes one score (logit) per expert, using a small learned matrix.
2. **Softmax:** turn the scores into probabilities.
3. **Top-k:** keep the k highest. Set the rest aside.
4. **Renormalise:** rescale the kept probabilities so they sum to 1. These are the **gate weights**.
5. **Run** only the chosen experts on x.
6. **Mix:** output = sum of (gate weight × expert output).

(Models differ in small details, for example whether softmax comes before or after picking the top k.)

### Total vs active parameters

This is the key trade-off:

- **Compute per token** depends on **active** parameters (the chosen experts + shared parts).
- **Memory** depends on **total** parameters: every expert must be loaded, because any token might need any expert.

So an MoE model can be as **fast** as a much smaller dense model, but needs as much **memory** as a big one.

### Do experts specialise?

Somewhat, but not in neat human categories like "the maths expert". Studies often find experts specialise in **token-level patterns**: certain kinds of words, punctuation, or syntax. The router learns whatever split helps most.

## 5. Try it yourself

### Route one token through 4 experts (top-2)

> **Made-up numbers.** Real experts output vectors; ours output a single number so you can follow by hand.

**Router logits** for this token: [2.0, 0.5, 1.0, −1.0]

**Step 1: softmax.**

```
e^2.0 = 7.389   e^0.5 = 1.649   e^1.0 = 2.718   e^−1.0 = 0.368     sum = 12.124
```

| Expert | Logit | Probability |
|---|---|---|
| 1 | 2.0 | 7.389 ÷ 12.124 = **0.609** |
| 2 | 0.5 | 0.136 |
| 3 | 1.0 | 2.718 ÷ 12.124 = **0.224** |
| 4 | −1.0 | 0.030 |

**Step 2: top-2.** Experts **1** and **3**.

**Step 3: renormalise.**

```
sum of kept = 0.609 + 0.224 = 0.833
gate 1 = 0.609 ÷ 0.833 = 0.731
gate 3 = 0.224 ÷ 0.833 = 0.269
```

(Using unrounded values gives 0.7311 and 0.2689.)

**Step 4: run only experts 1 and 3.** Suppose expert 1 outputs **10** and expert 3 outputs **6**. Experts 2 and 4 are **never run**.

**Step 5: mix.**

```
output = 0.731 × 10 + 0.269 × 6 = 7.31 + 1.61 = 8.92
```

### Total vs active, with real-looking sizes

Imagine a model with:

- shared parts (attention, embeddings): **2 billion** parameters
- MoE layers with **8 experts**, each **5 billion** parameters in total across layers
- **top-2** routing

```
total parameters  = 2 + 8 × 5 = 42 billion   ← must fit in memory
active per token  = 2 + 2 × 5 = 12 billion   ← compute per token
```

Each token costs roughly what a 12-billion-parameter dense model costs, while the model holds 42 billion parameters of knowledge. (Real models such as Mixtral 8×7B follow this pattern, with about 47B total and 13B active parameters.)

> **Quick check 2.** Same shared size (2 B) and expert size (5 B), but 16 experts with top-1 routing. Total and active parameters?

## 6. Variations, tips and mistakes

### The load-balancing problem

If the router learns to send almost every token to the same expert or two, the other experts sit idle and don't learn, and the busy experts become a bottleneck. To prevent this, training adds a **load-balancing loss** that rewards spreading tokens evenly. Some newer methods balance in other ways, for example by adjusting a bias on each expert's score.

### Capacity and dropped tokens

In a batch, each expert may have a **capacity** limit. If too many tokens choose the same expert, the extras may be **dropped** (they skip the expert and pass through unchanged via the residual connection) or sent to their next choice.

### Variations

| Variation | Idea |
|---|---|
| **Top-1 routing (Switch Transformer)** | each token goes to a single expert: simplest and cheapest |
| **Top-2 routing** | each token mixes two experts: common and robust |
| **Shared experts** | one or more experts that every token always uses, plus routed experts |
| **Fine-grained experts** | many small experts instead of a few big ones, with more of them chosen per token |

### Pros and cons

| Pros | Cons |
|---|---|
| more knowledge for the same compute per token | needs memory for **all** experts |
| faster training for a given quality | training can be unstable; load balancing needed |
| fast inference per token | spreading experts across many GPUs adds communication |
| | small batches may use experts unevenly, wasting hardware |

### Common misunderstandings

| Misunderstanding | Reality |
|---|---|
| "An 8-expert model is 8 separate models." | The experts are inside each layer, sharing attention and embeddings; routing happens per token, per layer. |
| "Each expert is a topic specialist." | Specialisation is mostly about token patterns, not clean topics. |
| "MoE saves memory." | It saves **compute**, not memory. All experts must be loaded. |
| "A token uses the same experts in every layer." | Each layer has its own router and can choose different experts. |

## 7. Remember this

### Five essential ideas

1. **Experts + router.** The MLP is replaced by several expert networks and a router.
2. **Top-k routing:** each token uses only k experts; the rest are skipped.
3. **Gate weights mix the outputs:** softmax, keep the top k, renormalise.
4. **Total vs active:** memory follows total parameters, compute follows active ones.
5. **Load balancing** keeps all experts useful and busy.

### Exercises

1. Router logits [1, 1, 0, 0], top-2. Which experts are chosen, and what are their gate weights?
2. Using the example's gates (0.731, 0.269), what's the output if expert 1 outputs 4 and expert 3 outputs 0?
3. Why must all experts be kept in memory even though each token only uses two?
4. What goes wrong without load balancing?

Try the exercises and the quick checks before reading on.

---

### Check your answers

**Quick checks**

1. **2.**
2. Total = 2 + 16 × 5 = **82 billion**. Active = 2 + 1 × 5 = **7 billion**.

**Exercises**

1. Experts **1 and 2** (logits 1 and 1). Equal logits give equal probabilities, so after renormalising each gets **0.5**.
2. 0.731 × 4 + 0.269 × 0 = **2.924**.
3. Different tokens choose different experts, and the next token could need any of them.
4. The router may send most tokens to a few experts. The others don't learn, and the busy ones become a bottleneck.

## 8. Going further

### Where this connects

- **[Attention Blocks](attention_blocks.md)**: the block whose MLP MoE replaces.
- **[Quantization](quantization.md)**: often used to fit large MoE models in memory.
- **[KV Cache](kv_cache.md)**: MoE doesn't change the attention cache.
- **[Distillation](distillation.md)**: a large MoE model can teach a smaller dense one.

### Short glossary

| Term | Meaning |
|---|---|
| Mixture of experts | a layer with many experts and a router |
| Expert | a small feed-forward network |
| Router / gate | chooses experts for each token |
| Top-k routing | use the k best experts |
| Active parameters | weights used per token |
| Total parameters | all weights stored |
| Load-balancing loss | an extra loss that spreads tokens across experts |
| Capacity | the most tokens an expert accepts per batch |

### Learn more

- 📖 [Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer](https://arxiv.org/abs/1701.06538) (Shazeer et al., 2017)
- 📖 [Switch Transformers](https://arxiv.org/abs/2101.03961) (Fedus, Zoph and Shazeer, 2021): top-1 routing at scale
- 📖 [Mixtral of Experts](https://arxiv.org/abs/2401.04088) (Jiang et al., 2024): an open MoE language model
