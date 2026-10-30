# KV Cache: Don't Recompute What You Already Know

> **The one idea in this guide:** when a model generates text one token at a time, the **keys and values** of earlier tokens never change. The **KV cache** saves them, so each new step only computes keys and values for the **one new token**. This makes generation much faster, but the cache uses memory that grows with every token.
>
> Everything below explains, demonstrates or deepens that idea.

## 1. In simple words

From [LLM Inference](llm_inference.md): a model writes one token, adds it to the text, and runs again. Each run, attention needs a **key** and a **value** for **every** token so far.

Without a cache, step 100 would recompute keys and values for all 100 tokens, even though 99 of them were already computed in earlier steps and haven't changed.

```
Without cache                          With cache
step 1:  compute K,V for  I love        step 1:  compute I love       → save
step 2:  compute K,V for  I love cats   step 2:  compute cats only    → save
step 3:  compute K,V for  I love cats … step 3:  compute the new one  → save
```

**Analogy: a running total on a receipt.** When the cashier scans a new item, they don't re-add every item from the start; they add the new price to the running total. The KV cache is the model's running record.

> **Quick check 1.** Why can earlier tokens' keys and values be reused safely?
>
> *Answers to all checks are in Section 7.*

## 2. What you need to know first

- **[LLM Inference](llm_inference.md)**: prefill, decode, and a first look at the KV cache. **Read it first.**
- **[Attention Blocks](attention_blocks.md)**: queries, keys and values.
- **[Attention Masks](attention_masks.md)**: the causal mask, which is why the cache works.

## 3. The key terms

| Term | Meaning |
|---|---|
| Query (Q) | what the current token is "looking for" |
| Key (K) | what each token "offers" to be matched against |
| Value (V) | the information each token passes on when attended to |
| Causal mask | each token may attend only to itself and earlier tokens |
| KV cache | stored keys and values for all earlier tokens, in every layer |
| Prefill | reading the whole prompt at once, filling the cache |
| Decode | generating one token at a time, adding one entry per step |
| Head dimension | the width of one attention head's key or value vector |

## 4. How it works

### Why the cache is correct

Because of the **causal mask**, a token never attends to tokens **after** it. So when new tokens are added, nothing about an earlier token's computation changes, including its keys and values. They can be computed once and reused forever (for this conversation).

### What happens at each decode step

For the **new** token only:

1. Compute its query, key and value (in every layer).
2. **Append** its key and value to the cache.
3. Attend: compare its query with **all** cached keys, and mix all cached values.

Note step 3: the new token still **reads** every cached key and value. The cache saves **recomputing** them, not reading them.

### Prefill and decode

| Phase | What happens | Cache |
|---|---|---|
| **Prefill** | all prompt tokens processed in parallel | filled with every prompt token's K and V |
| **Decode** | one new token per step | grows by one token per step |

### How much memory?

```
cache size = 2 × layers × KV heads × head dimension × bytes per number × tokens
             ▲
          keys and values
```

## 5. Try it yourself

### The toy model from LLM Inference

In [LLM Inference](llm_inference.md), the model generated "I love cats":

| Step | Without cache: compute K, V for | With cache: compute | Cache afterwards |
|---|---|---|---|
| 1 (prefill "I love") | I, love | I, love | I, love |
| 2 (decode "cats") | I, love, cats | **cats only** | I, love, cats |

In step 2, the key of "I" was [1, 0] and of "love" was [0, 1], exactly as in step 1. The cache just hands them over.

### Counting work

Generate 100 tokens, one at a time:

```
Without cache:  1 + 2 + 3 + ... + 100 = 100 × 101 ÷ 2 = 5,050 key/value computations
With cache:     1 + 1 + 1 + ... + 1   = 100
```

About **50 times less** work, and the gap grows with length.

### Counting memory

A typical mid-sized model:

- 32 layers
- 32 KV heads
- head dimension 128
- 16-bit numbers (2 bytes each)

**Per token:**

```
2 × 32 × 32 × 128 × 2 bytes = 524,288 bytes = 0.5 MiB
```

**For 4,096 tokens:**

```
0.5 MiB × 4,096 = 2,048 MiB = 2 GiB
```

That's for **one** conversation. A server handling 8 conversations of that length needs **16 GiB**, just for caches. This is why the KV cache is often the main memory limit for long contexts and many users.

### Grouped-query attention (GQA) shrinks it

Many models let several query heads **share** one key/value head. With 32 query heads but only **8 KV heads**:

```
2 × 32 × 8 × 128 × 2 bytes = 131,072 bytes = 0.125 MiB per token
4,096 tokens → 512 MiB = 0.5 GiB         (4 times smaller)
```

> **Quick check 2.** Using the first model (0.5 MiB per token), how big is the cache for 8,192 tokens?

## 6. Variations, tips and mistakes

### Ways to reduce KV cache memory

| Technique | Idea |
|---|---|
| **Multi-query / grouped-query attention** | fewer KV heads, shared by many query heads |
| **[Quantized](quantization.md) cache** | store keys and values in 8 bits or less |
| **Sliding window** | keep only the most recent N tokens (for some layers or models) |
| **Paged memory** | store the cache in fixed-size blocks, like pages, to avoid wasted space (as in vLLM's PagedAttention) |
| **Evicting less useful tokens** | drop entries that receive little attention (research area) |

### Reusing caches across requests: prompt caching

If many requests start with the **same** text (a long system prompt, a big document), the cache for that shared beginning can be computed once and reused. Some services offer this as **prompt caching**, often with lower cost and faster responses for the repeated part.

**It only works for an identical beginning.** Change one early token and every later cache entry is different, because later tokens attended to it.

### Common misunderstandings

| Misunderstanding | Reality |
|---|---|
| "The cache makes attention free." | Each new token still reads all cached keys and values; that grows with length. |
| "The cache is learning." | It's just saved intermediate numbers. No weights change. |
| "The cache is shared between users." | Normally each conversation has its own; only identical prefixes can be shared. |
| "Prefill uses the cache to go faster." | Prefill **fills** the cache; the savings come during decode. |
| "Memory only depends on model size." | The cache grows with context length × number of conversations. |

## 7. Remember this

### Five essential ideas

1. **Earlier keys and values never change**, thanks to the causal mask.
2. **Save them once, reuse them:** each decode step computes K and V for one new token.
3. **Reading still grows:** each step attends to all cached entries.
4. **Memory = 2 × layers × KV heads × head dim × bytes × tokens**, and it's often the main limit.
5. **GQA, quantization, paging and prompt caching** are the main tools for taming it.

### Exercises

1. Without a cache, how many key/value computations to generate 10 tokens one by one (starting from nothing)? With a cache?
2. A model has 40 layers, 8 KV heads, head dimension 128, 16-bit numbers. How many bytes per token?
3. Why does changing the first word of a long, cached system prompt break prompt caching?
4. Why doesn't the cache reduce the cost of comparing the new query with all earlier keys?

Try the exercises and the quick checks before reading on.

---

### Check your answers

**Quick checks**

1. Because of the **causal mask**: earlier tokens never attend to later ones, so adding tokens doesn't change their keys and values.
2. 0.5 MiB × 8,192 = **4,096 MiB = 4 GiB**.

**Exercises**

1. Without: 1 + 2 + … + 10 = **55**. With: **10**.
2. 2 × 40 × 8 × 128 × 2 = **163,840 bytes** (160 KiB).
3. Every later token attended to the first word, so all their keys and values change.
4. The cache saves **computing** keys and values, but the new token must still **compare** with every one of them to do attention.

## 8. Going further

### Where this connects

- **[LLM Inference](llm_inference.md)**: prefill and decode.
- **[Quantization](quantization.md)**: storing the cache in fewer bits.
- **[In-Context Learning](in_context_learning.md)**: long prompts with many examples mean big caches.
- **[Mixture of Experts](mixture_of_experts.md)**: MoE saves compute in the MLP, but the attention KV cache is still needed.

### Short glossary

| Term | Meaning |
|---|---|
| KV cache | saved keys and values of earlier tokens |
| Prefill | processing the prompt and filling the cache |
| Decode | generating one token per step |
| GQA | grouped-query attention: query heads share KV heads |
| MQA | multi-query attention: all query heads share one KV head |
| PagedAttention | storing the cache in blocks to reduce waste |
| Prompt caching | reusing the cache for an identical prompt beginning |

### Learn more

- 📖 [GQA: Training Generalized Multi-Query Transformer Models](https://arxiv.org/abs/2305.13245) (Ainslie et al., 2023)
- 📖 [Efficient Memory Management for LLM Serving with PagedAttention](https://arxiv.org/abs/2309.06180) (Kwon et al., 2023): the vLLM paper
