# Attention Masks: Who Is Allowed to Look at Whom

> **The one idea in this guide:** a mask is a table of **"allowed"** and **"blocked"** that decides which tokens each token may pay attention to. Every mask works the same way; only the pattern of the table changes.

## 1. In simple words

In attention, every token looks at other tokens to gather information (see [Attention Blocks](attention_blocks.md)). But sometimes a token **must not** look at some of them:

- When writing text, a word must not peek at the words that come **after** it.
- Padding tokens are just filler and should be **ignored**.
- In very long texts, looking at **everything** is too expensive.

A **mask** handles all of these. It's a table with one row per token that's looking, and one column per token it might look at. Each cell says **allowed** or **blocked**.

**Analogy: a classroom with rules about who can listen to whom.** Every student (token) wants to listen to the others before answering. The teacher gives each student a list:

- "You may only listen to students who spoke **before** you" (causal mask).
- "Ignore the empty chairs" (padding mask).
- "Only listen to your **neighbours**" (sliding window).

The students are the same. Only the rules on the list change.

In this guide, ✓ means **allowed** and · means **blocked**. Without any rules, every token may look at every token (**full attention**):

```
            looks at → token 1   token 2   token 3   token 4
token 1                  ✓         ✓         ✓         ✓
token 2                  ✓         ✓         ✓         ✓
token 3                  ✓         ✓         ✓         ✓
token 4                  ✓         ✓         ✓         ✓
```

A mask changes this pattern. For example, the **causal** mask (Section 5.2) only allows looking at earlier tokens:

```
            looks at → token 1   token 2   token 3   token 4
token 1                  ✓         ·         ·         ·
token 2                  ✓         ✓         ·         ·
token 3                  ✓         ✓         ✓         ·
token 4                  ✓         ✓         ✓         ✓
```

**Rows** are the token doing the looking (the query). **Columns** are the tokens being looked at (the keys).

## 2. What you need to know first

- **[Attention Blocks](attention_blocks.md)**: what the score matrix S is and how softmax turns scores into attention weights.
- **Softmax** in one line: it turns a list of scores into weights that are ≥ 0 and add up to 1, using e^score.

## 3. The key terms

| Term | Meaning |
|---|---|
| **Mask** | a table saying, for each pair of tokens, whether attention is allowed |
| **Query (row)** | the token that is looking |
| **Key (column)** | the token being looked at |
| **−∞** | "minus infinity": the value used to block a cell, because e^(−∞) = 0 |
| **Combining masks** | a cell is allowed only if **every** mask allows it |

## 4. How a mask works

### The trick: add 0 or −∞ before softmax

A mask is stored as numbers:

- **0** where attention is **allowed** (adding 0 changes nothing)
- **−∞** where attention is **blocked**

It's **added** to the scores just before softmax:

```
weights = softmax( scores + mask )
```

Since e^(−∞) = 0, every blocked cell gets a weight of **exactly 0**. The remaining weights still add up to 1.

> **In real code:** implementations may use a very large negative number (like −10,000) instead of −∞, or a true/false (Boolean) mask handled directly by the attention code. A large negative number makes the weight negligible, and often exactly 0 in practice, but it isn't guaranteed to be exactly 0 in every implementation. This guide uses −∞ because it makes the math exact.

### A worked example, step by step

Token 2 has these attention scores for tokens 1 to 4:

```
scores = [1, 2, 0, 3]
```

**Without a mask:**

```
e^score = [e¹, e², e⁰, e³]
        = [2.718, 7.389, 1.000, 20.086]
sum     = 2.718 + 7.389 + 1.000 + 20.086 = 31.193

weights = [2.718/31.193, 7.389/31.193, 1.000/31.193, 20.086/31.193]
        = [0.087, 0.237, 0.032, 0.644]
```

Token 2 puts most of its attention (64%) on **token 4**, a token that comes **after** it. When generating text, that's cheating: token 4 hasn't been written yet.

**With a causal mask** (token 2 may only see tokens 1 and 2):

```
mask            = [0, 0, −∞, −∞]
scores + mask   = [1, 2, −∞, −∞]

e^(scores+mask) = [2.718, 7.389, 0, 0]
sum             = 2.718 + 7.389 + 0 + 0 = 10.107

weights = [2.718/10.107, 7.389/10.107, 0/10.107, 0/10.107]
        = [0.269, 0.731, 0, 0]
```

The blocked tokens get **0**, and the allowed ones share all the attention between them.

### The shape of a mask

Conceptually, a mask says which **query–key pairs** are allowed, so its pattern lines up with the score matrix: one row per query and one column per key. For n tokens attending to themselves, that's **n × n**. Adding it never changes the shape of the scores.

In real code, the **stored** mask is often smaller and is automatically repeated (**broadcast**) across the batch, the attention heads or the query rows. For example, a padding mask only needs one row per text, because every query blocks the same columns.

## 5. The mask types, one by one

### 5.1 No mask (full attention)

**What it does:** every token sees every token, before and after it. There's no causal or local restriction, although padding may still be masked (Section 5.3).

```
✓ ✓ ✓ ✓
✓ ✓ ✓ ✓
✓ ✓ ✓ ✓
✓ ✓ ✓ ✓
```

**Why:** when the whole input is already known and nothing is being generated, there's nothing to hide. Each token can use context from both sides.

**Where it's used:** encoders like BERT, and the [Vision Transformer](vit.md), where all image patches are available at once.

---

### 5.2 Causal mask (look-ahead mask)

**What it does:** each token sees **itself and the tokens before it**, never the ones after.

```
✓ · · ·
✓ ✓ · ·
✓ ✓ ✓ ·
✓ ✓ ✓ ✓
```

The allowed cells form a triangle: everything on or below the diagonal.

**Why:** a text generator writes one token at a time (see [LLM Inference](llm_inference.md)). When it's really generating, the following tokens don't exist yet. During training, the whole text **is** available, but the model is learning to predict each next token, so the tokens after each position must be **hidden** from it. The mask does that, so the model can't learn to cheat by peeking ahead.

**Where it's used:** GPT-style language models (decoders).

**Picture it:** reading a book with a sheet of paper covering everything after the current word.

---

### 5.3 Padding mask

**What it does:** blocks the **padding tokens**. In this example, token 4 is padding:

```
✓ ✓ ✓ ·
✓ ✓ ✓ ·
✓ ✓ ✓ ·
✓ ✓ ✓ ·
```

It blocks a whole **column**: no token may look at the padding.

**Why:** texts in a batch have different lengths, but they're processed together as one rectangular block. Short texts are filled up with padding tokens to match the longest one. Padding is just a placeholder for batching, not real content, so nobody should pay attention to it.

```
"I love cats"      →  [I, love, cats]
"Hi"               →  [Hi, PAD, PAD]     ← filled up to length 3
```

**Note:** the padding mask blocks padding **keys** (columns). The padding token's own **row** still gets computed, but its output is simply ignored later, for example by leaving it out of the loss.

**Where it's used:** almost every model that processes batches.

---

### 5.4 Sliding-window mask (local attention)

**What it does:** each token sees only its **nearest previous tokens**. With a window of 2, each token sees itself and the one before:

```
✓ · · ·
✓ ✓ · ·
· ✓ ✓ ·
· · ✓ ✓
```

**Why:** full attention compares every token with every token, so the number of pairs grows like n × n. For 100,000 tokens, that's 10 billion pairs. With a fixed window of size w, the number of pairs grows only like n × w. (In math notation: O(n·w) instead of O(n²).) This saving only happens if the code actually skips the blocked pairs. Building a full n × n mask and still computing every pair gives the same result, but no speed-up.

**Example:** token 4 with scores [1, 2, 0, 3] and a window of 2 sees only tokens 3 and 4:

```
scores + mask = [−∞, −∞, 0, 3]
e^(...)       = [0, 0, 1.000, 20.086]
sum           = 21.086
weights       = [0, 0, 0.047, 0.953]
```

**Doesn't the model lose long-range information?** Not completely: information can travel farther through stacked layers. With a causal window of size w (itself plus the w − 1 tokens before it), each layer can extend the reach by up to **w − 1** earlier positions. After L layers, information may travel up to **L·(w − 1)** positions back.

Example with w = 2: in layer 1, token 4 sees token 3. But in layer 1, token 3 already saw token 2. So in layer 2, token 4 indirectly gets information from token 2. That's 2 · (2 − 1) = 2 positions back. This is the *possible* reach; it doesn't guarantee the model actually uses all that information.

**Where it's used:** Longformer and Mistral 7B use forms of sliding-window attention.

**Picture it:** looking through a small window that slides along the text.

---

### 5.5 Sparse masks with global tokens

**What it does:** some sparse-attention designs combine local windows with a few **global** tokens. In the example below, token 1 is global: it can attend to every token, and every token can attend to it. The others see their neighbours on both sides. Other designs use different connection patterns, sometimes including random connections.

```
✓ ✓ ✓ ✓      ← token 1 (global) sees everyone
✓ ✓ ✓ ·      ← token 2 sees global token 1, itself, and neighbour 3
✓ ✓ ✓ ✓      ← token 3 sees global token 1, itself, and neighbours 2, 4
✓ · ✓ ✓      ← token 4 sees global token 1, itself, and neighbour 3
```

**Why:** local windows are cheap but short-sighted. A few global tokens act like **messengers**: they collect information from the whole text and share it with everyone. Some designs also add a few **random** connections.

**Where it's used:** Longformer (window + global) and BigBird (window + global + random).

**Picture it:** a classroom where students only talk to their neighbours, except the class representative, who talks to everyone.

---

### 5.6 Prefix-LM mask

**What it does:** the first part of the text (the **prefix**, such as a question or instruction) sees itself fully, in both directions. The rest, the generated part, is causal. Here tokens 1–2 are the prefix:

```
✓ ✓ · ·
✓ ✓ · ·
✓ ✓ ✓ ·
✓ ✓ ✓ ✓
```

Compare with the causal mask: the **prefix–prefix** region (top-left 2 × 2 block) is now bidirectional, so token 1 can also see token 2. The rest (the generated part) stays causal.

**Why:** the prefix is fully given in advance, so there's no reason to hide parts of it from itself. Letting it look both ways can help the model understand the input better. The answer is still generated one token at a time, so that part stays causal.

**Where it's used:** some language models trained as "prefix language models".

---

### 5.7 Block-diagonal mask (document mask)

**What it does:** when several short documents are packed into one long sequence, each token sees only tokens from **its own document**. Here tokens 1–2 are document A and tokens 3–4 are document B, combined with causal:

```
✓ · · ·
✓ ✓ · ·
· · ✓ ·
· · ✓ ✓
```

The allowed cells form separate blocks along the diagonal, one per document.

**Why:** packing documents end to end avoids wasting space on padding. But a sentence from document B shouldn't get information from an unrelated document A. This mask keeps them separate.

**Where it's used:** training large models on packed sequences.

---

### 5.8 Cross-attention mask

**What it does:** in **encoder–decoder** models (e.g. translation), the decoder's tokens look at the **encoder's** tokens. The queries come from one sequence and the keys from another, so the mask is **not square**.

Example: 3 decoder tokens looking at 4 encoder tokens, where encoder token 4 is padding:

```
                 enc 1  enc 2  enc 3  enc 4
decoder token 1    ✓      ✓      ✓      ·
decoder token 2    ✓      ✓      ✓      ·
decoder token 3    ✓      ✓      ✓      ·
```

Shape: **3 × 4** (decoder length × encoder length).

**Why:** the decoder may see the **whole** input sentence; it's already known. So a causal mask is generally unnecessary here, although padding in the input may still need to be masked. The causal mask is applied in the decoder's **self**-attention instead.

**Where it's used:** translation and other encoder–decoder models, like the original Transformer and T5.

## 6. Combining masks, and common mistakes

### Combining masks

Real models often use several masks at once. The rule: **a cell is allowed only if every mask allows it** (the intersection of the allowed cells). With the −∞ trick, adding the masks does this: any −∞ wins, so a blocked cell stays blocked. Code that uses true/false masks combines them with a logical AND instead.

Example: **causal + padding**, where token 4 is padding:

```
   causal          padding          combined
✓ · · ·          ✓ ✓ ✓ ·          ✓ · · ·
✓ ✓ · ·    +     ✓ ✓ ✓ ·    =     ✓ ✓ · ·
✓ ✓ ✓ ·          ✓ ✓ ✓ ·          ✓ ✓ ✓ ·
✓ ✓ ✓ ✓          ✓ ✓ ✓ ·          ✓ ✓ ✓ ·
```

Only the last row changed: token 4 can no longer look at itself, because it's padding.

### Related ideas that are *not* attention masks

- **Soft masks (position biases).** Instead of blocking with −∞, some methods add a **penalty** that grows with distance, such as −1 for one step away and −2 for two steps away. Far tokens are **discouraged**, not forbidden. ALiBi works this way.
- **Input masking.** In BERT's training, some input words are replaced with a `[MASK]` token, and the model learns to guess them. That hides **words in the input**; it doesn't change the attention table. Same word "mask", different idea.

Three kinds of "mask" are easy to mix up:

| Kind | What it controls | Example |
|---|---|---|
| **Attention mask** | which keys each query may use to compute its attention output | causal, padding |
| **Loss mask** | which positions count toward the training loss | ignoring the outputs at padding positions |
| **Input masking** | which input tokens are replaced or hidden | BERT's `[MASK]` |

### Common mistakes

- **Blocking with 0 instead of −∞.** Adding 0 changes nothing. The score must become −∞ so that e^score = 0 (or, in practice, a very large negative number, so the weight is negligible).
- **Applying the mask after softmax without renormalizing.** Setting blocked weights to 0 after softmax generally leaves the remaining weights adding up to less than 1. Masking the scores **before** softmax automatically makes the allowed weights add up to 1.
- **Blocking a whole row.** If every cell in a row is blocked, softmax has nothing to share and divides 0 by 0. Make sure each query can see at least one key (usually itself).
- **Mixing up rows and columns.** Rows are the tokens **looking**; columns are the tokens **being looked at**. The padding mask blocks **columns**.
- **Confusing attention masks with `[MASK]` tokens.** See above: two different ideas.

## 7. Remember this

- A mask says, for each pair of tokens, **allowed or blocked**.
- Conceptually, a mask blocks selected query–key pairs **before softmax**. In the additive version, **0** leaves a score unchanged and **−∞** blocks it, giving a weight of exactly 0. Implementations may instead use true/false (Boolean) masks or large finite negative values.
- Its pattern lines up with the score matrix (one row per query, one column per key), and it never changes the shape of the scores.
- Combine masks by adding them: a cell is allowed only if **all** masks allow it.

| Mask | Rule in one line | Main reason |
|---|---|---|
| None | everyone sees everyone | nothing to hide |
| Causal | see only yourself and earlier tokens | no peeking at the future |
| Padding | never look at padding | padding is only a placeholder |
| Sliding window | see only the last few tokens | save computation |
| Sparse + global | neighbours, plus a few tokens that see all | long texts, cheaply |
| Prefix-LM | prefix sees both ways, answer is causal | prompt is fully known |
| Block-diagonal | see only your own document | packed documents stay separate |
| Cross-attention | decoder sees the encoder, minus padding | input is fully known |

### Quick check

1. Draw the causal mask for 3 tokens.
2. Scores are [2, 0, 1] and the mask is [0, −∞, 0]. What are the attention weights?
3. Why must blocked cells be −∞ and not 0?
4. A batch has the texts "Good morning everyone" (3 tokens) and "Hi" (1 token). Which tokens does the padding mask block for "Hi"?

Try answering before reading on.

---

### Check your answers

1. Causal mask for 3 tokens:
   ```
   ✓ · ·
   ✓ ✓ ·
   ✓ ✓ ✓
   ```
2. Step by step:
   ```
   scores + mask = [2, −∞, 1]
   e^(...)       = [7.389, 0, 2.718]
   sum           = 7.389 + 0 + 2.718 = 10.107
   weights       = [7.389/10.107, 0, 2.718/10.107] = [0.731, 0, 0.269]
   ```
3. The mask is **added** to the scores. Adding 0 changes nothing, so the token would still get attention. Adding −∞ makes e^score = 0, so the weight becomes exactly 0.
4. "Hi" is padded to 3 tokens: [Hi, PAD, PAD]. The padding mask blocks the **two PAD columns**, so every token looks only at "Hi".

## 8. Going further

- **[Attention Blocks](attention_blocks.md)**: the full attention calculation, including a causal mask worked out with real numbers.
- **[LLM Inference](llm_inference.md)**: why text generators need the causal mask, and how the KV cache relies on it.
- **[Vision Transformer (ViT)](vit.md)**: an example of attention with no mask at all.

### Learn more

- 📖 [Longformer: The Long-Document Transformer](https://arxiv.org/abs/2004.05150): sliding-window and global attention
- 📖 [Big Bird: Transformers for Longer Sequences](https://arxiv.org/abs/2007.14062): window, global and random attention
- 📖 [Mistral 7B](https://arxiv.org/abs/2310.06825): a language model that uses sliding-window attention
- 📖 [Train Short, Test Long (ALiBi)](https://arxiv.org/abs/2108.12409): distance penalties instead of position embeddings
