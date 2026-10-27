# Attention Blocks: A Shape-Focused Walkthrough

## 1. What an attention block does

An attention block lets each token gather information from other tokens by computing attention weights over their value vectors. Masks can restrict which tokens it attends to.

This walkthrough follows **one attention head**. Section 8 explains how multiple heads are combined.

Each token produces three vectors:

- a **query**: what this token is looking for
- a **key**: what this token offers to others
- a **value**: the information this token carries

Each token compares its query with the keys of the tokens it may attend to, turns those scores into weights, and takes a weighted mix of their values.

## 2. Prerequisites

### Matrix multiplication

For a matrix product (a × **b**)(**b** × c) → a × c:

- The inner dimensions must match.
- The inner dimension is **contracted**: it is summed over and does not appear in the result.
- The outer dimensions are what's left.

Every step in this walkthrough applies this one rule.

To compute one entry, multiply a **row** of the left matrix with a **column** of the right matrix, element by element, and add up the products.

### Softmax

Softmax converts scores into nonnegative weights that sum to 1. For finite inputs, each weight is positive. Masked positions (Section 6) can receive zero weight.

```
softmax(x)ᵢ = e^(xᵢ) / Σⱼ e^(xⱼ)
```

Larger inputs get larger weights. Example: softmax([1, 0]) = [2.718, 1] / 3.718 = [0.731, 0.269].

## 3. The four dimensions

The four dimensions have different values in this example, which makes it easy to track how each operation changes the shape.

| Symbol | Value | Meaning | Where it appears |
|---|---|---|---|
| **n** | 3 | number of tokens | the **rows** of every activation matrix |
| **d** | 5 | model width | input X and final output |
| **d_k** | 4 | query/key width | only Q and K |
| **d_v** | 2 | value width | V and the head output |

Every activation matrix has n rows, one per token. As data moves through the block, only the number of columns changes. The score matrices S and A are special because their columns also index tokens, so they are n × n.

## 4. The core idea and shape flow

- **Q and K:** compute pairwise token scores (n × n).
- **Softmax:** converts the scores into attention weights, applying any mask.
- **V:** combines token information using those weights (n × d_v).
- **Output projection:** maps the head output back to width d (n × d).

```
               W_Q (5×4)
X (3×5) ─────────────────▶ Q (3×4) ─┐
               W_K (5×4)            ├─ Q·Kᵀ ─▶ S (3×3) ─(mask)─softmax─▶ A (3×3) ─┐
X (3×5) ─────────────────▶ K (3×4) ─┘                                             │
               W_V (5×2)                                                          ├─ A·V ─▶ head (3×2) ─W_O (2×5)─▶ out (3×5)
X (3×5) ────────────────────────────────────────────────────▶ V (3×2) ────────────┘
```

| Step | Product | Contracted dim | Result | Each row means |
|---|---|---|---|---|
| Q = X W_Q | (n×d)(d×d_k) | d | n × d_k | a token's query |
| K = X W_K | (n×d)(d×d_k) | d | n × d_k | a token's key |
| V = X W_V | (n×d)(d×d_v) | d | n × d_v | a token's value |
| S = Q Kᵀ / √d_k | (n×d_k)(d_k×n) | d_k | n × n | one token's score for every token |
| A = softmax(S) | row-wise | none | n × n | attention weights, summing to 1 |
| head = A V | (n×n)(n×d_v) | n | n × d_v | a weighted mix of values |
| out = head W_O | (n×d_v)(d_v×d) | d_v | n × d | back to model width |

What the table shows:

- **d_k is contracted** in Q Kᵀ. That's why Q and K must have the same width, and why d_k never appears in the output.
- **S is always n × n**, whatever d and d_k are. S[i, j] is the raw attention score between tokens i and j. A[i, j] is the corresponding normalized attention weight: the coefficient applied to token j's value vector when computing token i's head output.
- **The token dimension n is contracted in A V**: attention combines the n value vectors into one output vector for each token.
- **d_v doesn't depend on d_k.** V only needs n rows so they line up with A's columns.
- **W_O brings the width back to d**, so out has the same shape as X and the residual X + out works.

## 5. Worked example

> **Note:** The weights below are chosen to keep the arithmetic easy. In a real model they are learned during training, so don't look for linguistic meaning in these particular numbers. The point is to follow the shapes and the mechanics.
>
> **Rounding:** Attention weights are shown to four decimal places, and final results to three. Small discrepancies may occur when calculations use rounded intermediate values.

### Input: X, n × d = 3 × 5

```
        d=5 →
X = [[1, 0, 1, 0, 0],    ← token 1
     [0, 1, 0, 1, 0],    ← token 2
     [1, 1, 0, 0, 1]]    ← token 3
```

### Weights

```
W_Q (d×d_k = 5×4)   W_K (d×d_k = 5×4)   W_V (d×d_v = 5×2)   W_O (d_v×d = 2×5)
[[1,0,0,0],         [[1,0,0,0],         [[1,0],             [[1,0,1,0,1],
 [0,1,0,0],          [0,1,0,0],          [0,1],              [0,1,1,1,0]]
 [0,0,1,0],          [1,0,0,0],          [1,0],
 [0,0,0,1],          [0,1,0,0],          [0,1],
 [0,0,0,0]]          [0,0,1,1]]          [1,1]]
```

A weight matrix's first dimension is the width of what comes in, and its second is the width of what goes out. None of them depend on n, so the same weights work for any sequence length.

### Step 1: Projections

**Q = X W_Q**: (3×**5**)(**5**×4) → **3×4**

Full calculation for token 1, x = [1, 0, 1, 0, 0], times each column of W_Q:

```
column 1: 1·1 + 0·0 + 1·0 + 0·0 + 0·0 = 1
column 2: 1·0 + 0·1 + 1·0 + 0·0 + 0·0 = 0
column 3: 1·0 + 0·0 + 1·1 + 0·0 + 0·0 = 1
column 4: 1·0 + 0·0 + 1·0 + 0·1 + 0·0 = 0
→ [1, 0, 1, 0]
```

**Shortcut:** because X contains only 0s and 1s, multiplying a row of X by a matrix just **adds up the matrix rows where X has a 1**. Token 1 has 1s in positions 1 and 3, so it selects rows 1 and 3 of W_Q: [1,0,0,0] + [0,0,1,0] = [1,0,1,0]. Same result. The remaining projections use this shortcut.

```
Q = [[1, 0, 1, 0],
     [0, 1, 0, 1],
     [1, 1, 0, 0]]    (3×4)
```

**K = X W_K**: (3×**5**)(**5**×4) → **3×4**

- Token 1 selects rows 1 and 3 of W_K: [1,0,0,0] + [1,0,0,0] = [2,0,0,0]
- Token 2 selects rows 2 and 4: [0,1,0,0] + [0,1,0,0] = [0,2,0,0]
- Token 3 selects rows 1, 2 and 5: [1,0,0,0] + [0,1,0,0] + [0,0,1,1] = [1,1,1,1]

```
K = [[2, 0, 0, 0],
     [0, 2, 0, 0],
     [1, 1, 1, 1]]    (3×4)
```

**V = X W_V**: (3×**5**)(**5**×2) → **3×2**

- Token 1 selects rows 1 and 3: [1,0] + [1,0] = [2,0]
- Token 2 selects rows 2 and 4: [0,1] + [0,1] = [0,2]
- Token 3 selects rows 1, 2 and 5: [1,0] + [0,1] + [1,1] = [2,2]

```
V = [[2, 0],
     [0, 2],
     [2, 2]]    (3×2)
```

Each token now has a 4-dim query, a 4-dim key and a 2-dim value.

### Step 2: Scores

**Kᵀ** turns 3×4 into 4×3, so the inner dimensions line up.

**Q Kᵀ**: (3×**4**)(**4**×3) → **3×3**. Each entry is the dot product of a query row with a key row. For example, query 1 · key 1 = [1,0,1,0]·[2,0,0,0] = 2.

```
            key1  key2  key3
query1  [[  2,    0,    2 ],
query2   [  0,    2,    2 ],
query3   [  2,    2,    2 ]]
```

Reading the scores:

- Token 3's key [1,1,1,1] overlaps equally with every query, so column 3 is all 2s.
- Token 1's key [2,0,0,0] only matches queries with a 1 in the first position (tokens 1 and 3).
- Token 2's key [0,2,0,0] only matches queries with a 1 in the second position (tokens 2 and 3).

Divide by √d_k = √4 = 2:

```
S = [[1, 0, 1],
     [0, 1, 1],
     [1, 1, 1]]    (3×3)
```

Dividing by √d_k keeps the dot products from growing so large that the softmax saturates (puts almost all weight on one token).

### Step 3: Softmax, row by row

The shape stays **3×3**. Use e¹ ≈ 2.7183 and e⁰ = 1.

- Row 1: [2.7183, 1, 2.7183] / 6.4366 → [0.4223, 0.1554, 0.4223]
- Row 2: [1, 2.7183, 2.7183] / 6.4366 → [0.1554, 0.4223, 0.4223]
- Row 3: [2.7183, 2.7183, 2.7183] / 8.1548 → [0.3333, 0.3333, 0.3333]

```
A = [[0.4223, 0.1554, 0.4223],
     [0.1554, 0.4223, 0.4223],
     [0.3333, 0.3333, 0.3333]]    (3×3), each row sums to 1
```

Token 3's scores were all equal, so it spreads its attention evenly.

### Step 4: Mix the values

**head = A V**: (3×**3**)(**3**×2) → **3×2**

The token dimension n is contracted here. Each output row is a weighted sum of the 3 tokens' value vectors, using that token's row of A as the weights:

- Token 1: 0.4223·[2,0] + 0.1554·[0,2] + 0.4223·[2,2] = [0.8446 + 0.8446, 0.3108 + 0.8446] = [1.689, 1.155]
- Token 2: 0.1554·[2,0] + 0.4223·[0,2] + 0.4223·[2,2] = [0.3108 + 0.8446, 0.8446 + 0.8446] = [1.155, 1.689]
- Token 3: 0.3333·([2,0] + [0,2] + [2,2]) = 0.3333·[4,4] = [1.333, 1.333]

```
head = [[1.689, 1.155],
        [1.155, 1.689],
        [1.333, 1.333]]    (3×2)
```

The output is d_v = 2 wide because it is built from values.

### Step 5: Project back to d

**out = head W_O**: (3×**2**)(**2**×5) → **3×5**

Each output column is a weighted combination of the two components h₁ and h₂ of a head-output row. With this W_O, the five columns are [h₁, h₂, h₁+h₂, h₂, h₁]:

```
out = [[1.689, 1.155, 2.845, 1.155, 1.689],
       [1.155, 1.689, 2.845, 1.689, 1.155],
       [1.333, 1.333, 2.667, 1.333, 1.333]]    (3×5)
```

(2.845 = 1.6893 + 1.1554, computed before rounding.)

This has the same shape as X, so X + out is valid, and the next block receives a 3 × 5 input just as this one did.

## 6. Masked attention

### Causal mask

In autoregressive models such as GPT, a token must not attend to future positions. The causal mask prevents information leakage from tokens that have not yet been generated.

The mask M is an n × n matrix that is **added to the scores before the softmax**. Because M has the same shape as S, no shapes change anywhere in the block. Positions on or below the diagonal get 0, which keeps them. Positions above the diagonal get −∞, which blocks them.

```
          key1  key2  key3
M  = [[   0,   −∞,   −∞ ],    ← token 1 sees only token 1
      [   0,    0,   −∞ ],    ← token 2 sees tokens 1–2
      [   0,    0,    0 ]]    ← token 3 sees all
```

Only three things change compared with Section 5: the scores, the attention weights and the head output.

**Masked scores**

```
S + M = [[1,  −∞, −∞],
         [0,   1, −∞],
         [1,   1,  1]]
```

**Masked attention weights** (using e^(−∞) = 0)

- Row 1: [2.7183, 0, 0] / 2.7183 → [1, 0, 0]
- Row 2: [1, 2.7183, 0] / 3.7183 → [0.2689, 0.7311, 0]
- Row 3: unchanged, [0.3333, 0.3333, 0.3333]

```
A_masked = [[1.0000, 0.0000, 0.0000],
            [0.2689, 0.7311, 0.0000],
            [0.3333, 0.3333, 0.3333]]    lower-triangular, each row sums to 1
```

Blocked positions get exactly zero weight, and the remaining weights in each row still sum to 1.

**Masked head output**

- Token 1: 1·[2,0] = [2.000, 0.000]
- Token 2: 0.2689·[2,0] + 0.7311·[0,2] = [0.538, 1.462]
- Token 3: [1.333, 1.333], unchanged

| Token | Unmasked head | Masked head | Why |
|---|---|---|---|
| 1 | [1.689, 1.155] | [2.000, 0.000] | now sees only itself |
| 2 | [1.155, 1.689] | [0.538, 1.462] | can no longer see token 3 |
| 3 | [1.333, 1.333] | [1.333, 1.333] | already the last token, so nothing changes |

The output projection (Step 5) works exactly as before on this new head.

### Padding mask

Batches often contain sequences of different lengths, so shorter ones are filled with padding tokens. A **padding mask** puts −∞ in the key columns of padded positions, so that valid queries assign them zero weight. This masks padded *keys* only. The rows for padded *query* positions still produce outputs, and implementations handle those separately, for example by ignoring them in the loss.

## 7. Summary

```
3×5  →  Q, K: 3×4  →  S, A: 3×3  →  head: 3×2  →  out: 3×5
 n×d      n×d_k          n×n          n×d_v         n×d
```

- **n** tracks tokens and stays as the rows throughout.
- **d_k** determines the score computation and is contracted there.
- **d_v** determines the width of the head output, and W_O maps it back to d.

The conceptual backbone:

| Quantity | Meaning | Shape |
|---|---|---|
| S | raw scores | n × n |
| A | normalized attention weights | n × n |
| A V | weighted value combinations | n × d_v |
| A V W_O | output projected back to model width | n × d |

## 8. Beyond the example

The sections below have no worked numbers. They show how the example fits into a real model.

### Multi-head attention

With h heads, steps 1–4 run h times in parallel. Each head has its own W_Q, W_K and W_V and produces an n × d_v matrix. In this document, **MultiHead** means the complete operation: all heads, concatenation, and the output projection W_O.

| Step | Shape |
|---|---|
| each head_i | n × d_v |
| Concat(head_1, …, head_h) | n × (h·d_v) |
| W_O | (h·d_v) × d |
| MultiHead output | n × d |

Different heads can learn to focus on different relationships. Many Transformer architectures use d_k = d_v = d/h, but these dimensions are design choices and do not have to be equal.

### The full transformer block

Attention is wrapped with layer normalization, residual connections and a feed-forward network (pre-norm form shown):

```
X₁ = X  + MultiHead(LayerNorm(X))      n × d
X₂ = X₁ + FFN(LayerNorm(X₁))           n × d
```

| Part | Shapes |
|---|---|
| LayerNorm | n × d → n × d (normalizes each row) |
| FFN(x) = max(0, x W₁ + b₁) W₂ + b₂ | W₁: d × d_ff, W₂: d_ff × d |
| FFN flow | n × d → n × d_ff → n × d |

- **Residual connections** add the input back to each sublayer's output, which keeps gradients flowing through deep stacks.
- **Layer normalization** keeps values at a stable scale.
- **The feed-forward network** processes each token independently. Attention moves information between tokens, and the FFN computes on each one.

d_ff is usually 4d. The block's input and output are both n × d, which is what lets you stack blocks.

### Batching

Real implementations add a leading batch dimension b:

- X: b × n × d
- S, M and A: b × n × n (or b × h × n × n with heads)
- the same weight matrices are shared across the whole batch

### Positional information

Attention on its own ignores token order: if you shuffle the rows of X, the output rows shuffle the same way. Models supply order by adding positional encodings to X or by rotating Q and K (RoPE). The shapes don't change.
