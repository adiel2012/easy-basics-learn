# LLM Inference: How a Language Model Writes

> **The one idea in this guide:** an LLM repeatedly predicts **one next token**, adds it to the text, and repeats until it stops.
>
> Everything below explains, demonstrates or deepens that idea.

## 1. In simple words

**Inference** is what happens when you *use* a trained language model: you give it some text, and it writes a reply.

An LLM writes **one token at a time**:

```
"I love"        →  predict next token  →  "cats"
"I love cats"   →  predict next token  →  [EOS]  →  stop

Result: "I love cats"
```

**[EOS]** means "end of sequence". It's a special token the model produces when it's finished.

**Analogy: phone autocomplete, on repeat.** Your phone suggests the next word as you type. Imagine always tapping the suggestion, then letting the phone suggest the next one, and the next. That's an LLM writing. The difference is that an LLM reads *everything* written so far, not just the last word, before each suggestion.

That's the whole process. The rest of this guide looks inside the "predict next token" box.

> **Quick check 1.** The model has written "I love cats" and now predicts [EOS]. What happens next?
> (a) It predicts another token. (b) It stops generating. (c) It updates its weights.
>
> *Answers to all checks are in Section 7.*

## 2. What you need to know first

- **[Tokenization](tokenization.md)**: how text becomes token IDs and embeddings.
- **[Attention Blocks](attention_blocks.md)**: how tokens share information. This guide explains the parts it uses as it goes, so you can come back to the attention guide later.

## 3. How one token is predicted

Each prediction follows the same path:

```
text → token IDs → embeddings → transformer → last row → logits → probabilities → chosen token
```

Each stage in one or two sentences:

1. **Token IDs and embeddings.** The text is split into tokens, and each token is turned into a vector (a list of numbers). See [Tokenization](tokenization.md).
2. **Transformer.** The vectors pass through the model's layers. Inside, **attention** lets each token gather information from the tokens before it.
3. **Last row.** Only the vector of the **last token** is used for the prediction. It has seen the whole text, so it carries everything needed to guess what comes next.
4. **Logits: raw scores.** The model gives a score to **every** token in its vocabulary. A higher score means the model prefers that token.
5. **Probabilities.** **Softmax** turns the scores into probabilities between 0 and 1 that add up to 1.
6. **Chosen token.** A **decoding rule** picks one token. The simplest is **greedy decoding**: pick the most likely token.

> **Key distinction:** logits are *scores*. Softmax turns them into *probabilities*. Decoding *chooses* a token. These are three different stages.

## 4. A tiny LLM in action

> **This is a toy.** The model below has made-up weights, one attention head and a 5-token vocabulary. It shows the *mechanics* of inference, not how a real trained model behaves. All arithmetic is in the [Appendix](#appendix-full-arithmetic) if you want to check it.

### The model

**Vocabulary:** `I`, `love`, `cats`, `dogs`, `[EOS]`

**Embeddings:** each token is a vector of 2 numbers:

| Token | Embedding |
|---|---|
| I | [1, 0] |
| love | [0, 1] |
| cats | [1, 1] |
| dogs | [−1, 1] |
| [EOS] | [2, 0.5] |

**What the model does with them:**

1. **Attention:** the last token compares itself with every token so far (including itself) and takes a weighted mix of them. To keep it simple, each token's query, key and value are just its embedding.
2. **Residual:** the mix is added to the last token's own embedding. Call the result **h**.
3. **Scores:** each token's logit is the dot product of h with that token's embedding. A token whose embedding "points the same way" as h gets a high score.

### Step 1: what comes after "I love"?

**Goal:** score every possible next token, turn the scores into probabilities, and pick the most likely one.

**Attention.** The last token is "love". It attends to both tokens:

| Token | Attention weight |
|---|---|
| I | 0.330 |
| love | 0.670 |

**Mix and add (residual):**

```
mix = 0.330·[1, 0] + 0.670·[0, 1] = [0.330, 0.670]
h   = [0, 1] + [0.330, 0.670]     = [0.330, 1.670]
```

**Score every token (h · embedding), then softmax:**

| Token | Logit (score) | Probability |
|---|---|---|
| I | 0.330 | 0.062 |
| love | 1.670 | 0.237 |
| **cats** | **2.000** | **0.330** ← highest |
| dogs | 1.340 | 0.171 |
| [EOS] | 1.495 | 0.199 |

**Choose (greedy):** **cats**. The text becomes **"I love cats"**.

### Step 2: what comes after "I love cats"?

The model repeats **exactly the same process**. Only one thing changed: **the input now includes the token it just generated.**

The last token is now "cats". It attends to all three tokens, with weights I 0.248, love 0.248 and cats 0.503. That gives h = [1.751, 1.751].

| Token | Logit | Probability |
|---|---|---|
| I | 1.751 | 0.046 |
| love | 1.751 | 0.046 |
| cats | 3.502 | 0.265 |
| dogs | 0.000 | 0.008 |
| **[EOS]** | **4.378** | **0.635** ← highest |

**Choose (greedy):** **[EOS]**. The model stops.

### Result

```
Output: "I love cats"
```

The [EOS] token isn't shown; it only signals the stop.

> **Quick check 2.** In Step 1, which row of the transformer's output was used to score the next token, and why?

## 5. How the next token is chosen

Softmax gives probabilities. A **decoding rule** decides how to use them.

### Greedy decoding

- **What it is:** always pick the most likely token.
- **Why it matters:** it's simple and predictable: the same prompt always gives the same answer.
- **Example:** in Step 1, greedy picks "cats" (0.330).

### Sampling

- **What it is:** pick a token at random, following the probabilities.
- **Why it matters:** it adds variety, so answers sound less repetitive. This is why the same prompt can give different answers.
- **Example:** in Step 1, "cats" would be picked 33% of the time, "love" 24%, and [EOS] 20%. If [EOS] were picked, the answer would simply be "I love".

### Temperature

- **What it is:** a knob that reshapes the probabilities before sampling. The logits are divided by the temperature T before softmax.
- **Why it matters:** it controls how adventurous the model is.
- **Example:** the Step 1 probability of "cats":

| Temperature | Effect | P(cats) |
|---|---|---|
| T = 0.5 | sharper: the top token wins more often | 0.458 |
| T = 1 | unchanged | 0.330 |
| T = 2 | flatter: unlikely tokens get more chances | 0.265 |

### Optional: top-k and top-p

Both **limit which tokens can be sampled**, then renormalize the probabilities of the ones that remain so they sum to 1:

- **Top-k:** keep only the k most likely tokens. With k = 2 in Step 1, only "cats" and "love" can be picked.
- **Top-p:** keep the smallest set of top tokens whose probabilities add up to at least p (for example 0.9).

The full numbers are in the [Appendix](#appendix-full-arithmetic).

> **Quick check 3.** Probabilities are cats 0.6 and dogs 0.4. What does greedy decoding choose? What might sampling choose?

## 6. How inference becomes efficient

### Two phases: prefill and decode

- **Prefill:** the model reads the **whole prompt at once**. It computes all prompt positions in parallel, and the **last position** supplies the logits for the first new token.
- **Decode:** the model then generates **one token at a time**. Each step depends on the token chosen just before, so these steps can't run in parallel.

That's why a long prompt is read quickly, but the answer appears word by word.

### The KV cache

- **What it is:** a saved copy of each token's **keys and values**, the vectors that later tokens attend to.
- **Why it matters:** without it, every decode step would recompute them for the whole text.
- **Example:** in Step 2, "cats" attended to "I" and "love". Their keys and values were already computed in Step 1, and they don't change, because earlier tokens never look at later ones. So only "cats" needs new ones.

| Step | Without cache: compute keys/values for | With cache |
|---|---|---|
| 1 (prefill) | I, love | I, love (saved) |
| 2 (decode) | I, love, cats | cats only |
| 100 | all 101 tokens | 1 new token |

The cost is memory: the cache grows with every token.

> **Quick check 4.** At Step 2, which keys and values come from the cache, and which must be computed?

## 7. Remember this

### Five essential ideas

1. **One token at a time.** An LLM loops: read → score → choose → append → repeat, until [EOS] or a length limit. It never writes a whole sentence at once.
2. **Only the last row predicts.** The other rows still matter: they provide the keys and values the last token attends to.
3. **Scores → probabilities → choice.** Logits are scores, softmax makes probabilities, and decoding chooses.
4. **Greedy is fixed; sampling varies.** Temperature, top-k and top-p shape the sampling.
5. **Nothing is learned during inference.** The weights are fixed. The KV cache only saves computation, and the context window limits how many tokens fit.

### Exercises

1. Using the Step 2 table, which token would greedy decoding pick if [EOS] didn't exist?
2. Logits are [2, 2]. What are the probabilities after softmax?
3. What happens to sampling if the temperature is very low?

### Answers

**Quick checks**

1. **(b)** It stops generating. [EOS] is the stop signal. The weights never change during inference.
2. **The last row**, for "love". It's the last token, so it has seen the whole text.
3. **Greedy always chooses cats**, the highest probability. **Sampling** picks cats about 60% of the time and dogs about 40%.
4. **"I" and "love" come from the cache** (computed in Step 1). Only **"cats"** needs new keys and values.

**Exercises**

1. **cats** (logit 3.502, the next highest).
2. **[0.5, 0.5].** Equal scores give equal probabilities.
3. **It becomes almost greedy.** The top token's probability approaches 1.

## 8. Going further

### How real LLMs differ from the toy

| Part | Toy model | Real LLM |
|---|---|---|
| Vocabulary | 5 tokens | tens of thousands to hundreds of thousands |
| Vector width | 2 | thousands |
| Layers and heads | 1 layer, 1 head | dozens of layers, many heads each |
| Weights | made up | learned during training |
| Feed-forward, normalization, positions | left out | included |

The loop is exactly the same.

### What a chat model adds

Chat assistants run the same loop. The conversation is turned into one long text with special tokens marking who is speaking, something like:

```
[user] What do cats like? [assistant] Cats like ...
```

The model continues the text after the assistant marker, one token at a time, until its end token.

### Where this connects

- **[Tokenization](tokenization.md)**: the first step (text → IDs) and the last (IDs → text).
- **[Attention Blocks](attention_blocks.md)**: what happens inside each layer, including the causal mask.
- **[Vision Transformer (ViT)](vit.md)**: the same transformer used on images. ViT classifies in one pass, with no generation loop.

### Short glossary

| Term | Meaning |
|---|---|
| Inference | using a trained model to produce output |
| Token | a piece of text the model works with |
| [EOS] | the "I'm done" token |
| Logits | raw scores, one per vocabulary token |
| Softmax | turns scores into probabilities |
| Greedy decoding | pick the most likely token |
| Sampling | pick randomly, following the probabilities |
| Temperature | sharpens (low) or flattens (high) the probabilities |
| Top-k / top-p | limit sampling to the most likely tokens |
| Prefill / decode | read the prompt at once / generate one token at a time |
| KV cache | saved keys and values from earlier tokens |
| Context window | the maximum number of tokens the model can handle |

### Learn more

- 📖 [The Illustrated GPT-2](https://jalammar.github.io/illustrated-gpt2/) (Jay Alammar): a visual walk-through of how GPT-2 generates text
- 📖 [nanoGPT README](https://github.com/karpathy/nanoGPT/blob/master/README.md) (Andrej Karpathy): a small, readable GPT you can train and sample from
- 📖 [Generation strategies](https://huggingface.co/docs/transformers/main/en/generation_strategies) (Hugging Face docs): greedy search, sampling and other ways to pick the next token, with code

---

## Appendix: full arithmetic

*Optional. For readers who want to verify every number. Values are rounded to 3 decimals, and each step uses the rounded values shown.*

### The model, precisely

- Query, key and value of each token = its embedding (W_Q, W_K, W_V are identity matrices).
- Attention scores are divided by √d = √2 ≈ 1.414, then passed through softmax.
- A **causal mask** lets each token attend only to itself and earlier tokens. For the last token, nothing is blocked.
- h = (last token's embedding) + (attention output).
- logit(token) = h · embedding(token). Reusing the embedding table this way is called **tied embeddings**.
- Left out: position embeddings, layer normalization, the feed-forward network, multiple heads and multiple layers.

### Step 1: "I love"

Query of "love": q = [0, 1].

| Key | q·k | ÷ √2 | e^score | Weight |
|---|---|---|---|---|
| I [1, 0] | 0 | 0.000 | 1.000 | 1.000 / 3.028 = 0.330 |
| love [0, 1] | 1 | 0.707 | 2.028 | 2.028 / 3.028 = 0.670 |
| **Sum** | | | **3.028** | **1.000** |

```
attention output = 0.330·[1, 0] + 0.670·[0, 1] = [0.330, 0.670]
h = [0, 1] + [0.330, 0.670] = [0.330, 1.670]
```

| Token | Embedding | Logit = h · embedding | e^logit | Probability |
|---|---|---|---|---|
| I | [1, 0] | 0.330 | 1.391 | 0.062 |
| love | [0, 1] | 1.670 | 5.312 | 0.237 |
| cats | [1, 1] | 0.330 + 1.670 = 2.000 | 7.389 | 0.330 |
| dogs | [−1, 1] | −0.330 + 1.670 = 1.340 | 3.819 | 0.171 |
| [EOS] | [2, 0.5] | 0.660 + 0.835 = 1.495 | 4.459 | 0.199 |
| **Sum** | | | **22.371** | **1.000** (to rounding) |

### Step 2: "I love cats"

Query of "cats": q = [1, 1].

| Key | q·k | ÷ √2 | e^score | Weight |
|---|---|---|---|---|
| I [1, 0] | 1 | 0.707 | 2.028 | 2.028 / 8.169 = 0.248 |
| love [0, 1] | 1 | 0.707 | 2.028 | 2.028 / 8.169 = 0.248 |
| cats [1, 1] | 2 | 1.414 | 4.113 | 4.113 / 8.169 = 0.503 |
| **Sum** | | | **8.169** | **1.000** (to rounding) |

```
attention output = 0.248·[1, 0] + 0.248·[0, 1] + 0.503·[1, 1] = [0.751, 0.751]
h = [1, 1] + [0.751, 0.751] = [1.751, 1.751]
```

| Token | Embedding | Logit | e^logit | Probability |
|---|---|---|---|---|
| I | [1, 0] | 1.751 | 5.760 | 0.046 |
| love | [0, 1] | 1.751 | 5.760 | 0.046 |
| cats | [1, 1] | 3.502 | 33.182 | 0.265 |
| dogs | [−1, 1] | 0.000 | 1.000 | 0.008 |
| [EOS] | [2, 0.5] | 1.751·2 + 1.751·0.5 = 4.378 | 79.679 | 0.635 |
| **Sum** | | | **125.381** | **1.000** |

### Temperature: all Step 1 probabilities

Probabilities = softmax(logits / T):

| Token | T = 0.5 | T = 1 | T = 2 |
|---|---|---|---|
| I | 0.016 | 0.062 | 0.115 |
| love | 0.237 | 0.237 | 0.224 |
| cats | 0.458 | 0.330 | 0.265 |
| dogs | 0.122 | 0.171 | 0.190 |
| [EOS] | 0.167 | 0.199 | 0.206 |

("love" happens to round to 0.237 at both T = 0.5 and T = 1.)

### Top-k with k = 2 (Step 1)

Keep cats (0.330) and love (0.237). Their sum is 0.567. Renormalize:

```
cats: 0.330 / 0.567 = 0.582
love: 0.237 / 0.567 = 0.418
```
