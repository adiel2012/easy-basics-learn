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

**[EOS]** means "end of sequence". It's a special token that signals the end of the text. Generation can also stop when a length limit or another stopping rule is reached.

The loop, as a picture:

```
        ┌──────────────────────────────────────┐
        ▼                                      │
  read the text so far                         │
        │                                      │
  score every possible next token              │
        │                                      │
  choose one token                             │
        │                                      │
  [EOS] or length limit? ── no ──▶ add the token to the text
        │
       yes
        ▼
      stop
```

**Analogy: phone autocomplete, on repeat.** Your phone suggests the next word as you type. Imagine always tapping the suggestion, then letting the phone suggest the next one, and the next. That's an LLM writing. Like autocomplete, an LLM predicts what comes next, but it can use a much richer context: all the text so far.

That's the whole process. The rest of this guide looks inside the "predict next token" box.

> **Quick check 1.** The model has written "I love cats" and now predicts [EOS]. What happens next?
> (a) It predicts another token. (b) It stops generating. (c) It updates its weights.
>
> *Answers to all checks are in Section 7.*

## 2. What you need to know first

Nothing is required: this guide explains what it uses. For more depth:

- **[Tokenization](tokenization.md)**: how text becomes token IDs.
- **[Attention Blocks](attention_blocks.md)**: how tokens share information.

## 3. How one token is predicted

Each prediction follows the same path:

```
text → token IDs → embeddings → transformer → last row → logits → probabilities → chosen token
```

Each stage in one or two sentences:

1. **Token IDs.** **Tokenization** splits the text into tokens and gives each one a number, its ID. See [Tokenization](tokenization.md).
2. **Embeddings.** Each ID is used to look up a vector (a list of numbers) in the model's embedding table.
3. **Transformer.** The vectors pass through the model's layers. Inside, **attention** lets each token gather information from the tokens before it.
4. **Last row.** Only the vector of the **last token** is used for the prediction. Through attention, it has gathered information from all the tokens before it. A final output layer turns this vector into scores.
5. **Logits: raw scores.** The model gives a score to **every** token in its vocabulary. A higher logit becomes a higher probability in the next stage.
6. **Probabilities.** **Softmax** turns the scores into probabilities between 0 and 1 that add up to 1.
7. **Chosen token.** A **decoding rule** picks one token. The simplest is **greedy decoding**: pick the most likely token.

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

1. **Attention:** in this toy model, the last token's embedding acts as its **query**, and each token's embedding acts as its **key** and **value**. Query-key scores decide how much each value contributes to a weighted mix. (Real transformers first transform the embeddings with learned query, key and value matrices.)
2. **Residual:** the mix is added to the last token's own embedding. Call the result **h**.
3. **Scores:** each token's logit is the dot product of h with that token's embedding. The dot product depends on both how well the two vectors line up and how long they are; a higher dot product gives a higher logit.

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
- **Why it matters:** it's simple and predictable. With the same model, prompt and settings (and deterministic hardware execution), it produces the same answer every time. If two tokens tie, a fixed tie-breaking rule decides.
- **Example:** in Step 1, greedy picks "cats" (0.330).

### Sampling

- **What it is:** pick a token at random, following the probabilities.
- **Why it matters:** it adds variety, so answers sound less repetitive. This is why the same prompt can give different answers.
- **Example:** in Step 1, "cats" would be picked 33% of the time, "love" 24%, and [EOS] 20%. If [EOS] were picked, the answer would simply be "I love".

Plain sampling has a problem: **every** token keeps some chance, even silly ones. In a real model with 100,000 tokens, thousands of unlikely tokens each have a tiny chance, and together they add up. Now and then one gets picked and the text goes off the rails.

Three settings control this. **Temperature** reshapes the probabilities. **Top-k** and **top-p** cut off the unlikely tokens.

> **Analogy: picking a restaurant.** Temperature is how adventurous you feel: low means "my usual place", high means "anything goes". Top-k is "only consider my k favourite places". Top-p is "only consider the places that together make up, say, 80% of my visits".

### Temperature

- **What it is:** a number T that the logits are divided by, **before** softmax:

  ```
  probabilities = softmax(logits / T)
  ```

- **Why it matters:** it controls how **sharp** the probabilities are.
  - **Low T (below 1):** dividing by a small number stretches the gaps between logits. The top token gets even more likely.
  - **T = 1:** nothing changes.
  - **High T (above 1):** dividing by a big number shrinks the gaps. The probabilities become more even, so unlikely tokens get more chances.
- **Important:** temperature never changes the **order** of the tokens. The most likely token stays the most likely. It also doesn't make text "better" or "more creative" by itself, only more or less varied.
- **Example:** all of Step 1 (logits 0.330, 1.670, 2.000, 1.340, 1.495):

| Token | T = 0.5 (sharper) | T = 1 (unchanged) | T = 2 (flatter) |
|---|---|---|---|
| cats | **0.458** | **0.330** | **0.265** |
| love | 0.237 | 0.237 | 0.224 |
| [EOS] | 0.167 | 0.199 | 0.206 |
| dogs | 0.122 | 0.171 | 0.190 |
| I | 0.016 | 0.062 | 0.115 |

At T = 0.5, "I" almost disappears (1.6%). At T = 2, it gets almost twice its normal chance (11.5%).

**The extremes:**

- **T close to 0:** the top token's probability approaches 1. Sampling becomes **greedy**. (You can't divide by exactly 0, so most tools treat T = 0 as "use greedy".)
- **Very high T:** all probabilities approach equal. The model picks almost at random.

### Top-k

- **What it is:** keep only the **k most likely** tokens. Throw away the rest. Then **renormalize**: divide each kept probability by their sum, so they add up to 1 again.
- **Why it matters:** the unlikely tokens can never be picked, however lucky the dice roll.
- **Example:** Step 1 with k = 2:

| Token | Probability | Kept? | New probability |
|---|---|---|---|
| cats | 0.330 | ✅ | 0.330 / 0.567 = **0.582** |
| love | 0.237 | ✅ | 0.237 / 0.567 = **0.418** |
| [EOS] | 0.199 | ❌ | 0 |
| dogs | 0.171 | ❌ | 0 |
| I | 0.062 | ❌ | 0 |
| **Sum of kept** | **0.567** | | **1.000** |

- **The weakness:** k is **fixed**. It keeps the same number of tokens whether the model is sure or unsure.
  - When the model is **sure** (one token has 95%), k = 50 still lets 49 poor choices in.
  - When the model is **unsure** (many good options), k = 2 cuts off good choices.

### Top-p (nucleus sampling)

Top-p fixes that weakness. Instead of a fixed **number** of tokens, it keeps a fixed **amount of probability**.

- **What it is:**
  1. Sort the tokens from most to least likely.
  2. Add up their probabilities, one by one (a **running total**).
  3. Stop as soon as the total reaches **p** or more. Keep those tokens.
  4. Renormalize the kept tokens.
- **Why it matters:** the number of kept tokens **adapts** to how sure the model is. The kept set is called the **nucleus**, which is why this is also called **nucleus sampling**.
- **Example:** p = 0.8, at both steps of our toy model.

**Step 1 ("I love"): the model is unsure.**

| Token | Probability | Running total | Kept? |
|---|---|---|---|
| cats | 0.330 | 0.330 | ✅ (not yet 0.8) |
| love | 0.237 | 0.567 | ✅ (not yet 0.8) |
| [EOS] | 0.199 | 0.766 | ✅ (not yet 0.8) |
| dogs | 0.171 | 0.937 | ✅ reached 0.8, stop here |
| I | 0.062 | | ❌ |

**4 tokens kept.** Renormalized (÷ 0.937): cats 0.352, love 0.253, [EOS] 0.212, dogs 0.182.

**Step 2 ("I love cats"): the model is sure.**

| Token | Probability | Running total | Kept? |
|---|---|---|---|
| [EOS] | 0.635 | 0.635 | ✅ (not yet 0.8) |
| cats | 0.265 | 0.900 | ✅ reached 0.8, stop here |
| love | 0.046 | | ❌ |
| I | 0.046 | | ❌ |
| dogs | 0.008 | | ❌ |

**Only 2 tokens kept.** Renormalized (÷ 0.900): [EOS] 0.706, cats 0.294.

Same p, but 4 tokens in the unsure step and 2 in the sure step. That's the advantage over top-k.

> **Common mistake:** top-p keeps the token that **crosses** p, not just the ones below it. In Step 1, the total before "dogs" is only 0.766, which is less than 0.8, so "dogs" is needed to reach 0.8.

### Putting it all together

These settings are usually used **together**, in this order:

```
logits
  │  ÷ T                       (temperature: reshape)
  ▼
softmax → probabilities
  │  keep the top k            (top-k: cut by count)
  │  keep the top p            (top-p: cut by probability)
  ▼
renormalize → sample one token
```

The order matters, because temperature changes the probabilities that top-p adds up. Step 1 with p = 0.8:

| Temperature | Running totals (cats, love, [EOS], dogs) | Tokens kept by top-p = 0.8 |
|---|---|---|
| T = 0.5 | 0.458, 0.695, **0.861** | 3 (cats, love, [EOS]) |
| T = 1 | 0.330, 0.567, 0.766, **0.937** | 4 (adds dogs) |
| T = 2 | 0.265, 0.489, 0.695, **0.885** | 4 (adds dogs) |

A sharper distribution reaches p sooner, so fewer tokens survive. (Libraries differ slightly in the exact order they apply these steps, but temperature almost always comes first.)

**Settings that switch things off, or turn them into greedy:**

| Setting | Effect |
|---|---|
| T = 1 | temperature has no effect |
| T → 0 | same as greedy |
| k = 1 | same as greedy: only the top token is left |
| k = vocabulary size | top-k has no effect |
| p = 1.0 | top-p has no effect |
| very small p (e.g. 0.01) | almost always greedy: only the top token is kept |

**Common starting points.** There are no universal best values; each model and tool has its own defaults. Roughly:

- **Facts, code, data extraction:** low temperature (around 0 to 0.3), or greedy. You want the most likely answer.
- **General chat:** moderate temperature (around 0.7), often with top-p around 0.9.
- **Brainstorming, stories:** higher temperature (around 0.8 to 1.0), with top-p to stop the strange tail tokens.

A common tip is to adjust **temperature or top-p, not both at once**, so you can tell which change made the difference.

> **Quick check 3.** Probabilities are cats 0.6 and dogs 0.4. What does greedy decoding choose? What might sampling choose?
>
> **Quick check 4.** Probabilities are A 0.5, B 0.3, C 0.15, D 0.05. Which tokens does top-p = 0.7 keep? Which does top-k = 3 keep?

## 6. How inference becomes efficient

### Two phases: prefill and decode

- **Prefill:** the model reads the **whole prompt at once**. It computes all prompt positions in parallel, and the **last position** supplies the logits for the first new token.
- **Decode:** the model then generates **one token at a time**. Each step depends on the token chosen just before, so these steps can't run in parallel.

So prompt tokens can be processed in parallel, while generated tokens come out one after another, which is why the answer appears word by word. Long prompts still take real time and memory to process.

### The KV cache

- **What it is:** a saved copy of each token's **keys and values**, the vectors that later tokens attend to. The cache stores them for **every layer** of the model.
- **Why it matters:** without it, every decode step would recompute them for the whole text.
- **Example:** in Step 2, "cats" attended to "I" and "love". Their keys and values were already computed in Step 1, and they don't change, because earlier tokens never look at later ones. So only "cats" needs new ones.

| Step | Without cache: compute keys/values for | With cache |
|---|---|---|
| 1 (prefill) | I, love | I, love (saved) |
| 2 (decode) | I, love, cats | cats only |
| When the text has 101 tokens | all 101 tokens | 1 new token |

The cost is memory: the cache grows with every token.

> **Quick check 5.** At Step 2, which keys and values come from the cache, and which must be computed?

## 7. Remember this

### Five essential ideas

1. **One token at a time.** An LLM loops: read → score → choose → append → repeat, until [EOS] or a length limit. It never writes a whole sentence at once.
2. **Only the last row predicts.** The other rows still matter: they provide the keys and values the last token attends to.
3. **Scores → probabilities → choice.** Logits are scores, softmax makes probabilities, and decoding chooses.
4. **Greedy is fixed; sampling varies.** Temperature reshapes the probabilities (low = sharper, high = flatter). Top-k keeps a fixed number of tokens; top-p keeps just enough tokens to reach probability p, so it adapts to how sure the model is.
5. **Nothing is learned during inference.** The weights are fixed. The KV cache only saves computation, and the context window limits how many tokens fit.

### Exercises

1. Using the Step 2 table, which token would greedy decoding pick if [EOS] didn't exist?
2. Logits are [2, 2]. What are the probabilities after softmax?
3. What happens to sampling if the temperature is very low?
4. Using the Step 2 probabilities, which tokens does top-k = 3 keep? What problem do you notice?
5. Using the Step 1 probabilities, which tokens does top-p = 0.5 keep, and what are their new probabilities?

Try the exercises and the quick checks before reading on.

---

### Check your answers

**Quick checks**

1. **(b)** It stops generating. [EOS] is the stop signal. The weights never change during inference.
2. **The last row**, for "love". It's the last token, so through attention it has gathered information from all the tokens before it.
3. **Greedy always chooses cats**, the highest probability. **Sampling** picks cats about 60% of the time and dogs about 40%.
4. **Top-p = 0.7 keeps A and B** (running totals 0.5, then 0.8, which reaches 0.7). **Top-k = 3 keeps A, B and C.**
5. **"I" and "love" come from the cache** (computed in Step 1). Only **"cats"** needs new keys and values.

**Exercises**

1. **cats** (logit 3.502, the next highest).
2. **[0.5, 0.5].** Equal scores give equal probabilities.
3. **It becomes almost greedy.** The top token's probability approaches 1.
4. **[EOS] and cats, plus one of "I" or "love".** "I" and "love" tie at 0.046, so a tie-breaking rule decides which one gets the third place. It also lets in a weak token (4.6%) even though the model is fairly sure. Top-p = 0.8 would keep only [EOS] and cats.
5. **cats and love** (running totals 0.330, then 0.567, which reaches 0.5). New probabilities: cats 0.330 / 0.567 = **0.582**, love 0.237 / 0.567 = **0.418**. Here top-p = 0.5 happens to give the same result as top-k = 2.

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
- **[Zero-Shot and Few-Shot Learning](zero_shot_few_shot_learning.md)**: how examples placed in the prompt change what the model predicts, with no weight changes.
- **[Vision Transformer (ViT)](vit.md)**: the same transformer used on images. A standard image-classification ViT typically produces its class prediction in one forward pass, with no generation loop.

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
| Top-k | keep only the k most likely tokens, then sample |
| Top-p (nucleus sampling) | keep the fewest top tokens whose probabilities reach p, then sample |
| Renormalize | divide the kept probabilities by their sum so they add up to 1 again |
| Prefill / decode | read the prompt at once / generate one token at a time |
| KV cache | saved keys and values from earlier tokens |
| Context window | the maximum number of tokens the model can handle |

### Learn more

- 📖 [The Illustrated GPT-2](https://jalammar.github.io/illustrated-gpt2/) (Jay Alammar): a visual walk-through of how GPT-2 generates text
- 📖 [nanoGPT README](https://github.com/karpathy/nanoGPT/blob/master/README.md) (Andrej Karpathy): a small, readable GPT you can train and sample from
- 📖 [Generation strategies](https://huggingface.co/docs/transformers/main/en/generation_strategies) (Hugging Face docs): greedy search, sampling and other ways to pick the next token, with code

---

## Appendix: full arithmetic

*Optional. For readers who want to verify every number. All values are **approximate**: they are rounded to 3 decimals, and each step uses the rounded values shown, so you can reproduce them by hand. Recomputing at full precision can differ in the last digit.*

### The model, precisely

This is a deliberately simplified model: tied embeddings, and no learned query, key or value projections.

- Query, key and value of each token = its embedding (W_Q, W_K, W_V are identity matrices).
- Attention scores are divided by √d_k = √2 ≈ 1.414 (the key width is 2), then passed through softmax. This scaling keeps scores from growing too large as the key width grows, which would make softmax put almost all weight on one token.
- A **causal mask** lets each token attend only to itself and earlier tokens. For the last token, nothing is blocked.
- h = (last token's embedding) + (attention output).
- logit(token) = h · embedding(token). Reusing the embedding table as the output layer is called **tied embeddings**. Many real models instead have a separate, learned output layer.
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

### Top-p with p = 0.8

**Step 1.** Running totals: 0.330, 0.567, 0.766, 0.937. The total first reaches 0.8 at "dogs", so 4 tokens are kept. Their sum is 0.937:

```
cats:  0.330 / 0.937 = 0.352
love:  0.237 / 0.937 = 0.253
[EOS]: 0.199 / 0.937 = 0.212
dogs:  0.171 / 0.937 = 0.182
```

**Step 2.** Running totals: 0.635, 0.900. The total reaches 0.8 at "cats", so 2 tokens are kept. Their sum is 0.900:

```
[EOS]: 0.635 / 0.900 = 0.706
cats:  0.265 / 0.900 = 0.294
```

### Temperature, then top-p (Step 1, p = 0.8)

Using the temperature table above, sorted from most to least likely:

| T | Sorted probabilities | Running totals | Kept |
|---|---|---|---|
| 0.5 | 0.458, 0.237, 0.167, 0.122, 0.016 | 0.458, 0.695, 0.861 | 3 |
| 1 | 0.330, 0.237, 0.199, 0.171, 0.062 | 0.330, 0.567, 0.766, 0.937 | 4 |
| 2 | 0.265, 0.224, 0.206, 0.190, 0.115 | 0.265, 0.489, 0.695, 0.885 | 4 |

The order of tokens is the same at every temperature: cats, love, [EOS], dogs, I.
