# Tokenization

## 1. In simple words

Computers represent text using numbers called bytes. A language model needs text in a different numeric form: a sequence of numbered pieces from a fixed list. **Tokenization** is the step that produces that sequence.

It works in two moves:

1. **Cut** the text into small pieces called **tokens**.
2. **Look up** each piece in a numbered list, and replace it with its number.

**Analogy: LEGO bricks.** Imagine building any word out of a box of LEGO bricks. Common words get their own big brick ("the", "and"). Rare words are built from several smaller bricks ("un" + "believ" + "able"). With a good set of bricks, you can build most words, even ones you've never seen before, as long as you have the small bricks they need.

```
"I love cats"  →  ["I", " love", " cats"]  →  [40, 1842, 9246]
     text              tokens                     token IDs
```

(The numbers here are made up. Every tokenizer has its own list.)

## 2. What you need to know first

Nothing! Just the idea that computers store everything as numbers.

## 3. The key terms

| Term | Meaning | Example |
|---|---|---|
| **Token** | one piece of text | `" cats"` |
| **Vocabulary** | the full list of tokens the tokenizer knows | 50,000 tokens |
| **Token ID** | a token's number in the vocabulary | `9246` |
| **Tokenizer** | the tool that cuts text and looks up IDs | |
| **Merge** | joining two tokens into one bigger token | `u` + `g` → `ug` |
| **Unknown token** | a placeholder for text not in the vocabulary | `[UNK]` |

## 4. How it works

### Three ways to cut text

There are three main choices for what a "piece" should be.

**Option A: whole words**

```
"unbelievable cats"  →  ["unbelievable", "cats"]
```

- ✅ Short sequences
- ❌ The vocabulary gets huge: every word, plus every form of it ("cat", "cats", "catlike"...)
- ❌ A new word or typo is either mapped to an unknown token or cannot be represented directly

**Option B: single characters**

```
"cats"  →  ["c", "a", "t", "s"]
```

- ✅ A small vocabulary can represent any word made from the supported characters
- ❌ Sequences get very long
- ❌ A single letter carries almost no meaning

**Option C: subwords (what modern models use)**

```
"unbelievable cats"  →  ["un", "believ", "able", " cats"]
```

- ✅ Common words stay whole
- ✅ Rare words can be split into smaller, reusable pieces. These pieces sometimes match meaningful word parts (like "un" or "able"), but not always.
- ✅ Unfamiliar words can be built by combining smaller pieces, as long as those pieces are in the vocabulary. Byte-level tokenizers (Section 6) can represent arbitrary input text without needing an unknown-token placeholder.

Subwords are the middle ground. They take the good parts of both other options.

### Where the vocabulary comes from

The tokenizer is typically **trained** on a large amount of text to learn its vocabulary, before the language model itself is trained. (The model then learns its own parameters, including the embeddings in Section 8, during model training.) One of the most popular methods is **Byte Pair Encoding (BPE)**, and the idea is very simple:

> Start with single characters. Find the pair that appears next to each other most often. Glue it into one new token. Repeat.

Frequent pieces of text end up as single tokens. Rare ones stay split.

## 5. Try it yourself: BPE by hand

> **Note:** This is a simplified BPE example. Real tokenizers may first split text into chunks, and they may handle bytes, spaces and special tokens differently. The core idea, repeatedly merging the most frequent pair, is the same.

### The training text

Imagine our whole training text contains only these five words, with these counts:

| Word | How many times |
|---|---|
| hug | 10 |
| pug | 5 |
| pun | 12 |
| bun | 4 |
| hugs | 5 |

### Start: split everything into characters

```
h u g      (×10)
p u g      (×5)
p u n      (×12)
b u n      (×4)
h u g s    (×5)
```

Starting vocabulary: `b, g, h, n, p, s, u` (7 tokens)

### Round 1: find the most frequent neighbouring pair

Count each pair of tokens that sit next to each other. A pair inside "hug" appears 10 times in the text, because "hug" appears 10 times, so it counts 10. That's why each count is weighted by how often its word appears:

| Pair | Where it appears | Count |
|---|---|---|
| h u | hug (10) + hugs (5) | 15 |
| **u g** | hug (10) + pug (5) + hugs (5) | **20** ← most frequent |
| p u | pug (5) + pun (12) | 17 |
| u n | pun (12) + bun (4) | 16 |
| b u | bun (4) | 4 |
| g s | hugs (5) | 5 |

**Merge `u` + `g` → `ug`.** The words become:

```
h ug       (×10)
p ug       (×5)
p u n      (×12)
b u n      (×4)
h ug s     (×5)
```

Vocabulary: `b, g, h, n, p, s, u, ug` (8 tokens)

### Round 2: count again, using the newly merged pieces

`ug` is now one token, so pairs are counted between the new pieces.

| Pair | Count |
|---|---|
| h ug | 10 + 5 = 15 |
| p ug | 5 |
| p u | 12 |
| **u n** | 12 + 4 = **16** ← most frequent |
| b u | 4 |
| ug s | 5 |

**Merge `u` + `n` → `un`.**

```
h ug       (×10)
p ug       (×5)
p un       (×12)
b un       (×4)
h ug s     (×5)
```

Vocabulary: `b, g, h, n, p, s, u, ug, un` (9 tokens)

### Round 3: repeat

Same process, one more time.

| Pair | Count |
|---|---|
| **h ug** | 10 + 5 = **15** ← most frequent |
| p ug | 5 |
| p un | 12 |
| b un | 4 |
| ug s | 5 |

**Merge `h` + `ug` → `hug`.**

```
hug        (×10)
p ug       (×5)
p un       (×12)
b un       (×4)
hug s      (×5)
```

Vocabulary: `b, g, h, n, p, s, u, ug, un, hug` (10 tokens)

We could keep going, but let's stop here. The tokenizer has learned **3 merge rules, in this order**:

```
1.  u + g  → ug
2.  u + n  → un
3.  h + ug → hug
```

### Using the tokenizer on new words

To tokenize new text in this simplified example, split it into characters and repeatedly apply the available merge with the **highest learned priority** (the merge learned earliest has the highest priority). Stop when no learned merge can be applied. At each step, choose the highest-priority merge that is currently possible.

**"hugs"**

```
h u g s   →  rule 1  →  h ug s  →  rule 3  →  hug s
Result: ["hug", "s"]
```

**"bugs"**, a word that was never in the training text:

```
b u g s   →  rule 1  →  b ug s   (rules 2 and 3 don't apply)
Result: ["b", "ug", "s"]
```

The tokenizer handled a brand-new word by building it from known pieces. That's the big advantage of subwords.

**"mug"**

```
m u g   →  "m" is not in the vocabulary!
Result: ["[UNK]", "ug"]
```

The letter "m" never appeared in the training text, so the tokenizer has no token for it. Section 6 shows how real tokenizers avoid this problem.

### From tokens to IDs

Finally, each token is replaced by its position in the vocabulary:

| Token | b | g | h | n | p | s | u | ug | un | hug |
|---|---|---|---|---|---|---|---|---|---|---|
| **ID** | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |

```
"hugs"  →  ["hug", "s"]       →  [9, 5]
"bugs"  →  ["b", "ug", "s"]   →  [0, 7, 5]
```

These numbers are what the model actually receives.

## 6. Variations and common mistakes

### Variations (optional reading)

You don't need these details to understand tokenization. They're here so the names are familiar when you meet them.

- **Byte-level BPE** (used by many GPT-family tokenizers): it starts from the 256 possible **byte** values rather than directly from letters. Every possible text is made of bytes, so byte-level tokenizers can represent arbitrary input bytes without requiring an unknown-token placeholder, provided the tokenizer supports all byte values. This solves the "mug" problem.
- **WordPiece** (used by BERT): another subword method. It learns its vocabulary with a different procedure from standard BPE. BERT marks pieces that continue a word with `##`, as in `["play", "##ing"]`.
- **SentencePiece**: a tokenizer framework that can learn subword vocabularies directly from raw text, using methods such as BPE or Unigram. It often represents spaces with a visible marker such as `▁`, which helps it handle languages with different word-boundary conventions.
- **Special tokens**: extra tokens with special jobs, such as `[PAD]` (padding), `[BOS]` (start of text), `[EOS]` (end of text) and `[MASK]`.

### Common mistakes

- **"One token = one word."** Not true. For some English text and tokenizers, one token averages roughly three-quarters of a word, but the actual ratio varies considerably with the tokenizer, the language and the kind of text.
- **Spaces matter.** `"cats"` and `" cats"` (with a leading space) are usually different tokens.
- **Capital letters matter.** `"Cat"` and `"cat"` can be different tokens.
- **Numbers are often split oddly.** `"2026"` might become `["20", "26"]`. This is one reason language models sometimes struggle with arithmetic.
- **Not all languages cost the same.** Languages or writing systems that are poorly represented in a tokenizer's training data often require more tokens to express the same amount of text. The exact difference depends on the tokenizer and the text.

## 7. Remember this

- Tokenization turns **text → tokens → numbers (IDs)**.
- Modern models use **subwords**: common words stay whole, and rare words are split into reusable pieces.
- **BPE** learns its vocabulary by repeatedly **merging the most frequent pair**.
- New text is tokenized by applying the learned merges **by priority**, until none apply.
- Byte-level tokenizers can represent **arbitrary input bytes** without an unknown-token placeholder, as long as the implementation covers all byte values.

### Quick check

Suppose a tokenizer knows these tokens: `c, a, t, s, ca, at, cat`.

1. Can it represent "cats" without having a "cats" token?
2. Does token ID 7 mean the same thing in every tokenizer?
3. Why might a byte-level tokenizer handle an unfamiliar character better than a character tokenizer with an incomplete vocabulary?

Try answering before reading on.

### Answers

1. **Yes.** One possible segmentation is `["cat", "s"]`, assuming the tokenizer's rules permit it. Having the right pieces makes it possible; the tokenizer's encoding rules decide the exact split.
2. **No.** Token IDs depend on each tokenizer's own vocabulary. ID 7 can be a different token in every tokenizer.
3. **Every character is made of bytes**, and a byte-level tokenizer always knows all 256 byte values. So it can represent the character through its bytes, even if it has never seen that character before.

## 8. Going further

Tokenization is the first step before a model like a transformer can work with text:

```
text  →  tokens  →  token IDs  →  embedding vectors  →  transformer
                                  (one row per token)
```

The model has an **embedding table**: one row for every token in the vocabulary. Each row is a vector (a list of numbers) that the model **learned** during training. When the model processes text, each token ID simply selects its row, called the token's **embedding**. Nothing new is learned during the lookup itself.

Stacking the embeddings of all the tokens in a text gives a matrix with **one row per token**. Depending on the model, this matrix is usually combined with positional information (where each token sits in the text) before entering the transformer. The result is the input X in the [Attention Blocks](attention_blocks.md) guide, and the number of tokens is its number of rows, n. The transformer then processes X through its neural-network layers to produce contextual representations.

This is why tokenization matters so much: it decides how many tokens a text uses, which affects how much text fits in a model's context and how much it costs to process.
