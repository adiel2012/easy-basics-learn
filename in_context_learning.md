# In-Context Learning: Learning From the Prompt, Without Training

> **The one idea in this guide:** a large language model can pick up a new task **from the text in its prompt alone**: instructions, examples, documents. Its weights never change. The "learning" lasts only as long as the text stays in the prompt.
>
> Everything below explains, demonstrates or deepens that idea.

## 1. In simple words

Normally, a machine learning model learns by **training**: its weights are adjusted, again and again, until it gets good at a task.

**In-context learning (ICL)** is different. You leave the model exactly as it is, and just put useful information in the prompt: what the task is, some examples, maybe a document. The model reads it and adapts its answer.

```
Training (weights change)          In-context learning (weights stay the same)
─────────────────────────          ───────────────────────────────────────────
data → adjust weights → new model  same model + new prompt → new behaviour
lasts forever                      lasts only while the text is in the prompt
```

**Analogy: an open-book exam.** Training is studying for weeks until the knowledge is in your head. In-context learning is being handed a sheet of notes during the exam. You can use the notes right away, but you don't take them home. Next exam, no notes, no help.

> **Quick check 1.** After in-context learning, which of these has changed: (a) the model's weights, (b) the model's answer, (c) both?
>
> *Answers to all checks are in Section 7.*

## 2. What you need to know first

Nothing is required: this guide explains what it uses. For more depth:

- **[LLM Inference](llm_inference.md)**: the one-token-at-a-time loop, prefill and the context window.
- **[Zero-Shot and Few-Shot Learning](zero_shot_few_shot_learning.md)**: the most common way to use in-context learning, with examples.
- **[Attention Blocks](attention_blocks.md)**: the mechanism that lets the answer read the prompt.

## 3. The key terms

| Term | Meaning |
|---|---|
| Context | everything in the prompt the model can see: instructions, examples, documents, earlier messages |
| Context window | the maximum number of tokens the context can hold |
| In-context learning (ICL) | adapting to a task using only the context, without weight changes |
| Demonstration | an example inside the prompt (also called a "shot") |
| Fine-tuning | training the model further, which **does** change its weights |
| Induction head | an attention pattern that finds an earlier match and copies what came after it |

## 4. How it works

### What can go in the context

ICL is broader than few-shot examples. The model can "learn" from any of these:

| What you put in the context | What the model picks up | Example |
|---|---|---|
| **Instructions** | what to do | "Answer in one sentence, in Spanish." |
| **Examples** | the pattern, format and labels | `dog → DOG-3`, `bird → BIRD-4` |
| **Documents** | facts it didn't know | a company's refund policy |
| **Definitions** | new words or rules | "A *flurb* is a red square." |
| **Earlier conversation** | your preferences, corrections | "No, I meant the 2024 figures." |

### Why it works: a pattern-continuing machine

An LLM is trained to **predict the next token** on huge amounts of text. That text is full of patterns that start and then repeat: lists, tables, question-answer pairs, translations, code.

To predict well, the model had to get good at a skill: **notice the pattern in the text so far, and continue it**. That skill is exactly what in-context learning uses.

### What happens inside

There's nothing special about the prompt. From [LLM Inference](llm_inference.md):

```
context tokens → prefill (read all at once) → last position → logits → next token
```

**Attention** lets the last token look back at **every** earlier token. So the examples and instructions change the vector at the last position, which changes the logits, which changes the answer.

Researchers have found specific attention patterns that help. A simple one is the **induction head**:

```
... Harry Potter ... ... ... Harry  →  ?
```

The head looks back for the earlier **"Harry"**, sees what came **after** it ("Potter"), and boosts "Potter" as the next token. Copying "what came after the last match" is a basic building block of pattern-continuing.

### In-context learning vs fine-tuning

| | In-context learning | Fine-tuning |
|---|---|---|
| Changes weights? | no | yes |
| Setup time | none: just write the prompt | a training run |
| Data needed | a few examples or a document | usually hundreds or more |
| How long it lasts | only this prompt | permanently |
| Cost per request | higher (prompt tokens every time) | lower (no examples needed) |
| Limit | the context window | the quality and quantity of training data |

## 5. Try it yourself

### Learning a rule from examples

```
1 → 3
2 → 5
3 → 7
10 →
```

Work out the rule with pen and paper:

| Input | Output | Output − 2 × Input |
|---|---|---|
| 1 | 3 | 1 |
| 2 | 5 | 1 |
| 3 | 7 | 1 |

The rule is **output = 2 × input + 1**, so `10 → 21`. A capable model would usually write 21. Nobody told it the rule; it inferred it from three examples in the context. (Researchers have shown transformers can learn to do this kind of in-context fitting for simple functions, such as straight lines: Garg et al., 2022.)

### Learning a new fact from a document

```
Note: In our office, the meeting room "Blue" is on floor 3, and "Green" is on floor 5.

Question: Which floor is the Green room on?
```

The model can't know this from training. It's not a pattern; it's a fact given in the context. The answer is **floor 5**. This is the idea behind **[RAG](rag.md)**: put the right documents into the context so the model can use them.

### How many examples fit?

The context window is a hard limit. Suppose:

- context window: **8,000 tokens**
- instructions: **200 tokens**
- each example: **150 tokens**
- room left for the answer: **800 tokens**

```
space for examples = 8,000 − 200 − 800 = 7,000 tokens
number of examples = 7,000 ÷ 150 ≈ 46.7  →  at most 46 examples
```

Real context windows are often much larger, but every example still costs tokens (time and money) on **every** request.

> **Quick check 2.** A context window is 4,000 tokens. Instructions take 100, the answer needs 400, and each example is 250. How many examples fit?

## 6. Variations, tips and mistakes

### Many-shot in-context learning

With very long context windows, you can include hundreds of examples. This is called **many-shot** ICL. It can approach fine-tuning quality on some tasks, but each request becomes long and expensive.

### What affects how well it works

| Factor | Effect |
|---|---|
| **Model size** | larger models are generally much better at ICL |
| **Clear instructions** | often matter as much as examples |
| **Example format** | the model copies it closely, including mistakes |
| **Example order** | answers can lean toward the label of the last example |
| **Label balance** | answers can lean toward the most common label |
| **Where the key information sits** | in very long contexts, models can miss details buried in the middle |

### Common mistakes

| Mistake | What goes wrong | Fix |
|---|---|---|
| Expecting it to persist | a new conversation starts from scratch | include the context every time, or fine-tune |
| Too much irrelevant text | important details get lost, and cost rises | include only what's needed |
| Conflicting information | the model may follow either version | remove or clearly mark outdated text |
| Treating it as reliable memory for facts | the model may still mix in what it learned in training | ask it to answer only from the given text, and to say if the answer isn't there |
| Putting untrusted text in the context | text can contain instructions the model follows (**prompt injection**) | mark it clearly as data, and limit what the model can do |

## 7. Remember this

### Five essential ideas

1. **Learning from the prompt.** The model adapts to instructions, examples and documents in its context.
2. **No weights change.** It lasts only while the text is in the context.
3. **It's pattern continuation.** Training on next-token prediction made the model good at noticing and continuing patterns.
4. **Attention is the mechanism.** The answer's tokens look back at the context; induction heads are one known pattern.
5. **The context window is the limit**, and every token in it costs time and money.

### Exercises

1. Examples: `2 → 4`, `3 → 9`, `4 → 16`. What does the model probably answer for `5 →`, and what's the rule?
2. Give one advantage of in-context learning over fine-tuning, and one disadvantage.
3. Why does a model forget your examples in a new conversation?
4. Context window 16,000 tokens. Instructions 500, answer 1,500, each document 2,000. How many documents fit?

Try the exercises and the quick checks before reading on.

---

### Check your answers

**Quick checks**

1. **(b)** Only the answer. The weights are the same before and after.
2. 4,000 − 100 − 400 = 3,500. 3,500 ÷ 250 = **14 examples**.

**Exercises**

1. **25.** The rule is output = input × input (squaring).
2. Advantage: no training needed, works immediately (or: easy to change). Disadvantage: costs prompt tokens on every request (or: limited by the context window, or: doesn't persist).
3. The examples were only in the old conversation's context. The weights never stored them.
4. 16,000 − 500 − 1,500 = 14,000. 14,000 ÷ 2,000 = **7 documents**.

## 8. Going further

### Where this connects

- **[Zero-Shot and Few-Shot Learning](zero_shot_few_shot_learning.md)**: in-context learning with examples.
- **[Chain-of-Thought Reasoning](chain_of_thought_reasoning.md)**: examples that teach the model to show steps.
- **[RAG](rag.md)**: automatically choosing which documents to put in the context.
- **[KV Cache](kv_cache.md)**: why long contexts cost memory.
- **[LoRA](lora.md)**: a cheap way to fine-tune when in-context learning isn't enough.

### Short glossary

| Term | Meaning |
|---|---|
| In-context learning | adapting to a task from the prompt alone |
| Context window | maximum tokens the model can see at once |
| Demonstration | an example in the prompt |
| Many-shot ICL | in-context learning with hundreds of examples |
| Induction head | attention pattern that copies what followed an earlier match |
| Prompt injection | instructions hidden in context text that the model may wrongly follow |

### Learn more

- 📖 [Language Models are Few-Shot Learners](https://arxiv.org/abs/2005.14165) (Brown et al., 2020): the GPT-3 paper that popularised in-context learning
- 📖 [In-context Learning and Induction Heads](https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html) (Olsson et al., 2022): the attention patterns behind it
- 📖 [What Can Transformers Learn In-Context?](https://arxiv.org/abs/2208.01066) (Garg et al., 2022): transformers learning simple functions from examples
