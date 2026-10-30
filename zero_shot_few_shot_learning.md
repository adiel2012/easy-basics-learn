# Zero-Shot and Few-Shot Learning: Teaching by Example, Inside the Prompt

> **The one idea in this guide:** you can teach a language model a new task just by **showing it examples in the prompt**. Nothing inside the model changes. The examples are simply more text it reads before it writes.
>
> Everything below explains, demonstrates or deepens that idea.

## 1. In simple words

A **shot** is one **example** you put in the prompt: an input, plus the answer you want.

- **Zero-shot:** you only describe the task. No examples.
- **One-shot:** you describe the task and give **one** example.
- **Few-shot:** you give **a few** examples (usually 2 to 10).

```
Zero-shot                     Few-shot
─────────                     ────────
Translate to French:          Translate to French:
cat →                         dog → chien
                              bird → oiseau
                              cat →
```

In both cases the model then continues the text, and should write "chat".

**Analogy: a new job.** On your first day, your manager can either *tell* you "file each invoice by month" (zero-shot), or *show* you three invoices already filed (few-shot). Showing is often clearer, especially when the rule is unusual. Either way, you haven't gone back to school: you just used what you already knew, plus the examples in front of you.

> **The surprising part:** despite the name, no "learning" happens in the usual sense. The model's weights are **not** updated. When the conversation ends, the model has forgotten the examples. This is why it's also called **in-context learning**: the learning lives only in the context (the prompt).

> **Quick check 1.** You give a model 5 examples in a prompt, and it does the task well. Tomorrow, in a new conversation, you ask the same task without examples. Does it still remember your 5 examples?
>
> *Answers to all checks are in Section 7.*

## 2. What you need to know first

Nothing is required: this guide explains what it uses. For more depth:

- **[LLM Inference](llm_inference.md)**: how a model writes one token at a time. Few-shot examples are just tokens it reads first.
- **[Tokenization](tokenization.md)**: why examples "cost" tokens.

## 3. The key terms

| Term | Meaning |
|---|---|
| Prompt | the text you give the model |
| Shot | one worked example (input + desired output) inside the prompt |
| Zero-shot | instruction only, no examples |
| One-shot / few-shot | one / a few examples |
| Query | the new input you actually want answered, placed after the examples |
| In-context learning | doing a new task from information in the prompt, with no weight changes |
| Fine-tuning | actually training the model on examples, which **does** change its weights |

## 4. How it works

### Why examples help at all

An LLM is trained to **continue text**. During training it read huge amounts of text, which is full of patterns: lists, question-and-answer pairs, tables, translations.

When you write a few examples in the same format, you create a pattern. The most natural continuation of a pattern is **another item that follows it**. So the model writes an answer in the same style as your examples.

Examples tell the model several things at once:

| What the examples show | Example |
|---|---|
| **The task** | "this is about sentiment, not topic" |
| **The allowed answers** | "only `positive` or `negative`" |
| **The format** | "one lowercase word, no explanation" |
| **The style** | "short, formal, uses bullet points" |
| **Unusual rules** | anything the model couldn't guess from words alone |

### What happens inside (connecting to inference)

From the model's point of view, there's nothing special about examples. The whole prompt, examples included, goes through the same steps as any text (see [LLM Inference](llm_inference.md)):

```
instruction + examples + query  →  prefill (read all at once)  →  predict next token  →  ...
```

When the model predicts the answer, **attention** lets the last token look back at the examples. Their pattern changes the logits, and so changes which token is most likely.

### Structure of a good few-shot prompt

```
1. Instruction   (optional but helpful)   Classify the review as positive or negative.
2. Examples      (the shots)              Review: Loved it!           Label: positive
                                          Review: Total waste of time. Label: negative
3. Query         (the real question)      Review: Best film this year.
4. Answer slot   (left open)              Label:
```

Ending with `Label:` invites the model to write exactly one label next.

## 5. Try it yourself

### A rule the model cannot guess

Suppose your company uses a made-up product code. The rule is:

> **Write the word in capitals, then a dash, then the number of letters.**

**Zero-shot prompt:**

```
Convert the word to our product code.
horse →
```

The model has never seen this rule. "Product code" could mean anything, so it has to guess.

**Few-shot prompt:**

```
Convert the word to our product code.
dog → DOG-3
bird → BIRD-4
cat → CAT-3
horse →
```

Now the pattern is visible. Check it yourself with pen and paper:

| Word | Capitals | Letters | Code |
|---|---|---|---|
| dog | DOG | 3 | DOG-3 |
| bird | BIRD | 4 | BIRD-4 |
| cat | CAT | 3 | CAT-3 |
| **horse** | **HORSE** | **5** | **HORSE-5** |

The expected answer is **HORSE-5**.

### How the probabilities shift

> **Made-up numbers.** The table below is an illustration of the idea, not output from a real model.

Possible first piece of the answer after `horse →`:

| Next token | Zero-shot | Few-shot (3 examples) |
|---|---|---|
| `HORSE` | 0.10 | **0.85** |
| `Horse` | 0.30 | 0.05 |
| `HRS` | 0.20 | 0.02 |
| `#` | 0.15 | 0.01 |
| something else | 0.25 | 0.07 |

The model is the same, and its weights are the same. Only the **text it reads** changed, and that moved most of the probability onto the right answer.

### One example can mislead

**One-shot prompt:**

```
dog → DOG-3
horse →
```

With only one example, several rules fit:

- "capitals, dash, number of letters" → HORSE-5
- "capitals, then always -3" → HORSE-3

Both explain `dog → DOG-3`! A second example with a **different** length (`bird → BIRD-4`) rules out the "always 3" idea. That's why examples should **vary** in the right ways.

> **Quick check 2.** Your examples are `cat → CAT-3`, `dog → DOG-3`, `cow → COW-3`. What problem might the model have with `horse`?

## 6. Variations, tips and mistakes

### Zero-shot vs few-shot vs fine-tuning

| | Zero-shot | Few-shot | Fine-tuning |
|---|---|---|---|
| Examples needed | 0 | a few | hundreds to thousands |
| Changes the model's weights? | no | no | **yes** |
| Effort | lowest | low | high (training run, data) |
| Cost per request | lowest | more tokens | low (no examples needed) |
| Best for | common, clearly described tasks | specific formats, custom labels, unusual rules | many requests of the same task, or when few-shot isn't enough |

**Rule of thumb:** try zero-shot first. If the output's format or judgement is off, add examples. Consider fine-tuning only if many well-chosen examples still don't work, or the prompt becomes too long and expensive.

### Tips for choosing examples

1. **Keep the format identical.** Same labels, same punctuation, same spacing. The model copies what it sees, including mistakes.
2. **Cover the variety.** Include easy and tricky cases, short and long inputs, every label.
3. **Balance the labels.** If 4 of 5 examples say `positive`, the model may lean toward `positive`. This is called **majority label bias**.
4. **Watch the order.** Models can lean toward the label of the **last** example (**recency bias**). Shuffle, or mix the labels.
5. **Make the examples like the real inputs.** Examples about movies won't help much if your queries are about medical notes.
6. **More is not always better.** Each example costs tokens: more money, slower answers, and less room in the context window. After a handful, gains often level off.

### Few-shot with reasoning (chain-of-thought)

Examples can show *how to think*, not just the answer:

```
Q: Ann has 3 apples and buys 2 more. How many?
A: She starts with 3. 3 + 2 = 5. The answer is 5.

Q: Tom has 10 pens and gives away 4. How many?
A:
```

The model tends to write its steps too, which often helps on multi-step problems. Even zero-shot, adding a phrase like "Let's think step by step" can encourage this.

### Does each example's answer need to be correct?

Surprisingly, not always as much as you'd expect. A 2022 study (Min et al.) found that examples help largely by showing the **format, the kind of inputs and the set of possible answers**. Replacing correct labels with random ones often hurt less than expected. Still, use correct examples: wrong ones can hurt, and some models follow them closely.

### Common mistakes

| Mistake | What goes wrong | Fix |
|---|---|---|
| Examples in a different format from the query | the model doesn't know which format to follow | make them identical |
| All examples alike | the model learns the wrong rule (Section 5) | vary them |
| Answer slot not left open | the model may write extra examples, or explanations | end with `Label:` or `→` |
| Expecting the model to "remember" next time | in-context learning doesn't persist | include the examples every time, or fine-tune |
| High temperature for a classification task | random, inconsistent labels | use low temperature or greedy (see [LLM Inference](llm_inference.md)) |

### A different meaning: zero-shot in classic machine learning

Outside LLMs, **zero-shot learning** has an older meaning: recognising **classes never seen in training**. For example, a model that has never seen a zebra is told "a zebra looks like a horse with black-and-white stripes", and then recognises one. The idea is similar (doing something without examples of it), but the method is different. If you see the term in a computer-vision paper, it probably means this.

## 7. Remember this

### Five essential ideas

1. **A shot is an example.** Zero-shot = instruction only. Few-shot = instruction plus a few examples.
2. **No weights change.** It's in-context learning: the examples work only while they're in the prompt.
3. **Examples are just tokens.** The model reads them in prefill, and attention lets the answer use their pattern.
4. **Examples show what words can't.** Format, allowed answers, style and unusual rules.
5. **Choose examples carefully.** Same format, varied cases, balanced labels; and remember that each one costs tokens.

### Exercises

1. Write a zero-shot prompt and a 2-shot prompt that turn a word into its plural (e.g. `cat → cats`).
2. Using the product-code rule, what is the code for `elephant`?
3. A sentiment prompt has 5 examples, all labelled `negative`. What problem do you expect?
4. Is "Let's think step by step", added to a question with no examples, zero-shot or few-shot?

Try the exercises and the quick checks before reading on.

---

### Check your answers

**Quick checks**

1. **No.** The model's weights never changed. The examples only existed in that conversation's prompt.
2. **All the examples have 3 letters**, so the model can't tell whether the number is "the letter count" or "always 3". It might answer `HORSE-3`. Add an example with a different length.

**Exercises**

1. Zero-shot: `Write the plural of the word.` then `cat →`. 2-shot: `Write the plural of the word.` then `dog → dogs`, `box → boxes`, and finally `cat →`. (Choosing `box → boxes` shows that plurals aren't always just "+s".)
2. **ELEPHANT-8.** E-L-E-P-H-A-N-T is 8 letters.
3. **Majority label bias.** The model will lean toward `negative`, even for positive reviews. It might not even realise `positive` is allowed.
4. **Zero-shot.** It adds an instruction, but no examples. (It's called **zero-shot chain-of-thought**.)

## 8. Going further

### Where this connects

- **[LLM Inference](llm_inference.md)**: examples are processed in prefill and stored in the KV cache, so long few-shot prompts cost memory and time.
- **[Attention Blocks](attention_blocks.md)**: the mechanism that lets the query look back at the examples.
- **[Tokenization](tokenization.md)**: how to count what your examples cost.

### Short glossary

| Term | Meaning |
|---|---|
| Shot | one example in the prompt |
| Zero-shot | no examples, only the task description |
| One-shot | one example |
| Few-shot | a few examples |
| Query | the new input to answer |
| In-context learning | doing a task from the prompt alone, without changing weights |
| Fine-tuning | training the model on examples, which changes its weights |
| Chain-of-thought | showing or asking for step-by-step reasoning |
| Majority label bias | leaning toward the label that appears most in the examples |
| Recency bias | leaning toward the label of the last example |

### Learn more

- 📖 [Language Models are Few-Shot Learners](https://arxiv.org/abs/2005.14165) (Brown et al., 2020): the GPT-3 paper that made zero-, one- and few-shot prompting famous
- 📖 [Rethinking the Role of Demonstrations](https://arxiv.org/abs/2202.12837) (Min et al., 2022): what parts of few-shot examples actually matter
- 📖 [Chain-of-Thought Prompting Elicits Reasoning](https://arxiv.org/abs/2201.11903) (Wei et al., 2022): few-shot examples that show reasoning steps
- 📖 [Large Language Models are Zero-Shot Reasoners](https://arxiv.org/abs/2205.11916) (Kojima et al., 2022): the "Let's think step by step" paper
- 📖 [Prompt engineering overview](https://docs.claude.com/en/docs/build-with-claude/prompt-engineering/overview) (Anthropic docs): practical advice on instructions and examples
