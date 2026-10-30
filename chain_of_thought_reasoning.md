# Chain-of-Thought Reasoning: Letting a Model Show Its Work

> **The one idea in this guide:** a language model often gets harder problems right if it **writes out the steps first** and gives the answer last. The steps are extra text it can read back, and extra tokens it gets to compute with.
>
> Everything below explains, demonstrates or deepens that idea.

## 1. In simple words

**Chain-of-thought (CoT)** means the model writes its reasoning **step by step**, like a chain where each link leads to the next, before giving the final answer.

```
Without chain-of-thought            With chain-of-thought
────────────────────────            ─────────────────────
Q: 17 × 24 = ?                      Q: 17 × 24 = ?
A: 418          ✗                   A: 17 × 20 = 340.
                                       17 × 4 = 68.
                                       340 + 68 = 408.
                                       The answer is 408.   ✓
```

**Analogy: showing your work in maths class.** You can't multiply 17 × 24 in your head in one go, but you can on paper: split it into small, easy pieces and write each result down. The paper remembers the in-between results so your head doesn't have to. Chain-of-thought gives the model that sheet of paper.

> **Quick check 1.** In chain-of-thought, does the final answer come before or after the reasoning? Why does that order matter?
>
> *Answers to all checks are in Section 7.*

## 2. What you need to know first

Nothing is required: this guide explains what it uses. For more depth:

- **[LLM Inference](llm_inference.md)**: how a model writes one token at a time, and how temperature and sampling work.
- **[Zero-Shot and Few-Shot Learning](zero_shot_few_shot_learning.md)**: how examples in the prompt shape the answer.

## 3. The key terms

| Term | Meaning |
|---|---|
| Chain-of-thought (CoT) | writing intermediate reasoning steps before the final answer |
| Reasoning step | one small, easy piece of the problem, written down |
| Few-shot CoT | showing examples that include their reasoning steps |
| Zero-shot CoT | asking for steps with a phrase like "Let's think step by step", without examples |
| Self-consistency | generating several chains and taking the most common answer |
| Answer extraction | pulling the final answer out of the reasoning text |
| Faithfulness | whether the written steps truly reflect how the model reached its answer |

## 4. How it works

### Why writing steps helps: two reasons

Recall from [LLM Inference](llm_inference.md) that the model produces **one token at a time**, and each token comes from **one pass** through the model.

**Reason 1: more tokens means more computation.**

Each pass through the model does a fixed amount of work. If the model must write the answer immediately, the whole calculation must fit into the pass that produces the first answer token. If it writes 30 tokens of reasoning first, it gets 30 extra passes to work the problem through.

```
Direct answer:   Q ... =  [one pass]  → "418"
CoT:             Q ... =  [pass] "17" [pass] "×" [pass] "20" ... [pass] "408"
                          many passes before the answer
```

**Reason 2: written steps become memory.**

Once "340" is written, it is part of the text. When the model later writes "340 + 68", **attention** lets it look back at "340" instead of keeping it hidden inside its calculations. Each step only needs to be a small, easy jump from what's already on the page.

> **Key point:** nothing about the model changes. Same weights, same inference loop. CoT just changes **what text it produces before the answer**.

### Three ways to get chain-of-thought

**1. Few-shot CoT: show it.** Give examples whose answers include the steps:

```
Q: A box has 5 red and 3 blue balls. You add 4 blue. How many balls?
A: It starts with 5 + 3 = 8 balls. Adding 4 gives 8 + 4 = 12. The answer is 12.

Q: <your question>
A:
```

The model copies the pattern: steps first, then "The answer is …".

**2. Zero-shot CoT: ask for it.** No examples, just a nudge:

```
Q: <your question>
A: Let's think step by step.
```

This simple phrase was found to improve accuracy on many maths and logic problems (Kojima et al., 2022).

**3. Reasoning models: built in.** Some newer models are **trained** to produce a long chain of reasoning (often called "thinking") before every answer, without being asked. The idea is the same; the habit is just trained in.

## 5. Try it yourself

### A word problem

> A shop has **23** apples. It sells **9**. Then it receives **3** boxes of **6** apples. How many apples does it have now?

**Direct answer.** The model must jump straight to a number. A wrong jump is easy, for example 23 − 9 + 3 = 17 (forgetting the boxes hold 6 each).

**Chain-of-thought.** Work it out with pen and paper:

| Step | Reasoning | Result |
|---|---|---|
| 1 | Start with 23, sell 9: 23 − 9 | 14 |
| 2 | 3 boxes of 6: 3 × 6 | 18 |
| 3 | Add the new apples: 14 + 18 | 32 |
| | **The answer is 32.** | |

Each step is easy on its own. The hard part, keeping track of everything, is handled by the written text.

### Self-consistency: ask several times, then vote

Recall from [LLM Inference](llm_inference.md) that **sampling** with temperature can give different outputs for the same prompt. **Self-consistency** uses that on purpose:

1. Generate several chains of thought (with sampling, so they differ).
2. Take the final answer from each.
3. Pick the answer that appears **most often** (a majority vote).

Five sampled chains for the apple problem:

| Chain | Final answer |
|---|---|
| 1 | 32 |
| 2 | 32 |
| 3 | 26 (made a slip: 14 + 12) |
| 4 | 32 |
| 5 | 17 (forgot the boxes) |

**Vote:** 32 appears 3 times. **The answer is 32**, even though two chains were wrong.

**Why it works, with numbers.** Suppose each chain is right with probability 0.6, independently. With 3 chains, the vote is right when at least 2 are right:

```
all 3 right:     0.6 × 0.6 × 0.6               = 0.216
exactly 2 right: 3 × (0.6 × 0.6 × 0.4)          = 0.432
                                          total = 0.648
```

So the vote is right **64.8%** of the time, better than one chain's 60%. With 5 chains (at least 3 right) it rises to about **68.3%**. In practice wrong chains usually disagree with each other (26, 17 …), which helps the right answer win even more often.

The cost: 5 chains means about 5 times the tokens.

> **Quick check 2.** Three chains answer 12, 15, 12. What does self-consistency choose?

## 6. Variations, tips and mistakes

### When chain-of-thought helps, and when it doesn't

| Task | Does CoT help? | Why |
|---|---|---|
| Multi-step arithmetic, word problems | **a lot** | many small steps to keep track of |
| Logic puzzles, planning, "if … then" questions | **a lot** | each conclusion builds on the last |
| Code that needs careful reasoning | often | thinking through edge cases first |
| Simple facts ("capital of France?") | little | there are no steps to take |
| Very short classification ("positive or negative?") | little, sometimes none | the answer is one quick judgement |

**Model size matters.** The original study (Wei et al., 2022) found that CoT gave large gains mainly in **large** models. Small models often wrote steps that looked reasonable but were wrong, and sometimes did worse than answering directly.

### The costs

- **More tokens:** more money and slower answers. A one-word answer might become a 100-word one.
- **More room used:** long reasoning fills the context window and the KV cache.
- **Harder to parse:** a program reading the output must find the final answer inside the text.

### Tips

1. **Ask for a fixed ending**, like `The answer is …`. It makes answer extraction easy and reliable.
2. **Make example steps small.** Each step in your few-shot examples should be a jump the model can make easily.
3. **Use low temperature for a single chain**, so it doesn't wander. Use higher temperature only for self-consistency, where you *want* different chains.
4. **Match the steps to the problem.** Arithmetic problems want calculations; logic problems want "since … therefore …".
5. **Check the steps, not just the answer**, when the stakes are high.

### A warning: the steps may not be the real reason

The written chain **looks** like the model's thought process, but it isn't guaranteed to be. Studies have found that models can give a confident step-by-step explanation while being influenced by something they never mention (Turpin et al., 2023). And a chain can look perfectly reasonable while containing one wrong step.

So treat the chain as **useful working**, not as proof. This property is called **faithfulness**: a faithful chain truly reflects how the answer was reached.

### Common mistakes

| Mistake | What goes wrong | Fix |
|---|---|---|
| Asking for the answer first, then the reasoning | the answer is fixed before any reasoning happens; the "reasoning" just justifies it | reasoning first, answer last |
| No fixed answer format | a program can't find the answer | end with `The answer is …` |
| One early mistake | every later step builds on it (errors **propagate**) | self-consistency, or ask the model to check its steps |
| Using CoT for everything | slower and more expensive for no gain | save it for multi-step problems |
| Trusting the chain as an explanation | it may not reflect the real cause | verify important claims |

### Related ideas

- **Least-to-most prompting:** first ask the model to split the problem into sub-questions, then answer them in order.
- **Tree of thoughts:** explore several possible next steps at each point, like branches, and keep the promising ones.
- **Self-checking:** after answering, ask the model to review its own steps for mistakes.

## 7. Remember this

### Five essential ideas

1. **Steps first, answer last.** CoT means writing the reasoning before the final answer.
2. **More tokens, more computation.** Each written token is another pass through the model.
3. **Written steps are memory.** Attention lets later steps read earlier results.
4. **Three ways to get it:** show examples (few-shot), ask ("Let's think step by step"), or use a model trained to reason.
5. **Useful, not perfect.** It costs tokens, helps mostly on multi-step problems, and the steps may not be the true reason. Self-consistency (sample and vote) makes it more reliable.

### Exercises

1. Write a chain-of-thought for **15 × 12**, using 15 × 10 and 15 × 2.
2. Find the wrong step: "There are 48 sweets. 15 are eaten, leaving 48 − 15 = 23. They are shared equally by 3 children, so each gets 23 ÷ 3 ≈ 7.7."
3. Five chains answer 8, 8, 9, 7, 8. What does self-consistency choose, and how many chains agreed?
4. Why is "Give the answer, then explain your reasoning" a weaker prompt than "Explain your reasoning, then give the answer"?

Try the exercises and the quick checks before reading on.

---

### Check your answers

**Quick checks**

1. **After.** The model writes tokens in order, so the answer can only use the reasoning if the reasoning is already written.
2. **12**, which appears twice.

**Exercises**

1. 15 × 10 = 150. 15 × 2 = 30. 150 + 30 = 180. **The answer is 180.**
2. **48 − 15 = 33, not 23.** Every later step inherits the mistake. Correct: 33 ÷ 3 = **11** each.
3. **8**, with 3 of the 5 chains agreeing.
4. With the answer first, the answer is chosen **before** any reasoning is written, so the reasoning can't help it. It can only explain an answer already given, right or wrong.

## 8. Going further

### Where this connects

- **[LLM Inference](llm_inference.md)**: the one-token-at-a-time loop that makes extra tokens mean extra computation; temperature and sampling, used by self-consistency.
- **[Zero-Shot and Few-Shot Learning](zero_shot_few_shot_learning.md)**: few-shot CoT is few-shot prompting with reasoning inside the examples.
- **[Attention Blocks](attention_blocks.md)**: how later tokens read the earlier steps.

### Short glossary

| Term | Meaning |
|---|---|
| Chain-of-thought | step-by-step reasoning written before the answer |
| Few-shot CoT | examples that include reasoning steps |
| Zero-shot CoT | asking for steps without examples ("Let's think step by step") |
| Reasoning model | a model trained to reason at length before answering |
| Self-consistency | sample several chains and take a majority vote |
| Answer extraction | finding the final answer inside the text |
| Error propagation | an early mistake carried into every later step |
| Faithfulness | whether the steps truly reflect how the answer was reached |

### Learn more

- 📖 [Chain-of-Thought Prompting Elicits Reasoning in Large Language Models](https://arxiv.org/abs/2201.11903) (Wei et al., 2022): the paper that introduced few-shot chain-of-thought
- 📖 [Large Language Models are Zero-Shot Reasoners](https://arxiv.org/abs/2205.11916) (Kojima et al., 2022): "Let's think step by step"
- 📖 [Self-Consistency Improves Chain of Thought Reasoning](https://arxiv.org/abs/2203.11171) (Wang et al., 2022): sample several chains and vote
- 📖 [Language Models Don't Always Say What They Think](https://arxiv.org/abs/2305.04388) (Turpin et al., 2023): when the written chain isn't the real reason
- 📖 [Prompt engineering overview](https://docs.claude.com/en/docs/build-with-claude/prompt-engineering/overview) (Anthropic docs): practical advice, including asking the model to think step by step
