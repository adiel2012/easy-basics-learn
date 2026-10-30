# Distillation: A Big Model Teaches a Small One

> **The one idea in this guide:** in **knowledge distillation**, a large, strong **teacher** model trains a small **student** model. The student learns to copy the teacher's **full probability outputs**, not just the right answers, because those probabilities carry extra knowledge about how the classes relate.
>
> Everything below explains, demonstrates or deepens that idea.

## 1. In simple words

Big models are accurate but slow and expensive. Small models are fast and cheap, but less accurate if trained the normal way.

**Distillation** gets the best of both: train the small model to **imitate** the big one.

```
                 ┌──────────────┐
  input  ───────▶│ teacher (big)│──▶ probabilities: cat 0.84, dog 0.11, fox 0.04
    │            └──────────────┘                      │
    │                                                  │  student learns to match these
    │            ┌───────────────┐                     ▼
    └───────────▶│ student (small)│──▶ probabilities: cat ?, dog ?, fox ?
                 └───────────────┘
```

**Analogy: learning from a tutor vs an answer key.** An answer key only says "the answer is B". A good tutor says "it's B, though C is a reasonable guess, and A is clearly wrong". The tutor's extra hints teach you **how** to think about the problem. The teacher model's probabilities are those hints.

> **Quick check 1.** In distillation, which model is trained: the teacher or the student?
>
> *Answers to all checks are in Section 7.*

## 2. What you need to know first

- **[Activation Functions](activation_functions.md)**: softmax, which turns scores into probabilities.
- **[Cross-Entropy](cross_entropy.md)** and **[KL Divergence](kl_divergence.md)**: ways to measure how different two probability distributions are.
- **[LLM Inference](llm_inference.md)**: logits and temperature.

## 3. The key terms

| Term | Meaning |
|---|---|
| Teacher | the large, trained model whose behaviour is copied; it is **not** changed |
| Student | the smaller model being trained |
| Hard label | the single correct answer, e.g. "cat" (probability 1 on one class) |
| Soft targets | the teacher's full probabilities over all classes |
| Temperature (T) | divides logits before softmax; higher T makes probabilities softer (flatter) |
| Dark knowledge | the information hidden in the teacher's small probabilities, such as "dog is more like cat than fox is" |
| Distillation loss | how far the student's soft probabilities are from the teacher's |

## 4. How it works

### Why soft targets teach more

Take a photo of a cat. The **hard label** says only: cat = 1, everything else = 0.

The **teacher** might say: cat 0.84, dog 0.11, fox 0.04. That also tells the student:

- dogs look somewhat like this cat (0.11),
- foxes look less like it (0.04).

That **similarity structure** is information the hard label throws away. A student that learns it often generalises better than one trained on hard labels alone.

### Temperature makes the hints visible

The teacher is often very confident, so the small probabilities are tiny and carry little signal. Dividing the logits by a **temperature T > 1** before softmax spreads probability out, making those hints larger and easier to learn from. (Same idea as temperature in [LLM Inference](llm_inference.md).)

### The training loss

The student is usually trained on a mix of two losses:

```
loss = α × (hard loss: student vs true label, at T = 1)
     + (1 − α) × T² × (soft loss: student vs teacher, both at temperature T)
```

- **Hard loss:** ordinary [cross-entropy](cross_entropy.md) with the correct answer.
- **Soft loss:** usually the [KL divergence](kl_divergence.md) from the teacher's softened probabilities to the student's.
- **α** (between 0 and 1) balances the two.
- **T²**: softening by T makes the soft loss's gradients about T² times smaller, so multiplying by T² keeps the two parts balanced.

Only the **student's** weights are updated. The teacher just produces targets.

## 5. Try it yourself

### Three classes: cat, dog, fox

The input is a photo of a **cat**.

**Teacher logits:** [4, 2, 1]

Teacher probabilities at different temperatures (softmax of logits ÷ T):

| Class | T = 1 | T = 2 | T = 4 |
|---|---|---|---|
| cat | 0.844 | 0.629 | 0.481 |
| dog | 0.114 | 0.231 | 0.292 |
| fox | 0.042 | 0.140 | 0.227 |

At T = 1 the hints are small (dog 0.114, fox 0.042). At T = 2 they're much clearer. Check T = 2 by hand:

```
logits ÷ 2 = [2, 1, 0.5]
e^2 = 7.389,  e^1 = 2.718,  e^0.5 = 1.649      sum = 11.756
cat = 7.389 ÷ 11.756 = 0.629
dog = 2.718 ÷ 11.756 = 0.231
fox = 1.649 ÷ 11.756 = 0.140
```

**Student logits (before training):** [3, 1, 1.5]

Student probabilities at T = 2:

| Class | Teacher (T = 2) | Student (T = 2) |
|---|---|---|
| cat | 0.629 | 0.543 |
| dog | **0.231** | **0.200** |
| fox | **0.140** | **0.257** |

The student already gets the top answer right (cat). But it thinks **fox** is more likely than **dog**, while the teacher says the opposite. A hard label ("cat") would never correct this. The soft loss will.

**The losses:**

| Part | Value |
|---|---|
| Hard loss: −log(student's P(cat) at T = 1) | 0.306 |
| Soft loss: KL(teacher ‖ student) at T = 2 | 0.040 |
| Total with α = 0.5: 0.5 × 0.306 + 0.5 × 2² × 0.040 = 0.153 + 0.080 | ≈ 0.233 |

Training lowers this total. The soft part specifically pushes the student's dog probability up and its fox probability down, toward the teacher's view.

> **Quick check 2.** Raising the temperature makes the teacher's probabilities (a) sharper or (b) flatter?

## 6. Variations, tips and mistakes

### Kinds of distillation

| Kind | What the student copies | Example |
|---|---|---|
| **Logit (response) distillation** | the teacher's output probabilities | the classic method above |
| **Feature distillation** | the teacher's internal vectors | match hidden layers |
| **Sequence-level distillation (for LLMs)** | text generated by the teacher | train a small LLM on a big LLM's answers |
| **Self-distillation** | a model teaches a copy of itself | sometimes improves accuracy |

For LLMs, **sequence-level** distillation is very common: generate many high-quality answers with a large model, then fine-tune a small model on them. If you have the teacher's token probabilities too, you can use them as soft targets for each next token.

### Distillation vs quantization vs pruning

| Method | How it makes a model smaller |
|---|---|
| **Distillation** | trains a different, smaller model |
| **[Quantization](quantization.md)** | stores the same model's numbers with fewer bits |
| **Pruning** | removes weights or parts of the same model |

They combine well: distil a small student, then quantize it.

### Tips and mistakes

| Tip / mistake | Why |
|---|---|
| The student can't exceed its capacity | a tiny student can't copy everything a huge teacher knows |
| Too big a gap can hurt | sometimes a medium "teaching assistant" model in between helps |
| Use a temperature above 1 for soft targets | at T = 1 the hints may be too small to learn from |
| Use the same T for teacher and student in the soft loss | otherwise you compare differently-shaped distributions |
| Remember the T² factor | without it, the soft loss is too weak at high T |
| Check licences | some model providers don't allow using their outputs to train other models |
| The teacher's mistakes are copied too | quality of the teacher matters |

## 7. Remember this

### Five essential ideas

1. **Teacher → student.** A big trained model provides targets; only the small model is trained.
2. **Soft targets carry more than the answer:** how similar the wrong options are (dark knowledge).
3. **Temperature softens** the teacher's probabilities so the hints are visible.
4. **Mixed loss:** hard loss with the true label + T² × soft loss with the teacher.
5. **For LLMs, students often learn from the teacher's generated text**, and it combines well with quantization.

### Exercises

1. Teacher logits [2, 0]. Compute the probabilities at T = 1 and T = 2.
2. Why can't a hard label teach the student that "dog" is more similar to "cat" than "fox" is?
3. With α = 1, what does the loss become?
4. Name one difference between distillation and quantization.

Try the exercises and the quick checks before reading on.

---

### Check your answers

**Quick checks**

1. **The student.** The teacher stays fixed.
2. **(b) Flatter.**

**Exercises**

1. T = 1: e² = 7.389, e⁰ = 1, sum 8.389 → **[0.881, 0.119]**. T = 2: logits ÷ 2 = [1, 0]; e¹ = 2.718, sum 3.718 → **[0.731, 0.269]**.
2. A hard label gives 0 to every wrong class, so "dog" and "fox" look equally wrong.
3. Only the **hard loss**: ordinary training on true labels, with no distillation.
4. Distillation trains a **different, smaller model**; quantization keeps the **same model** and stores its numbers with fewer bits.

## 8. Going further

### Where this connects

- **[Quantization](quantization.md)**: shrinking the numbers instead of the model.
- **[LoRA](lora.md)**: a cheap way to fine-tune a student on teacher outputs.
- **[KL Divergence](kl_divergence.md)**: the standard soft loss.

### Short glossary

| Term | Meaning |
|---|---|
| Knowledge distillation | training a small student to imitate a big teacher |
| Teacher / student | the model copied / the model trained |
| Soft targets | the teacher's full probabilities |
| Hard label | the single correct class |
| Dark knowledge | information in the teacher's small probabilities |
| Temperature | softens probabilities when above 1 |
| Sequence-level distillation | training on the teacher's generated text |

### Learn more

- 📖 [Distilling the Knowledge in a Neural Network](https://arxiv.org/abs/1503.02531) (Hinton, Vinyals and Dean, 2015): the classic paper
- 📖 [DistilBERT](https://arxiv.org/abs/1910.01108) (Sanh et al., 2019): a well-known distilled language model
