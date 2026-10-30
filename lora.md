# LoRA (Low-Rank Adaptation): Fine-Tuning With Tiny Add-Ons

> **The one idea in this guide:** to fine-tune a big model, **freeze** its weights and train only a small **update** made from two thin matrices, **B × A**. That update has a tiny fraction of the parameters of the original matrix, so fine-tuning becomes far cheaper, and the result is a small file you can add or remove.
>
> Everything below explains, demonstrates or deepens that idea.

## 1. In simple words

**Fine-tuning** means training an already-trained model a bit more, so it gets better at a specific task or style. Normally that updates **all** its weights: billions of numbers. That needs a lot of memory, and every fine-tuned version is a full copy of the model.

**LoRA** does it differently:

```
Full fine-tuning:   W  →  W + ΔW          (ΔW is as big as W; train all of it)
LoRA:               W  →  W + B × A       (W frozen; train only the thin B and A)
```

**Analogy: sticky notes in a textbook.** Instead of reprinting a whole 1,000-page textbook to update it for your class, you add a few sticky notes with corrections. The book (the frozen model) stays the same; the notes (the LoRA adapter) are small, cheap, and you can swap in a different set for a different class.

> **Quick check 1.** In LoRA, are the original model's weights updated during training?
>
> *Answers to all checks are in Section 7.*

## 2. What you need to know first

- **[Backpropagation](backpropagation.md)**: how training updates weights.
- **Matrix multiplication:** an (m × r) matrix times an (r × n) matrix gives an (m × n) matrix. This guide shows small examples.
- **[In-Context Learning](in_context_learning.md)**: the no-training alternative, useful to compare.

## 3. The key terms

| Term | Meaning |
|---|---|
| Fine-tuning | further training a pretrained model on new data |
| Frozen | not updated during training |
| Weight matrix W | one of the model's big matrices (e.g. in attention), size d × k |
| ΔW ("delta W") | the change added to W |
| Rank r | how many independent "directions" a matrix has; LoRA's r is small, like 4 to 64 |
| Low-rank | built from few directions: ΔW = B × A with a small middle size r |
| Adapter | the trained B and A matrices, saved separately |
| α (alpha) | a scaling setting; the update is multiplied by α ÷ r |
| Merge | add B × A into W once, so there's no extra cost when running |

## 4. How it works

### The core idea

A weight matrix W has size **d × k**. LoRA writes the update as:

```
ΔW = B × A        where  B is d × r   and   A is r × k,   with r much smaller than d and k
```

Multiplying B (tall and thin) by A (short and wide) gives a full d × k matrix, but it's described by far fewer numbers:

| | Number of parameters |
|---|---|
| Full ΔW | d × k |
| LoRA: B and A | d × r + r × k |

### Why is a low-rank update enough?

Research found that the changes needed to adapt a pretrained model to a new task tend to be **simple**: they can be captured well by a few directions. The model already knows a lot; fine-tuning just needs to steer it a little.

### The forward pass

For an input vector x:

```
output = W·x + (α ÷ r) · B·(A·x)
           ▲              ▲
       frozen part     small trained part
```

Computing **A·x** first and then **B·(that)** is cheap, because the middle size r is small.

### Starting at zero

B is usually initialised to **all zeros**, so at the start B × A = 0 and the model behaves exactly like the original. Training then grows the update gradually.

### After training: merge or swap

- **Merge:** compute W' = W + (α ÷ r) · B × A once. Now the model runs at the original speed, with no extra step.
- **Swap:** keep W and store each adapter separately. One base model can serve many tasks by switching small adapters.

## 5. Try it yourself

### A 4 × 4 matrix with rank 1

Let W be a 4 × 4 matrix (frozen). With r = 1:

```
B (4 × 1) = [1, 0, 2, 1]ᵀ        A (1 × 4) = [0.5, 0, −1, 0]
```

**B × A** multiplies each entry of B by the whole row A:

| | col 1 | col 2 | col 3 | col 4 |
|---|---|---|---|---|
| row 1 (B = 1) | 0.5 | 0 | −1 | 0 |
| row 2 (B = 0) | 0 | 0 | 0 | 0 |
| row 3 (B = 2) | 1 | 0 | −2 | 0 |
| row 4 (B = 1) | 0.5 | 0 | −1 | 0 |

A full 4 × 4 update, but described by only **4 + 4 = 8** numbers instead of **16**. Every row is a multiple of A: that's what "rank 1" means.

### Apply it to an input

Take W = the identity matrix (so W·x = x), x = [1, 1, 1, 1], and α ÷ r = 1:

```
A·x     = 0.5·1 + 0·1 + (−1)·1 + 0·1 = −0.5
B·(A·x) = [1, 0, 2, 1] × (−0.5) = [−0.5, 0, −1, −0.5]
W·x     = [1, 1, 1, 1]
output  = [1, 1, 1, 1] + [−0.5, 0, −1, −0.5] = [0.5, 1, 0, 0.5]
```

### Real sizes

One 4096 × 4096 attention matrix:

| | Parameters | Share of full |
|---|---|---|
| Full ΔW | 4096 × 4096 = **16,777,216** | 100% |
| LoRA r = 8 | 4096 × 8 + 8 × 4096 = **65,536** | 0.39% |
| LoRA r = 16 | **131,072** | 0.78% |

Applying r = 8 LoRA to 2 matrices per layer in a 32-layer model:

```
2 × 32 × 65,536 = 4,194,304 trainable parameters   (about 4.2 million)
```

For a 7-billion-parameter model, that's about **0.06%** of the weights. The adapter file is a few megabytes to tens of megabytes, instead of many gigabytes.

> **Quick check 2.** How many LoRA parameters for one 1000 × 1000 matrix with r = 4?

## 6. Variations, tips and mistakes

### Why LoRA saves so much memory

During training, each **trainable** parameter needs extra memory for its gradient and optimiser state (often several times the parameter's own size). With LoRA, only the small adapter needs that. The frozen model just needs to be loaded.

### QLoRA

**QLoRA** goes further: load the frozen base model in **4-bit** ([quantization](quantization.md)), and train LoRA adapters on top in higher precision. This made fine-tuning large models possible on a single GPU.

### Choosing settings

| Setting | Typical values | Effect |
|---|---|---|
| Rank r | 4 to 64 | higher = more capacity, more parameters |
| α | often r or 2r | scales the update's strength |
| Which matrices | attention matrices (query, value, often all); sometimes the MLP too | more matrices = more capacity |
| Learning rate | often higher than for full fine-tuning | the adapter starts at zero and must grow |

### LoRA vs other options

| Option | Changes weights? | Cost | Good for |
|---|---|---|---|
| [In-context learning](in_context_learning.md) | no | none to set up; more tokens per request | quick experiments, small tasks |
| [RAG](rag.md) | no | a search system | adding facts |
| **LoRA** | adds a small adapter | low | style, format, a specialised skill |
| Full fine-tuning | all weights | high | large changes, maximum quality |

### Common mistakes

| Mistake | What goes wrong | Fix |
|---|---|---|
| Expecting LoRA to add lots of new knowledge | it's best at adapting behaviour; facts are better handled by RAG | combine with RAG |
| Rank too low for a hard task | the adapter can't capture the change | raise r, or add more target matrices |
| Using an adapter with a different base model | the update doesn't match the weights | always pair it with the exact base model it was trained on |
| Forgetting α ÷ r | changing r silently changes the update's strength | adjust α together with r |
| Training too long on little data | the model overfits and forgets general skills | fewer steps, more varied data |

## 7. Remember this

### Five essential ideas

1. **Freeze W, train B × A.** Only the small update is learned.
2. **Low-rank = few parameters:** d × r + r × k instead of d × k.
3. **It works because fine-tuning changes are simple** enough to fit in a few directions.
4. **Merge or swap:** merge for full speed, or keep small adapters to switch tasks.
5. **QLoRA = 4-bit base + LoRA**, which makes fine-tuning big models cheap.

### Exercises

1. A 2048 × 2048 matrix with r = 8: how many LoRA parameters? What share of the full matrix is that?
2. B = [2, 1]ᵀ and A = [1, 3]. Compute B × A.
3. Why is B usually initialised to zero?
4. Give one reason to keep adapters separate instead of merging.

Try the exercises and the quick checks before reading on.

---

### Check your answers

**Quick checks**

1. **No.** W is frozen; only B and A are trained.
2. 1000 × 4 + 4 × 1000 = **8,000** (instead of 1,000,000).

**Exercises**

1. 2048 × 8 + 8 × 2048 = **32,768**. Full: 2048 × 2048 = 4,194,304. Share: 32,768 ÷ 4,194,304 ≈ **0.78%**.
2. **[[2, 6], [1, 3]]**: the first row is 2 × A, the second is 1 × A.
3. So that B × A = 0 at the start, and the model begins exactly as the original.
4. One base model can serve many tasks by swapping small adapters, instead of storing many full copies.

## 8. Going further

### Where this connects

- **[Quantization](quantization.md)**: QLoRA combines 4-bit base weights with LoRA.
- **[Distillation](distillation.md)**: LoRA is a cheap way to train a student on a teacher's outputs.
- **[Attention Blocks](attention_blocks.md)**: the query, key and value matrices are the usual LoRA targets.

### Short glossary

| Term | Meaning |
|---|---|
| LoRA | low-rank adaptation: train B × A, keep W frozen |
| Rank r | the small middle size of B and A |
| Adapter | the saved B and A matrices |
| α ÷ r | the scaling of the update |
| Merge | add the update into W permanently |
| QLoRA | LoRA on a 4-bit quantized base model |
| PEFT | parameter-efficient fine-tuning: the family LoRA belongs to |

### Learn more

- 📖 [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685) (Hu et al., 2021): the original paper
- 📖 [QLoRA: Efficient Finetuning of Quantized LLMs](https://arxiv.org/abs/2305.14314) (Dettmers et al., 2023)
