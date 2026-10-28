# MLP Inference: Solving XOR with a Trained Network

> **The one idea in this guide:** a trained neural network answers a question by pushing numbers through its layers: **multiply, add, bend, repeat**. No learning happens; the weights are already fixed.

## 1. In simple words

An **MLP** (multi-layer perceptron) is the simplest kind of neural network: layers of "neurons", where each neuron takes some numbers in, mixes them, and passes one number on.

**Inference** means *using* a network that has **already been trained**. Its weights (the numbers inside it) are fixed. We just feed in an input and read the answer.

Our task is **XOR** ("exclusive or"): the answer is 1 when **exactly one** of the two inputs is 1.

| x₁ | x₂ | XOR |
|---|---|---|
| 0 | 0 | 0 |
| 0 | 1 | 1 |
| 1 | 0 | 1 |
| 1 | 1 | 0 |

**Analogy: two light switches on a staircase.** The light is on when the switches are in *different* positions. If both are down or both are up, it's off. That's XOR.

```
input (x₁, x₂)  →  hidden layer  →  output layer  →  probability  →  answer 0 or 1
```

## 2. What you need to know first

Nothing beyond basic arithmetic. For more context:

- **[Attention Blocks](attention_blocks.md)**: the "feed-forward network" inside every transformer block is exactly the kind of MLP explained here.

## 3. Why XOR needs a hidden layer

Plot the four inputs as points, marked by their answer (● = 1, ○ = 0):

```
x₂
 1 │  ●(0,1)        ○(1,1)
   │
 0 │  ○(0,0)        ●(1,0)
   └──────────────────────── x₁
       0              1
```

The ● points sit on one diagonal, and the ○ points on the other. **No single straight line can separate them.** Try it: any line that puts both ● on one side puts at least one ○ there too.

A network with no hidden layer can only draw **one straight line**, so it can't solve XOR. A **hidden layer** fixes this: it first moves the points to new positions where one straight line *does* work. Section 5 shows this happening.

## 4. The trained network

### Its shape

```
 inputs          hidden layer         output
                 (2 neurons)          (1 neuron)

  x₁ ──┬──────▶  h₁ ──┐
       │   ╲  ╱       ├──────▶  z ──▶ sigmoid ──▶ probability
       │    ╳         │
  x₂ ──┴──────▶  h₂ ──┘
```

Every input connects to every hidden neuron, and every hidden neuron connects to the output.

### Its weights (already fitted)

> **Note:** These weights are hand-picked so the arithmetic is easy and the network solves XOR exactly. A network trained by a computer would end up with different, messier numbers that do the same job.

**Hidden layer**

| Neuron | Weight from x₁ | Weight from x₂ | Bias |
|---|---|---|---|
| h₁ | 1 | 1 | 0 |
| h₂ | 1 | 1 | −1 |

**Output layer**

| Neuron | Weight from h₁ | Weight from h₂ | Bias |
|---|---|---|---|
| z | 4 | −8 | −2 |

### The four steps of inference

For each input, the network does:

1. **Hidden layer, mix:** each hidden neuron computes (weight from x₁)·x₁ + (weight from x₂)·x₂ + bias.
2. **Hidden layer, bend (ReLU):** replace any negative number with 0. In symbols: ReLU(a) = max(0, a).
3. **Output layer, mix:** z = 4·h₁ + (−8)·h₂ + (−2). This **z** is a raw score, called a **logit**.
4. **Probability (sigmoid):** squash z into a number between 0 and 1:

```
sigmoid(z) = 1 / (1 + e^(−z))          (e ≈ 2.718)
```

**Answer:** if the probability is at least 0.5, answer **1**; otherwise answer **0**.

### The formulas

**Symbols**

| Symbol | Meaning | Value in our network |
|---|---|---|
| x₁, x₂ | the two inputs | 0 or 1 |
| w₁₁, w₂₁ | weights from x₁ and x₂ into h₁ | 1, 1 |
| w₁₂, w₂₂ | weights from x₁ and x₂ into h₂ | 1, 1 |
| d₁, d₂ | biases of h₁ and h₂ | 0, −1 |
| v₁, v₂ | weights from h₁ and h₂ into the output | 4, −8 |
| c | bias of the output | −2 |
| a₁, a₂ | hidden values before ReLU | |
| h₁, h₂ | hidden values after ReLU | |
| z | output logit | |
| p | output probability | |
| ŷ | the answer (0 or 1) | |

**Step 1: hidden layer, mix**

```
a₁ = w₁₁·x₁ + w₂₁·x₂ + d₁          →   a₁ = x₁ + x₂
a₂ = w₁₂·x₁ + w₂₂·x₂ + d₂          →   a₂ = x₁ + x₂ − 1
```

**Step 2: ReLU**

```
ReLU(a) = max(0, a)

h₁ = ReLU(a₁) = max(0, a₁)
h₂ = ReLU(a₂) = max(0, a₂)
```

**Step 3: output layer, mix**

```
z = v₁·h₁ + v₂·h₂ + c               →   z = 4·h₁ − 8·h₂ − 2
```

**Step 4: sigmoid**

```
σ(z) = 1 / (1 + e^(−z))

p = σ(z)
```

**Step 5: decision**

```
ŷ = 1   if p ≥ 0.5
ŷ = 0   if p < 0.5
```

(Because σ(0) = 0.5, this is the same as: ŷ = 1 if z ≥ 0, otherwise 0.)

**Everything in one formula**

```
p = σ( v₁·max(0, w₁₁x₁ + w₂₁x₂ + d₁) + v₂·max(0, w₁₂x₁ + w₂₂x₂ + d₂) + c )
```

With our weights:

```
p = σ( 4·max(0, x₁ + x₂) − 8·max(0, x₁ + x₂ − 1) − 2 )
```

**Matrix form** (used in Section 5 for all inputs at once)

```
x  = [x₁, x₂]                       1 × 2    (one input as a row)

W₁ = [[w₁₁, w₁₂],    = [[1, 1],     2 × 2    (column j = weights into hⱼ)
      [w₂₁, w₂₂]]       [1, 1]]
b₁ = [d₁, d₂]        = [0, −1]      1 × 2

W₂ = [[v₁],          = [[ 4],       2 × 1
      [v₂]]             [−8]]
b₂ = c               = −2

h = ReLU(x W₁ + b₁)                 1 × 2
z = h W₂ + b₂                       1 × 1
p = σ(z)                            1 × 1
```

All together: **p = σ( ReLU(x W₁ + b₁) W₂ + b₂ )**.

## 5. Try it yourself: inference by hand

> All four inputs are worked out in full, with no steps skipped. Results are rounded to 3 decimals.

### Input 1: (0, 0)

Both switches are down, so XOR should say **0**.

**Step 1: hidden layer, mix**

```
h₁ = 1·x₁ + 1·x₂ + 0
   = 1·0  + 1·0  + 0
   = 0    + 0    + 0
   = 0

h₂ = 1·x₁ + 1·x₂ + (−1)
   = 1·0  + 1·0  − 1
   = 0    + 0    − 1
   = −1
```

**Step 2: ReLU**

```
h₁ = max(0,  0) = 0
h₂ = max(0, −1) = 0      ← the negative value becomes 0
```

**Step 3: output layer, mix**

```
z = 4·h₁ + (−8)·h₂ + (−2)
  = 4·0  + (−8)·0  − 2
  = 0    + 0       − 2
  = −2
```

**Step 4: sigmoid**

```
−z           = −(−2)        = 2
e^(−z)       = e^2          ≈ 7.389
1 + e^(−z)   = 1 + 7.389    = 8.389
probability  = 1 / 8.389    ≈ 0.119
```

**Answer:** 0.119 < 0.5, so the network answers **0**. ✅ Correct.

### Input 2: (0, 1)

One switch up, so XOR should say **1**.

**Step 1: hidden layer, mix**

```
h₁ = 1·0 + 1·1 + 0     = 0 + 1 + 0     = 1
h₂ = 1·0 + 1·1 + (−1)  = 0 + 1 − 1     = 0
```

**Step 2: ReLU**

```
h₁ = max(0, 1) = 1
h₂ = max(0, 0) = 0
```

**Step 3: output layer, mix**

```
z = 4·1 + (−8)·0 + (−2)
  = 4   + 0      − 2
  = 2
```

**Step 4: sigmoid**

```
−z           = −2
e^(−z)       = e^(−2)       ≈ 0.135
1 + e^(−z)   = 1 + 0.135    = 1.135
probability  = 1 / 1.135    ≈ 0.881
```

**Answer:** 0.881 ≥ 0.5, so the network answers **1**. ✅ Correct.

### Input 3: (1, 0)

One switch up, so XOR should say **1**.

**Step 1: hidden layer, mix**

```
h₁ = 1·1 + 1·0 + 0     = 1 + 0 + 0     = 1
h₂ = 1·1 + 1·0 + (−1)  = 1 + 0 − 1     = 0
```

**Step 2: ReLU**

```
h₁ = max(0, 1) = 1
h₂ = max(0, 0) = 0
```

**Step 3: output layer, mix**

```
z = 4·1 + (−8)·0 + (−2)
  = 4   + 0      − 2
  = 2
```

**Step 4: sigmoid**

```
−z           = −2
e^(−z)       = e^(−2)       ≈ 0.135
1 + e^(−z)   = 1 + 0.135    = 1.135
probability  = 1 / 1.135    ≈ 0.881
```

**Answer:** 0.881 ≥ 0.5, so the network answers **1**. ✅ Correct.

### Input 4: (1, 1): the tricky case

Both switches up, so XOR should say **0**.

**Step 1: hidden layer, mix**

```
h₁ = 1·1 + 1·1 + 0     = 1 + 1 + 0     = 2
h₂ = 1·1 + 1·1 + (−1)  = 1 + 1 − 1     = 1
```

**Step 2: ReLU**

```
h₁ = max(0, 2) = 2
h₂ = max(0, 1) = 1
```

**Step 3: output layer, mix**

```
z = 4·2 + (−8)·1 + (−2)
  = 8   − 8      − 2
  = −2
```

**Step 4: sigmoid**

```
−z           = −(−2)        = 2
e^(−z)       = e^2          ≈ 7.389
1 + e^(−z)   = 1 + 7.389    = 8.389
probability  = 1 / 8.389    ≈ 0.119
```

**Answer:** 0.119 < 0.5, so the network answers **0**. ✅ Correct.

Here, h₂ is what saves the day: it only "switches on" when **both** inputs are 1, and its weight −8 pulls z down to −2.

### Summary of all four

| Input | Mix (h₁, h₂) | After ReLU (h₁, h₂) | z | Probability | Answer | XOR |
|---|---|---|---|---|---|---|
| (0, 0) | (0, −1) | (0, 0) | −2 | 0.119 | 0 | 0 ✅ |
| (0, 1) | (1, 0) | (1, 0) | 2 | 0.881 | 1 | 1 ✅ |
| (1, 0) | (1, 0) | (1, 0) | 2 | 0.881 | 1 | 1 ✅ |
| (1, 1) | (2, 1) | (2, 1) | −2 | 0.119 | 0 | 0 ✅ |

All four are correct. ReLU changed a value in only one place: the −1 for input (0, 0).

### What the hidden layer did

The hidden layer **moved** the four points to new positions (h₁, h₂), taken from the "After ReLU" column:

| Input | New position (h₁, h₂) | XOR |
|---|---|---|
| (0, 0) | (0, 0) | 0 |
| (0, 1) | (1, 0) | 1 |
| (1, 0) | (1, 0) | 1 |
| (1, 1) | (2, 1) | 0 |

```
h₂
 1 │                    ○(2,1)
   │
 0 │  ○(0,0)   ●(1,0) ← both "1" inputs land here
   └────────────────────────── h₁
      0        1        2
```

The two "1" inputs landed on **the same spot**, (1, 0). Now a single straight line separates the classes: the line where z = 0.

Check each new point against z = 4·h₁ − 8·h₂ − 2:

```
(0, 0):  4·0 − 8·0 − 2 = 0 − 0 − 2 = −2   → negative side → class 0
(1, 0):  4·1 − 8·0 − 2 = 4 − 0 − 2 =  2   → positive side → class 1
(2, 1):  4·2 − 8·1 − 2 = 8 − 8 − 2 = −2   → negative side → class 0
```

**The hidden layer turned an impossible problem into an easy one.**

### Doing all four at once (matrix form)

Real code processes many inputs together. The formulas are the same as in Section 4, with the single row x replaced by a matrix X that has one row per input:

```
H = ReLU(X W₁ + b₁)
z = H W₂ + b₂
P = σ(z)
```

Stack the inputs as the rows of a matrix X:

```
X = [[0, 0],
     [0, 1],
     [1, 0],
     [1, 1]]            4 × 2   (4 inputs, 2 numbers each)

W₁ = [[1, 1],           b₁ = [0, −1]
      [1, 1]]           2 × 2   (column 1 = weights of h₁, column 2 = weights of h₂)

W₂ = [[ 4],             b₂ = −2
      [−8]]             2 × 1
```

**Step 1: X W₁.** Shape: (4 × **2**)(**2** × 2) → **4 × 2**. Each entry is a row of X times a column of W₁:

```
row (0,0):  column 1: 0·1 + 0·1 = 0      column 2: 0·1 + 0·1 = 0
row (0,1):  column 1: 0·1 + 1·1 = 1      column 2: 0·1 + 1·1 = 1
row (1,0):  column 1: 1·1 + 0·1 = 1      column 2: 1·1 + 0·1 = 1
row (1,1):  column 1: 1·1 + 1·1 = 2      column 2: 1·1 + 1·1 = 2

X W₁ = [[0, 0],
        [1, 1],
        [1, 1],
        [2, 2]]
```

**Step 1b: add b₁ = [0, −1] to every row.** Shape stays **4 × 2**:

```
[0, 0] + [0, −1] = [0, −1]
[1, 1] + [0, −1] = [1,  0]
[1, 1] + [0, −1] = [1,  0]
[2, 2] + [0, −1] = [2,  1]
```

**Step 2: ReLU, entry by entry.** Shape stays **4 × 2**:

```
H = [[max(0,0), max(0,−1)],      [[0, 0],
     [max(0,1), max(0, 0)],   =   [1, 0],
     [max(0,1), max(0, 0)],       [1, 0],
     [max(0,2), max(0, 1)]]       [2, 1]]
```

**Step 3: H W₂ + b₂.** Shape: (4 × **2**)(**2** × 1) → **4 × 1**:

Each row of H is multiplied by W₂, then b₂ is added:

```
input (0,0), H row [0, 0]:  0·4 + 0·(−8) − 2 = 0 + 0 − 2 = −2
input (0,1), H row [1, 0]:  1·4 + 0·(−8) − 2 = 4 + 0 − 2 =  2
input (1,0), H row [1, 0]:  1·4 + 0·(−8) − 2 = 4 + 0 − 2 =  2
input (1,1), H row [2, 1]:  2·4 + 1·(−8) − 2 = 8 − 8 − 2 = −2

z = [[−2],
     [ 2],
     [ 2],
     [−2]]
```

**Step 4: sigmoid, entry by entry.** Shape stays **4 × 1**:

```
sigmoid(−2) = 1 / (1 + e^2)    = 1 / (1 + 7.389) = 1 / 8.389 ≈ 0.119
sigmoid( 2) = 1 / (1 + e^(−2)) = 1 / (1 + 0.135) = 1 / 1.135 ≈ 0.881

probabilities = [[0.119],
                 [0.881],
                 [0.881],
                 [0.119]]
```

Same answers as before, all four computed at once. Each row of the result is the answer for one input.

## 6. Variations and common mistakes

### Why ReLU is essential

Remove ReLU and redo input (0, 0):

**Step 1: hidden layer, mix** (same as before)

```
h₁ = 1·0 + 1·0 + 0     = 0
h₂ = 1·0 + 1·0 + (−1)  = −1
```

**Step 2: no ReLU.** The values stay as they are: h₁ = 0, h₂ = −1.

**Step 3: output layer, mix**

```
z = 4·0 + (−8)·(−1) + (−2)
  = 0   + 8         − 2
  = 6
```

**Step 4: sigmoid**

```
−z           = −6
e^(−z)       = e^(−6)       ≈ 0.002
1 + e^(−z)   = 1 + 0.002    = 1.002
probability  = 1 / 1.002    ≈ 0.998
```

**Answer:** 0.998 ≥ 0.5, so the network answers **1**. ❌ Wrong: XOR says 0.

Without the "bend", two layers of mixing collapse into **one** straight-line mix, and a single straight line can't solve XOR. The non-linear step is what gives the hidden layer its power.

### Other activation functions

ReLU isn't the only choice. **Sigmoid** and **tanh** are smooth "S-shaped" alternatives for hidden layers. What matters is that the function is **non-linear**, not a straight line.

### Inputs that aren't exactly 0 or 1

The network will answer any input, even ones it never saw. Take (0.9, 0.1):

**Step 1: hidden layer, mix**

```
h₁ = 1·0.9 + 1·0.1 + 0     = 0.9 + 0.1 + 0     = 1.0
h₂ = 1·0.9 + 1·0.1 + (−1)  = 0.9 + 0.1 − 1     = 0.0
```

**Step 2: ReLU**

```
h₁ = max(0, 1.0) = 1.0
h₂ = max(0, 0.0) = 0.0
```

**Step 3: output layer, mix**

```
z = 4·1.0 + (−8)·0.0 + (−2)
  = 4     + 0        − 2
  = 2
```

**Step 4: sigmoid**

```
−z           = −2
e^(−z)       = e^(−2)       ≈ 0.135
1 + e^(−z)   = 1 + 0.135    = 1.135
probability  = 1 / 1.135    ≈ 0.881
```

**Answer:** 0.881 ≥ 0.5, so the network answers **1**. That's sensible: the input is close to (1, 0).

### Common mistakes

- **"The network learns while answering."** No. During inference, the weights never change. Learning (training) is a separate process that happens beforehand.
- **"The probability is the answer."** Not quite. 0.881 is the network's confidence; the **threshold** (0.5) turns it into the answer 1.
- **"0.881 means the network is unsure."** It only reflects how sigmoid squashes z = 2. Weights that produced larger logits, such as ±6, would give probabilities closer to 1 or 0.
- **"More neurons always helps."** For XOR, 2 hidden neurons are enough. Bigger networks matter for harder problems.

## 7. Remember this

- **Inference = a forward pass with fixed weights.** No learning happens.
- Each layer does: **mix** (weights · inputs + bias), then **bend** (activation like ReLU).
- The output **logit** goes through **sigmoid** to become a **probability**, and a **threshold** turns it into an answer.
- XOR can't be solved by one straight line; the **hidden layer** moves the points so one line works.
- Without a **non-linear activation**, extra layers add nothing.

### Formula sheet

```
Hidden mix:     a = x W₁ + b₁
ReLU:           h = max(0, a)
Output mix:     z = h W₂ + b₂
Sigmoid:        p = σ(z) = 1 / (1 + e^(−z))
Decision:       ŷ = 1 if p ≥ 0.5, else 0

All in one:     p = σ( ReLU(x W₁ + b₁) W₂ + b₂ )
Our network:    p = σ( 4·max(0, x₁ + x₂) − 8·max(0, x₁ + x₂ − 1) − 2 )
Batch:          P = σ( ReLU(X W₁ + b₁) W₂ + b₂ ),   X is n × 2, P is n × 1
```

### Quick check

1. Compute the network's answer for input (0.5, 0.5), step by step.
2. You feed in a batch of 10 inputs. What are the shapes of X, the hidden layer and the output?
3. Why did removing ReLU only break the (0, 0) case?

Try answering before reading on.

---

### Check your answers

**1. Input (0.5, 0.5)**

```
Step 1:  h₁ = 1·0.5 + 1·0.5 + 0     = 0.5 + 0.5 + 0  = 1.0
         h₂ = 1·0.5 + 1·0.5 + (−1)  = 0.5 + 0.5 − 1  = 0.0
Step 2:  h₁ = max(0, 1.0) = 1.0
         h₂ = max(0, 0.0) = 0.0
Step 3:  z  = 4·1.0 + (−8)·0.0 + (−2) = 4 + 0 − 2 = 2
Step 4:  −z          = −2
         e^(−z)      = e^(−2)     ≈ 0.135
         1 + e^(−z)  = 1 + 0.135  = 1.135
         probability = 1 / 1.135  ≈ 0.881
Answer:  0.881 ≥ 0.5  →  1
```

XOR is only defined for 0 and 1, so there's no "right" answer here. The network still gives one, because inference always produces an output.

**2. Shapes**

```
X:       10 × 2
X W₁:   (10 × 2)(2 × 2) → 10 × 2      (hidden layer, before and after ReLU)
H W₂:   (10 × 2)(2 × 1) → 10 × 1      (output)
```

The weights stay 2 × 2 and 2 × 1, whatever the batch size.

**3. Why only (0, 0)?** From the summary table, the hidden values before ReLU are (0, −1), (1, 0), (1, 0) and (2, 1). The only negative number is the −1 for input (0, 0). ReLU only changes negative numbers, so removing it only matters there.

## 8. Going further

### Where MLPs show up

- **Inside transformers:** every block in the [Attention Blocks](attention_blocks.md) guide has a feed-forward network, FFN(x) = max(0, x W₁ + b₁) W₂ + b₂. That's the same mix → ReLU → mix you just computed, applied to every token.
- **In the [LLM Inference](llm_inference.md) and [ViT](vit.md) guides:** the final step also turns a vector into scores (logits) and then probabilities. There, softmax plays the role sigmoid plays here, for more than two classes.

### Training (not covered here)

How do the weights get their values? By **training**: start with random weights, measure how wrong the answers are, and nudge the weights to be less wrong, many thousands of times. That's a topic for its own guide.

### Learn more

- 📖 [Neural Networks and Deep Learning, Chapter 1](http://neuralnetworksanddeeplearning.com/chap1.html) (Michael Nielsen): a free, beginner-friendly introduction to neurons, sigmoid and network layers
- 📖 [Understanding neural networks with TensorFlow Playground](https://cloud.google.com/blog/products/gcp/understanding-neural-networks-with-tensorflow-playground) (Google Cloud blog): why some data can't be split by one line, and how hidden layers help
- 🧪 [TensorFlow Playground](https://playground.tensorflow.org): train a small network in your browser and watch the hidden layer at work
