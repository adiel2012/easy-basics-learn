# Backpropagation: How a Neural Network Learns

> **The one idea in this guide:** to learn, a network measures **how wrong** its answer is, works **backwards** to find how the error depends on **each weight**, and nudges every weight a little in the direction that makes the error smaller. Then it repeats, many times.

## 1. In simple words

In the [MLP Inference](mlp_xor_inference.md) guide, the weights were already given. But where do good weights come from? A network starts with **initial weights**, often chosen randomly, and may give inaccurate answers. **Training** improves the weights, step by step.

Each training step has four parts:

1. **Forward pass:** run the network on an example and get its answer.
2. **Loss:** measure how wrong the answer is, as one number.
3. **Backward pass (backpropagation):** work backwards through the network to find, for **every weight**, how the loss would change if that weight changed a little. These numbers are called **gradients**.
4. **Update:** move each weight a small step in the direction that **lowers** the loss.

**Analogy: walking down a hill in fog.** You can't see the valley, but you can feel the slope under your feet. The loss is your height. The gradient points **uphill** (the direction of steepest increase), so you step the **opposite** way, downhill. The learning rate is your step size. Backpropagation is how you feel the slope for **every** weight at once; the update is the step itself.

```
forward  →  loss  →  backward  →  update  →  (repeat)
```

## 2. What you need to know first

- **[MLP Inference: XOR](mlp_xor_inference.md)**: how the forward pass works. This guide uses the same network shape and the same symbols.
- **Basic derivatives**: a derivative tells you how much an output changes when an input changes a little. Section 4 lists the few rules you need.

## 3. The key terms

| Term | Meaning |
|---|---|
| **Training example** | one input with its correct answer, e.g. x = (1, 0) with y = 1 |
| **Loss** | one number that measures how wrong the prediction is (lower is better) |
| **Gradient** | how the loss changes when one parameter changes a little: ∂L/∂(parameter) |
| **Chain rule** | the rule for combining derivatives step by step |
| **Backpropagation** | applying the chain rule backward through the network to calculate how the loss depends on each weight and bias |
| **Learning rate (η)** | the step size: how far each parameter moves in one update |
| **Gradient descent** | the update rule that uses the gradients: parameter ← parameter − η · gradient |
| **Epoch** | one pass over all the training examples |

## 4. How it works

### The network

The same 2-2-1 shape as in the inference guide: 2 inputs, 2 hidden neurons with ReLU, 1 output neuron with sigmoid.

```
 inputs          hidden layer           output
                 (ReLU)                 (sigmoid)

  x₁ ──┬──────▶  a₁ → h₁ ──┐
       │   ╲  ╱            ├──────▶  z → p
       │    ╳              │
  x₂ ──┴──────▶  a₂ → h₂ ──┘
```

| Symbol | Meaning |
|---|---|
| x₁, x₂ | inputs |
| y | the correct answer (0 or 1) |
| w₁₁, w₂₁, d₁ | weights from x₁, x₂ into h₁, and h₁'s bias |
| w₁₂, w₂₂, d₂ | weights from x₁, x₂ into h₂, and h₂'s bias |
| v₁, v₂, c | weights from h₁, h₂ into the output, and the output bias |
| a₁, a₂ | hidden values before ReLU |
| h₁, h₂ | hidden values after ReLU |
| z | output logit |
| p | predicted probability that the answer is 1 |
| L | loss |

The network has **9 parameters** to learn: w₁₁, w₂₁, d₁, w₁₂, w₂₂, d₂, v₁, v₂, c.

### Phase 1: forward pass

```
a₁ = w₁₁·x₁ + w₂₁·x₂ + d₁           h₁ = max(0, a₁)
a₂ = w₁₂·x₁ + w₂₂·x₂ + d₂           h₂ = max(0, a₂)
z  = v₁·h₁ + v₂·h₂ + c               p  = σ(z) = 1 / (1 + e^(−z))
```

**Keep every value** (a₁, a₂, h₁, h₂, z, p). The backward pass needs them.

### Phase 2: loss

For a yes/no answer we use **binary cross-entropy**:

```
L = −[ y·ln(p) + (1 − y)·ln(1 − p) ]
```

It's simpler than it looks:

- If y = 1:  L = −ln(p).  Small when p is close to 1, large when p is close to 0.
- If y = 0:  L = −ln(1 − p).  Small when p is close to 0, large when p is close to 1.

### Phase 3: backward pass

Think of the error as a **message that travels backward**. First, we find how the loss changes at the output. Then we pass that message back, step by step, to find how the loss depends on each hidden neuron and each weight.

**The whole backward pass is three moves:**

| Move | Formula | Question it answers |
|---|---|---|
| 1. Output gradient (error signal) | δz = p − y | How does the loss change when the output logit changes? (Its sign shows which direction reduces the loss.) |
| 2. Send the error back | δaⱼ = δz · vⱼ · ReLU′(aⱼ) | How does the loss change when each hidden neuron's pre-activation (aⱼ) changes? |
| 3. Weight gradients | ∂L/∂wᵢⱼ = δaⱼ · xᵢ  (and ∂L/∂vⱼ = δz · hⱼ) | How does the loss change when each weight changes? (Receiving neuron's error signal × the input to that weight.) |

ReLU′(a) is ReLU's derivative: 1 if a > 0, otherwise 0.

The rest of this phase shows where these formulas come from.

**The chain rule** says: to find how L depends on something deep inside the network, multiply the local derivatives along the path between them. For example, v₁ affects L only through z, and z affects L only through p:

```
∂L/∂v₁ = ∂L/∂p · ∂p/∂z · ∂z/∂v₁
```

We compute the derivatives **from the output backwards**, reusing each result for the next layer. That reuse is what makes backpropagation efficient.

**The local derivatives you need**

| Piece | Formula | Its derivative |
|---|---|---|
| Loss | L = −[y ln p + (1−y) ln(1−p)] | ∂L/∂p = −y/p + (1−y)/(1−p) |
| Sigmoid | p = σ(z) | ∂p/∂z = p·(1 − p) |
| Weighted sum | z = v₁h₁ + v₂h₂ + c | ∂z/∂v₁ = h₁,  ∂z/∂h₁ = v₁,  ∂z/∂c = 1 |
| ReLU | h = max(0, a) | ∂h/∂a = 1 if a > 0, otherwise 0 |
| Weighted sum | a₁ = w₁₁x₁ + w₂₁x₂ + d₁ | ∂a₁/∂w₁₁ = x₁,  ∂a₁/∂w₂₁ = x₂,  ∂a₁/∂d₁ = 1 |

(At a = 0, ReLU is not differentiable: its graph has a corner there. In this guide, we use the conventional subgradient value of 0, as most machine-learning libraries do.)

**A useful shortcut.** Multiply the first two rows together:

```
∂L/∂z = ∂L/∂p · ∂p/∂z
      = [ −y/p + (1−y)/(1−p) ] · p(1 − p)
      = −y(1 − p) + (1 − y)p
      = −y + yp + p − yp
      = p − y
```

So the error signal at the output is simply **prediction minus target**. We call it **δz = p − y**.

**All 9 gradients, from the output back**

```
Output layer:
  δz      = p − y
  ∂L/∂v₁  = δz · h₁
  ∂L/∂v₂  = δz · h₂
  ∂L/∂c   = δz

Into the hidden layer:
  ∂L/∂h₁  = δz · v₁
  ∂L/∂h₂  = δz · v₂

Through ReLU:
  δa₁     = ∂L/∂h₁ · (1 if a₁ > 0, else 0)
  δa₂     = ∂L/∂h₂ · (1 if a₂ > 0, else 0)

Hidden layer:
  ∂L/∂w₁₁ = δa₁ · x₁        ∂L/∂w₁₂ = δa₂ · x₁
  ∂L/∂w₂₁ = δa₁ · x₂        ∂L/∂w₂₂ = δa₂ · x₂
  ∂L/∂d₁  = δa₁             ∂L/∂d₂  = δa₂
```

The pattern: **weight gradient = error signal at the neuron the weight connects to × the input to that weight**.

Why this makes sense:

- A **large input** means a change in the weight has a larger effect on the neuron's value, so the gradient is larger.
- A **zero input** means the weight has no effect on the current example, so its gradient is zero.
- The **error signal's sign** decides whether increasing the weight would raise or lower the loss.

### Phase 4: update (gradient descent)

Backpropagation only **calculates** the gradients. Gradient descent **uses** them. For every parameter θ:

```
θ_new = θ_old − η · ∂L/∂θ
```

- If the gradient is **positive**, a sufficiently small increase in θ would increase the loss (the gradient describes the local slope), so gradient descent moves θ **down**. A finite step isn't guaranteed to lower the loss; see the note in Section 5.
- If the gradient is **negative**, gradient descent moves θ **up**.
- η (the learning rate) controls the step size.

## 5. One complete training step, by hand

> All calculations are shown step by step. Values are rounded to 3 decimals (4 where needed), and each step uses the rounded values shown, so these numbers are **illustrative approximations** you can follow by hand. The numbers in Appendix A (gradient check) and Appendix B (training runs) are computed at full precision and are reproducible outputs.

### Starting point

**Training example:** x = (x₁, x₂) = (1, 0), correct answer **y = 1** (XOR of 1 and 0 is 1).

**Learning rate:** η = 0.5

**Starting weights** (as if chosen at random):

| Parameter | Value | | Parameter | Value | | Parameter | Value |
|---|---|---|---|---|---|---|---|
| w₁₁ | 0.5 | | w₁₂ | 0.3 | | v₁ | 0.4 |
| w₂₁ | −0.4 | | w₂₂ | 0.4 | | v₂ | −0.6 |
| d₁ | 0.1 | | d₂ | 0.2 | | c | 0.1 |

### Phase 1: forward pass

**Hidden layer, mix**

```
a₁ = w₁₁·x₁ + w₂₁·x₂ + d₁
   = 0.5·1  + (−0.4)·0 + 0.1
   = 0.5    + 0        + 0.1
   = 0.6

a₂ = w₁₂·x₁ + w₂₂·x₂ + d₂
   = 0.3·1  + 0.4·0  + 0.2
   = 0.3    + 0      + 0.2
   = 0.5
```

**ReLU**

```
h₁ = max(0, 0.6) = 0.6
h₂ = max(0, 0.5) = 0.5
```

**Output, mix**

```
z = v₁·h₁ + v₂·h₂ + c
  = 0.4·0.6 + (−0.6)·0.5 + 0.1
  = 0.24    − 0.30       + 0.1
  = 0.04
```

**Sigmoid**

```
−z           = −0.04
e^(−z)       = e^(−0.04)      ≈ 0.961
1 + e^(−z)   = 1 + 0.961      = 1.961
p            = 1 / 1.961      ≈ 0.510
```

The network says 0.510, barely above a coin flip. The correct answer is 1.

### Phase 2: loss

y = 1, so:

```
L = −ln(p)
  = −ln(0.510)
  ≈ 0.673
```

### Phase 3: backward pass

**Output error signal**

```
δz = p − y
   = 0.510 − 1
   = −0.490
```

δz is negative: a small increase in z would lower the loss. That makes sense, because we want p (and so z) to go up. (Appendix A checks this value the long way, with the chain rule.)

**Output layer gradients**

```
∂L/∂v₁ = δz · h₁ = −0.490 · 0.6 = −0.294
∂L/∂v₂ = δz · h₂ = −0.490 · 0.5 = −0.245
∂L/∂c  = δz      = −0.490
```

**Send the error back to the hidden layer**

```
∂L/∂h₁ = δz · v₁ = −0.490 · 0.4    = −0.196
∂L/∂h₂ = δz · v₂ = −0.490 · (−0.6) =  0.294
```

**Through ReLU**

Both a₁ = 0.6 and a₂ = 0.5 are positive, so ReLU passes the error through unchanged (its derivative is 1):

```
δa₁ = ∂L/∂h₁ · 1 = −0.196 · 1 = −0.196
δa₂ = ∂L/∂h₂ · 1 =  0.294 · 1 =  0.294
```

**Hidden layer gradients**

```
∂L/∂w₁₁ = δa₁ · x₁ = −0.196 · 1 = −0.196
∂L/∂w₂₁ = δa₁ · x₂ = −0.196 · 0 =  0
∂L/∂d₁  = δa₁      = −0.196

∂L/∂w₁₂ = δa₂ · x₁ =  0.294 · 1 =  0.294
∂L/∂w₂₂ = δa₂ · x₂ =  0.294 · 0 =  0
∂L/∂d₂  = δa₂      =  0.294
```

w₂₁ and w₂₂ get **zero** gradient. Because x₂ = 0, these weights make no contribution to this forward pass, so their gradients are zero for this example. Other examples with x₂ ≠ 0 can give them nonzero gradients.

**All 9 gradients**

| Parameter | Gradient | Gradient descent will… |
|---|---|---|
| v₁ | −0.294 | increase v₁ |
| v₂ | −0.245 | increase v₂ |
| c | −0.490 | increase c |
| w₁₁ | −0.196 | increase w₁₁ |
| w₂₁ | 0 | leave w₂₁ unchanged |
| d₁ | −0.196 | increase d₁ |
| w₁₂ | 0.294 | decrease w₁₂ |
| w₂₂ | 0 | leave w₂₂ unchanged |
| d₂ | 0.294 | decrease d₂ |

Why does w₁₂ go **down**? h₂ feeds the output through v₂ = −0.6, a negative weight. A bigger h₂ makes z smaller, which is the wrong direction. So the weights that make h₂ bigger are lowered.

### Phase 4: update

θ_new = θ_old − 0.5 · gradient:

```
v₁  = 0.4  − 0.5·(−0.294) = 0.4  + 0.147  =  0.547
v₂  = −0.6 − 0.5·(−0.245) = −0.6 + 0.1225 = −0.4775
c   = 0.1  − 0.5·(−0.490) = 0.1  + 0.245  =  0.345

w₁₁ = 0.5  − 0.5·(−0.196) = 0.5  + 0.098  =  0.598
w₂₁ = −0.4 − 0.5·0        = −0.4 + 0      = −0.4
d₁  = 0.1  − 0.5·(−0.196) = 0.1  + 0.098  =  0.198

w₁₂ = 0.3  − 0.5·0.294    = 0.3  − 0.147  =  0.153
w₂₂ = 0.4  − 0.5·0        = 0.4  − 0      =  0.4
d₂  = 0.2  − 0.5·0.294    = 0.2  − 0.147  =  0.053
```

### Did it learn? Forward pass again

Same input (1, 0), new weights:

```
a₁ = 0.598·1 + (−0.4)·0 + 0.198 = 0.598 + 0 + 0.198 = 0.796
a₂ = 0.153·1 + 0.4·0    + 0.053 = 0.153 + 0 + 0.053 = 0.206

h₁ = max(0, 0.796) = 0.796
h₂ = max(0, 0.206) = 0.206

z  = 0.547·0.796 + (−0.4775)·0.206 + 0.345
   = 0.4354      − 0.0984          + 0.345
   = 0.682

−z           = −0.682
e^(−z)       = e^(−0.682)     ≈ 0.506
1 + e^(−z)   = 1 + 0.506      = 1.506
p            = 1 / 1.506      ≈ 0.664

L            = −ln(0.664)     ≈ 0.409
```

| | Before | After one step |
|---|---|---|
| Prediction p | 0.510 | 0.664 |
| Loss L | 0.673 | 0.409 |

The prediction moved toward the correct answer (1), and the loss dropped. **That's learning.**

With a suitable learning rate, an update should reduce the loss locally, as it did here. It's not guaranteed: with a learning rate that's too large, the loss can sometimes go up. That's why the loss is monitored during training.

This step updated the parameters using **one example's** gradients. To learn all of XOR, [Appendix B](#appendix-b-training-on-all-of-xor) instead **averages** the gradients from all four examples before making each update, and repeats this many times.

## 6. Common mistakes

- **Updating weights during the backward pass.** Compute **all** gradients with the **old** weights first, then update. In our example, ∂L/∂h₁ used v₁ = 0.4, not the updated 0.547.
- **Forgetting to keep the forward values.** The backward pass needs a₁, a₂, h₁, h₂ and p from the forward pass.
- **Wrong sign in the update.** It's θ − η·gradient (downhill), not plus.
- **"Backpropagation updates the weights."** It only **calculates the gradients**. Gradient descent (or another optimizer) **uses** them to change the weights.
- **"A zero gradient means the weight is perfect."** No. A zero gradient means a small change in that parameter wouldn't change the loss at the current point. In our example it was because the input was 0; it can also happen when ReLU blocks the error (Appendix B), or for other reasons. It doesn't mean the parameter is optimal.

## 7. Remember this

- Training repeats **forward → loss → backward → update**.
- Backpropagation uses the **chain rule**, from the output back, to calculate every gradient.
- With sigmoid + cross-entropy, the output error is simply **δz = p − y**.
- Weight gradient = **error signal at the neuron the weight connects to × the input to that weight**.
- Gradient descent: **θ ← θ − η · ∂L/∂θ**.

### Quick check

1. The network predicts p = 0.8 for an example with y = 0. What is δz? Should z go up or down?
2. In our example, why did w₂₁ not change?
3. If a₁ had been −0.2, what would ∂L/∂w₁₁ have been?
4. A gradient is +0.4 and η = 0.1. How much does the parameter change, and in which direction?

Try answering before reading on.

---

### Check your answers

1. δz = p − y = 0.8 − 0 = **0.8**. It's positive, so a small increase in z would increase the loss: z should go **down**, making p smaller, closer to 0.
2. ∂L/∂w₂₁ = δa₁ · x₂ = −0.196 · 0 = **0**. Its input x₂ was 0, so a small change in w₂₁ wouldn't affect this answer.
3. ReLU's derivative is 0 when a₁ = −0.2. So, whatever the value of ∂L/∂h₁, δa₁ = (∂L/∂h₁) · 0 = 0, and ∂L/∂w₁₁ = δa₁ · x₁ = 0 · 1 = **0**.
4. Change = −η · gradient = −0.1 · 0.4 = **−0.04**: the parameter **decreases** by 0.04.

## 8. Going further

- **[MLP Inference: XOR](mlp_xor_inference.md)**: the forward pass that backpropagation runs in reverse.
- **[Attention Blocks](attention_blocks.md)**, **[LLM Inference](llm_inference.md)** and **[ViT](vit.md)**: every weight in those models is learned with this same process; the networks are just much bigger.
- **Real code** usually lets a library calculate the gradients automatically. See [Appendix C](#appendix-c-formula-sheet-and-automatic-differentiation).

### Learn more

- 📖 [How the backpropagation algorithm works](https://neuralnetworksanddeeplearning.com/chap2.html) (Michael Nielsen, chapter 2): the full derivation, step by step
- 📖 [micrograd README](https://github.com/karpathy/micrograd) (Andrej Karpathy): a tiny automatic-differentiation engine and neural network library you can read in one sitting

---

## Appendix A: Checking the gradients

*Optional.*

### Check δz with the full chain rule

```
∂L/∂p = −y/p + (1−y)/(1−p)  = −1/0.510 + 0      ≈ −1.961
∂p/∂z = p·(1 − p)           = 0.510 · 0.490     ≈ 0.2499
∂L/∂z = ∂L/∂p · ∂p/∂z       = −1.961 · 0.2499   ≈ −0.490   ✅ same as p − y
```

### Numerical gradient check

You can check a gradient numerically: nudge one parameter slightly and estimate how the loss changes per unit change in that parameter. This gives an estimate, not the exact derivative. Increase v₁ by 0.001 (from 0.4 to 0.401), keeping all other weights at their starting values.

Because the change in loss is tiny, the losses must be computed with **many decimals**, or rounding will swamp the result:

```
Before:  z = 0.04
         p = σ(0.04)      = 0.5099987
         L = −ln(p)       = 0.6733472

After:   z = 0.401·0.6 + (−0.6)·0.5 + 0.1 = 0.2406 − 0.30 + 0.1 = 0.0406
         p = σ(0.0406)    = 0.5101486
         L = −ln(p)       = 0.6730532

Change in L:    0.6730532 − 0.6733472 = −0.0002940
Slope:         −0.0002940 / 0.001     = −0.2940
```

The backward pass gave ∂L/∂v₁ = −0.294. ✅ They match. Small differences can appear because of finite precision and because the nudge isn't infinitely small. This test is called a **gradient check**, and it's how people test backpropagation code.

## Appendix B: Training on all of XOR

*Optional. Topics beyond a single update.*

### The training loop

The worked example updated after **one** example. Here, each epoch runs all four examples, **averages** their gradients, and updates once:

```
repeat for each epoch:
    forward pass and loss for all four examples
    backward pass: gradients for each example, then average them
    update every parameter once with the averaged gradients
```

### A reproducible run

This short Python script (it needs NumPy) uses the starting weights from Section 5 and η = 0.5. It prints the average loss and the four predictions at selected epochs:

```python
import numpy as np

# The four XOR examples and their correct answers
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
Y = np.array([0, 1, 1, 0], dtype=float)

def train(w21, w22, lr=0.5, epochs=5000):
    # Starting weights (same as the worked example)
    W1 = np.array([[0.5, 0.3],      # row 1: w11, w12  (from x1)
                   [w21, w22]])     # row 2: w21, w22  (from x2)
    b1 = np.array([0.1, 0.2])       # d1, d2
    W2 = np.array([0.4, -0.6])      # v1, v2
    b2 = 0.1                        # c

    for epoch in range(epochs + 1):
        # Forward pass, all four examples at once
        A = X @ W1 + b1
        H = np.maximum(A, 0)
        Z = H @ W2 + b2
        P = 1 / (1 + np.exp(-Z))
        loss = -np.mean(Y * np.log(P) + (1 - Y) * np.log(1 - P))
        if epoch in (0, 100, 500, 5000):
            print(epoch, round(loss, 3), P.round(3))

        # Backward pass: gradients averaged over the four examples
        dZ = (P - Y) / 4
        gW2 = H.T @ dZ
        gb2 = dZ.sum()
        dA = np.outer(dZ, W2) * (A > 0)
        gW1 = X.T @ dA
        gb1 = dA.sum(axis=0)

        # Update (gradient descent)
        W1 -= lr * gW1; b1 -= lr * gb1
        W2 -= lr * gW2; b2 -= lr * gb2

print("Run 1: w21 = -0.4, w22 = 0.4")
train(-0.4, 0.4)
print("Run 2: w21 = 0.2, w22 = -0.1")
train(0.2, -0.1)
```

**Run 1** (the starting weights from Section 5):

| Epoch | Average loss | p for (0,0) | p for (0,1) | p for (1,0) | p for (1,1) |
|---|---|---|---|---|---|
| 0 | 0.684 | 0.505 | 0.435 | 0.510 | 0.411 |
| 100 | 0.230 | 0.203 | 0.615 | 0.945 | 0.140 |
| 500 | 0.012 | 0.007 | 0.969 | 0.996 | 0.005 |
| 5000 | 0.001 | 0.000 | 0.998 | 1.000 | 0.000 |

Predictions are rounded to three decimals; 1.000 and 0.000 are not exact probabilities (sigmoid never reaches exactly 0 or 1).

The targets are 0, 1, 1, 0, and after training all four answers are right. The final weights differ from the hand-picked ones in the inference guide, but they solve XOR just as well: there are many good solutions.

### Starting weights matter

**Run 2** changes only w₂₁ and w₂₂, to 0.2 and −0.1. In this run, learning stalls: the average loss stays at 0.693 from epoch 100 to 5000, and training converges to a constant prediction of 0.5 for every example (by epoch 500, all four print as 0.5 at 3 decimals). A loss of 0.693 = ln 2 is exactly what predicting 0.5 for everything gives.

This is a particular outcome of this initialization and training setup (the starting weights, the learning rate, averaging the gradients), not typical training behavior. The general lesson is that **some starting points lead to slow learning or stuck neurons**, especially in a network this small. Common remedies: different random starting weights, more hidden neurons, or a different learning rate.

### ReLU can block the error

If a hidden value before ReLU is **negative**, ReLU's derivative is 0, and **no error passes through**. For example, suppose a₂ were −0.3 during a forward pass. Regardless of the incoming gradient:

```
δa₂ = ∂L/∂h₂ · 0 = 0
```

A negative pre-activation makes ReLU's derivative zero, which blocks the gradient for that step. Then w₁₂, w₂₂ and d₂ would all get zero gradient in that step. If a neuron's pre-activation aⱼ stays negative for **every** training example, ReLU blocks the gradient for that neuron, so its incoming weights and bias stop learning through this path. This is called a **dead ReLU**.

### Choosing the learning rate

- **Too small:** the loss goes down very slowly.
- **Too large:** the steps overshoot, and the loss can jump around or grow.
- Good values depend on the problem; trying a few is normal.

## Appendix C: Formula sheet and automatic differentiation

### Matrix form

**Shapes:** x, a, h, δa and δh are row vectors (1 × 2). W₁ is 2 × 2 (row i = weights from xᵢ, column j = weights into hⱼ), b₁ is 1 × 2, W₂ is 2 × 1, and b₂ is a scalar. z, p, y, L and δz are **scalars** (single numbers). ⊙ means multiply entry by entry.

```
Forward:   a  = x W₁ + b₁        (1×2)(2×2) + (1×2)  → 1 × 2
           h  = max(0, a)                            → 1 × 2
           z  = h W₂ + b₂        (1×2)(2×1) + scalar → scalar
           p  = σ(z)                                 → scalar
Loss:      L  = −[ y·ln(p) + (1 − y)·ln(1 − p) ]     → scalar

Backward:  δz      = p − y                   scalar
           ∂L/∂W₂  = hᵀ · δz                 (2×1) · scalar     → 2 × 1    (∂L/∂vⱼ = δz · hⱼ)
           ∂L/∂b₂  = δz                      scalar                        (∂L/∂c  = δz)
           δh      = δz · W₂ᵀ                scalar · (1×2)     → 1 × 2    (∂L/∂hⱼ = δz · vⱼ)
           δa      = δh ⊙ [a > 0]            (1×2) ⊙ (1×2)      → 1 × 2    (×1 if aⱼ > 0, else ×0)
           ∂L/∂W₁  = xᵀ · δa                 (2×1)(1×2)         → 2 × 2    (∂L/∂wᵢⱼ = xᵢ · δaⱼ)
           ∂L/∂b₁  = δa                      1 × 2                         (∂L/∂dⱼ = δaⱼ)

Update:    θ ← θ − η · ∂L/∂θ                 (each gradient has the same shape as its parameter)
```

xᵀ · δa is an **outer product**: a column (2 × 1) times a row (1 × 2) gives a 2 × 2 matrix, one entry per weight in W₁.

### Automatic differentiation

For large models, gradients are rarely derived by hand. Libraries such as PyTorch automate the gradient calculation: they record each operation during the forward pass, then run the chain rule backwards. This is called **automatic differentiation**. It's the same math you did in Section 5, applied to millions or billions of parameters.
