# Activation Functions: A Practical Guide

MLP inference applies an activation function after almost every linear layer, but they're easy to treat as magic. This guide explains what each well-known activation does, why it exists, where it's used, and how to compute it correctly. Code examples use NumPy.

```python
import numpy as np
```

---

## Contents

1. [Why activation functions exist](#1-why-activation-functions-exist)
2. [The families at a glance](#2-the-families-at-a-glance)
3. [Classic functions](#3-classic-functions): Identity, Step, Sigmoid, Tanh
4. [The ReLU family](#4-the-relu-family): ReLU, Leaky ReLU, PReLU, RReLU, ReLU6, ELU, SELU, CELU
5. [Smooth modern functions](#5-smooth-modern-functions): Softplus, GELU, Swish / SiLU, Mish
6. [Hard (piecewise) approximations](#6-hard-piecewise-approximations): Hard Sigmoid, Hard Tanh, Hard Swish, Softsign
7. [Output-layer functions](#7-output-layer-functions): Softmax, Log-Softmax, Sparsemax
8. [Gated and learned activations](#8-gated-and-learned-activations): GLU, SwiGLU, GeGLU, Maxout
9. [Putting it together: an MLP forward pass](#9-putting-it-together-an-mlp-forward-pass)
10. [Comparison table](#10-comparison-table)
11. [Choosing an activation](#11-choosing-an-activation)

---

## 1. Why activation functions exist

A layer in an MLP computes a linear transformation:

```
z = W·x + b
```

If you stack layers with nothing between them:

```
W2·(W1·x + b1) + b2  =  (W2·W1)·x + (W2·b1 + b2)  =  W'·x + b'
```

The whole stack collapses to **a single linear layer**. No matter how deep, the network could only separate data with straight lines (hyperplanes).

An activation function `f` is applied after each linear step:

```
h = f(W·x + b)
```

Because `f` is **non-linear**, the layers no longer collapse, and the network can model curved, complex relationships. This is what makes depth useful.

Activations play two different roles in an MLP:

| Where | Purpose | Typical choices |
|---|---|---|
| Hidden layers | Add non-linearity so depth matters | ReLU, GELU, SiLU, Leaky ReLU |
| Output layer | Turn raw scores into a usable answer | Sigmoid, Softmax, or nothing (regression) |

### Properties worth knowing

When comparing activations, these properties come up repeatedly:

- **Range**: bounded (e.g. sigmoid, 0–1) or unbounded (e.g. ReLU, 0–∞).
- **Zero-centered**: whether outputs average around 0. Zero-centered outputs tend to make optimization easier.
- **Monotonic**: whether the function always increases. Modern smooth functions (GELU, Swish, Mish) are *not* monotonic — they dip slightly below 0 for small negative inputs.
- **Smoothness**: whether the derivative is continuous. ReLU has a kink at 0; GELU and Swish are smooth everywhere.
- **Saturation**: whether the gradient approaches 0 for large inputs. Saturating functions (sigmoid, tanh) cause **vanishing gradients** in deep networks.
- **Cost**: element-wise comparisons are cheap; exponentials and `erf` are more expensive, which matters on mobile and embedded hardware.

> **Inference vs. training.** Many of these trade-offs (vanishing gradients, dead neurons) affect *training*. At inference time the weights are fixed, and you simply apply whatever function the model was trained with. Using a different activation at inference than at training time will break the model.

---

## 2. The families at a glance

```
Classic ─────────── Identity, Step, Sigmoid, Tanh
                       (historical; sigmoid/tanh still used in gates & outputs)

ReLU family ─────── ReLU → Leaky ReLU, PReLU, RReLU, ReLU6
                       → ELU, SELU, CELU   (smooth negative side)

Smooth modern ───── Softplus, GELU, Swish/SiLU, Mish
                       (default in transformers and modern CNNs)

Hard approximations ─ Hard Sigmoid, Hard Tanh, Hard Swish, Softsign
                       (cheap versions for mobile / quantized models)

Output ──────────── Softmax, Log-Softmax, Sparsemax

Gated / learned ─── GLU, SwiGLU, GeGLU, Maxout
                       (used in modern LLM feed-forward blocks)
```

---

## 3. Classic functions

### 3.1 Identity (Linear)

```
f(z) = z
```

No transformation at all. Used on the **output layer of regression models**, where the network should be able to produce any real number. Using it in hidden layers makes them collapse (see §1).

```python
def identity(z):
    return z
```

### 3.2 Binary Step (Heaviside)

```
f(z) = 1 if z ≥ 0 else 0
```

The activation of the original perceptron (1958). Its derivative is 0 everywhere (and undefined at 0), so it **cannot be trained with gradient descent**. Today it's of historical and teaching interest only.

```python
def step(z):
    return (z >= 0).astype(float)
```

### 3.3 Sigmoid (Logistic)

```
sigmoid(z) = 1 / (1 + e^(-z))
```

Squashes any real number into **(0, 1)**.

```
  1.0 |              ________
      |           /
  0.5 |---------•
      |       /
  0.0 |______/
      +------------------------
           -6    0    +6
```

| z | -∞ | -2 | 0 | 2 | +∞ |
|---|---|---|---|---|---|
| sigmoid(z) | → 0 | 0.119 | 0.5 | 0.881 | → 1 |

**Where it's used**

- **Binary classification output**: one neuron gives P(positive class). Predict "yes" if ≥ 0.5 (equivalently, if `z ≥ 0`).
- **Multi-label output**: several independent yes/no outputs; probabilities don't need to sum to 1.
- **Gates** in LSTMs, GRUs, and Swish/GLU, where a 0–1 "how much to let through" value is needed.

**Why not in hidden layers**: the slope is at most 0.25 and near 0 for large |z|, so gradients **vanish** through deep stacks. Outputs are also always positive (not zero-centered).

**Numerically stable version** (avoids overflow of `e^(-z)` for large negative z):

```python
def sigmoid(z):
    z = np.asarray(z, dtype=float)
    out = np.empty_like(z)
    pos = z >= 0
    out[pos] = 1 / (1 + np.exp(-z[pos]))
    ez = np.exp(z[~pos])              # safe: z is negative here
    out[~pos] = ez / (1 + ez)
    return out
```

### 3.4 Tanh (Hyperbolic Tangent)

```
tanh(z) = (e^z - e^(-z)) / (e^z + e^(-z))  =  2·sigmoid(2z) - 1
```

A rescaled sigmoid with range **(-1, 1)**.

| z | -∞ | -2 | 0 | 2 | +∞ |
|---|---|---|---|---|---|
| tanh(z) | → -1 | -0.964 | 0 | 0.964 | → 1 |

- **Zero-centered**, which made it preferable to sigmoid for hidden layers before ReLU.
- Steeper than sigmoid (max slope 1 vs 0.25) but still **saturates**, so gradients still vanish in deep networks.
- Still common in **RNN/LSTM cell states**, and when outputs must lie in [-1, 1] (e.g. image generators producing normalized pixels, or reinforcement-learning action outputs).

```python
def tanh(z):
    return np.tanh(z)
```

---

## 4. The ReLU family

### 4.1 ReLU (Rectified Linear Unit)

```
ReLU(z) = max(0, z)
```

```
  output
    |        /
    |       /
    |      /
----+-----/-------- input
    |    0
```

```
z       = [-2.0, -0.5, 0.0, 1.5, 3.0]
ReLU(z) = [ 0.0,  0.0, 0.0, 1.5, 3.0]
```

**Why it became the default (≈2011 onward)**

- **Cheap**: one comparison, no exponentials.
- **Non-saturating** for positive inputs: slope is exactly 1, so gradients don't shrink through many layers.
- **Sparse**: many neurons output exactly 0, which is efficient.
- Despite its simplicity, combinations of ReLUs approximate any function piecewise-linearly.

**Weakness — "dying ReLU"**: if a neuron's input becomes negative for every example, it outputs 0, its gradient is 0, and it never recovers during training. Most of the variants below exist to fix this.

```python
def relu(z):
    return np.maximum(0, z)
```

### 4.2 Leaky ReLU

```
LeakyReLU(z) = z        if z > 0
             = α·z      otherwise      (α typically 0.01)
```

A small slope on the negative side keeps gradients alive, so neurons can't fully die. Common in GANs (often with α = 0.2).

```python
def leaky_relu(z, alpha=0.01):
    return np.where(z > 0, z, alpha * z)
```

### 4.3 PReLU (Parametric ReLU)

Same formula as Leaky ReLU, but **α is learned** during training (one per channel, or one shared). At inference, α is just a stored weight.

```python
def prelu(z, alpha):        # alpha loaded from the trained model
    return np.where(z > 0, z, alpha * z)
```

### 4.4 RReLU (Randomized Leaky ReLU)

During **training**, α is sampled randomly from a range (e.g. [1/8, 1/3]) for each element, acting as a regularizer. During **inference**, α is fixed to the average of that range:

```python
def rrelu_inference(z, lower=1/8, upper=1/3):
    alpha = (lower + upper) / 2
    return np.where(z > 0, z, alpha * z)
```

This is a good example of an activation that behaves *differently* at training and inference time.

### 4.5 ReLU6

```
ReLU6(z) = min(max(0, z), 6)
```

ReLU capped at 6. Popularized by **MobileNet**: the bounded range works well with low-precision (8-bit) quantized inference on mobile hardware.

```python
def relu6(z):
    return np.clip(z, 0, 6)
```

### 4.6 ELU (Exponential Linear Unit)

```
ELU(z) = z                  if z > 0
       = α·(e^z - 1)        otherwise      (α typically 1.0)
```

- Smooth negative side that saturates at **-α**.
- Negative outputs push mean activations toward 0, which can speed up learning.
- Costs an exponential for negative inputs.

```python
def elu(z, alpha=1.0):
    return np.where(z > 0, z, alpha * np.expm1(np.minimum(z, 0)))
```

(`expm1` computes `e^z - 1` accurately for small z; `np.minimum(z, 0)` avoids overflow in the unused branch.)

### 4.7 SELU (Scaled ELU)

```
SELU(z) = λ · ELU(z, α)

λ ≈ 1.0507
α ≈ 1.67326
```

These specific constants make the network **self-normalizing**: activations tend to stay at mean 0, variance 1 across layers, without batch normalization. This only holds under specific conditions (LeCun-normal weight initialization, plain fully-connected layers, "alpha dropout" instead of regular dropout). Designed specifically for **deep MLPs**.

```python
def selu(z):
    lam, alpha = 1.0507009873554805, 1.6732632423543772
    return lam * np.where(z > 0, z, alpha * np.expm1(np.minimum(z, 0)))
```

### 4.8 CELU (Continuously Differentiable ELU)

```
CELU(z) = max(0, z) + min(0, α·(e^(z/α) - 1))
```

A reformulation of ELU whose derivative is continuous for every α (ELU's is only continuous when α = 1). Identical to ELU when α = 1.

```python
def celu(z, alpha=1.0):
    return np.maximum(0, z) + np.minimum(0, alpha * np.expm1(np.minimum(z, 0) / alpha))
```

---

## 5. Smooth modern functions

These dominate **transformers** and recent CNNs. They look like ReLU for large inputs but are smooth around 0, and they let small negative values through.

```
        ReLU                GELU / Swish / Mish
         |   /                    |   /
         |  /                     |  /
  _______|_/          ____     ___|_/
         0                \__/    0
                      (slight dip below zero)
```

### 5.1 Softplus

```
Softplus(z) = ln(1 + e^z)
```

A smooth version of ReLU: always positive, approaches `z` for large z and 0 for very negative z. Its derivative is exactly the sigmoid. Commonly used when a network must output a **strictly positive value** (e.g. a variance or standard deviation), and as a building block for Mish.

```python
def softplus(z):
    return np.logaddexp(0, z)     # = ln(e^0 + e^z), numerically stable
```

### 5.2 GELU (Gaussian Error Linear Unit)

```
GELU(z) = z · Φ(z)
```

where `Φ` is the cumulative distribution function of the standard normal distribution. Intuitively, the input is scaled by "the probability that a standard normal value is below it" — large positive inputs pass almost fully, large negative ones are nearly zeroed, and values near 0 are smoothly weighted.

**The default activation in BERT, the GPT family, and Vision Transformers.**

Exact form uses the error function:

```
GELU(z) = 0.5 · z · (1 + erf(z / √2))
```

A widely used fast approximation:

```
GELU(z) ≈ 0.5 · z · (1 + tanh( √(2/π) · (z + 0.044715 · z³) ))
```

> Use the same variant the model was trained with — exact and tanh-approximated GELU give slightly different numbers.

```python
from math import sqrt, pi
from scipy.special import erf

def gelu(z):
    return 0.5 * z * (1 + erf(z / sqrt(2)))

def gelu_tanh(z):
    return 0.5 * z * (1 + np.tanh(sqrt(2 / pi) * (z + 0.044715 * z**3)))
```

### 5.3 Swish / SiLU (Sigmoid Linear Unit)

```
Swish(z) = z · sigmoid(β·z)
SiLU(z)  = z · sigmoid(z)          (Swish with β = 1)
```

Discovered via automated search over candidate functions. Very similar in shape to GELU and often interchangeable. Used in **EfficientNet**, **YOLO** variants, and inside **SwiGLU** in LLMs such as LLaMA.

- β → ∞ makes it approach ReLU; β = 0 makes it the linear function z/2.
- Minimum value ≈ -0.278 at z ≈ -1.28.

```python
def silu(z):
    return z * sigmoid(z)

def swish(z, beta=1.0):
    return z * sigmoid(beta * z)
```

### 5.4 Mish

```
Mish(z) = z · tanh(Softplus(z)) = z · tanh(ln(1 + e^z))
```

Very similar to Swish, slightly smoother, a bit more expensive. Used in some YOLOv4 configurations.

```python
def mish(z):
    return z * np.tanh(softplus(z))
```

---

## 6. Hard (piecewise) approximations

Exponentials are relatively expensive on CPUs, mobile chips, and integer-only (quantized) hardware. These replace curves with straight-line segments that can be computed with just additions, multiplications and clipping.

### 6.1 Hard Sigmoid

```
HardSigmoid(z) = ReLU6(z + 3) / 6  =  clip(z/6 + 0.5, 0, 1)
```

(This is the PyTorch definition; some libraries use slope 0.2 instead of 1/6.)

```python
def hard_sigmoid(z):
    return np.clip(z / 6 + 0.5, 0, 1)
```

### 6.2 Hard Tanh

```
HardTanh(z) = clip(z, -1, 1)
```

```python
def hard_tanh(z):
    return np.clip(z, -1, 1)
```

### 6.3 Hard Swish

```
HardSwish(z) = z · ReLU6(z + 3) / 6  =  z · HardSigmoid(z)
```

Introduced in **MobileNetV3** as a cheap stand-in for Swish with nearly the same accuracy.

```python
def hard_swish(z):
    return z * hard_sigmoid(z)
```

### 6.4 Softsign

```
Softsign(z) = z / (1 + |z|)
```

A tanh-like curve in (-1, 1) without exponentials. It approaches its limits polynomially (more slowly than tanh), so it saturates more gently. Rarely used today but appears in some older models.

```python
def softsign(z):
    return z / (1 + np.abs(z))
```

---

## 7. Output-layer functions

### 7.1 Softmax

Turns a **vector** of raw scores (*logits*) into a probability distribution:

```
softmax(z)_i = e^(z_i) / Σ_j e^(z_j)
```

Unlike the functions above, softmax is **not element-wise**: each output depends on every input.

- Every output is in (0, 1), and outputs **sum to 1**.
- Order is preserved: the largest logit gets the largest probability.
- Exponentiation exaggerates differences, so the top score tends to dominate.

**Worked example** — logits for (cat, dog, bird):

```
z       = [2.0,   1.0,   0.1]
e^z     = [7.389, 2.718, 1.105]     sum = 11.212
softmax = [0.659, 0.242, 0.099]     sum = 1.000
```

Prediction: **cat** (65.9%).

**Use for**: multi-class, single-label classification (one correct class out of N), including next-token prediction in language models and attention weights in transformers.

**Numerical stability (important):** `e^1000` overflows. Subtract the max logit first; this doesn't change the result because `e^(-max)` cancels top and bottom.

```python
def softmax(z, axis=-1):
    z = z - np.max(z, axis=axis, keepdims=True)
    e = np.exp(z)
    return e / np.sum(e, axis=axis, keepdims=True)
```

**Temperature.** Dividing logits by a temperature `T` before softmax controls confidence: `T < 1` sharpens the distribution, `T > 1` flattens it. This is how LLM sampling "temperature" works.

```python
def softmax_t(z, T=1.0):
    return softmax(z / T)
```

**Inference shortcut.** If you only need the predicted class, skip softmax: `argmax(softmax(z)) == argmax(z)`.

```python
pred = np.argmax(logits, axis=-1)
```

**Relationship to sigmoid.** Softmax over two logits `[z, 0]` gives `sigmoid(z)` for the first class — sigmoid is the two-class special case.

### 7.2 Log-Softmax

```
log_softmax(z)_i = z_i - log Σ_j e^(z_j)
```

Computing `log(softmax(z))` in two steps can produce `log(0) = -∞` when a probability underflows. Log-softmax computes it directly and stably. Used with the negative log-likelihood loss and wherever log-probabilities are summed (e.g. scoring sequences in beam search).

```python
def log_softmax(z, axis=-1):
    z = z - np.max(z, axis=axis, keepdims=True)
    return z - np.log(np.sum(np.exp(z), axis=axis, keepdims=True))
```

### 7.3 Sparsemax

Like softmax, it outputs a probability distribution, but it can assign **exactly zero** to low-scoring classes. It works by projecting the logits onto the probability simplex. Useful when you want sparse, more interpretable attention or class selection.

```python
def sparsemax(z):
    z_sorted = np.sort(z)[::-1]
    cumsum = np.cumsum(z_sorted)
    k = np.arange(1, len(z) + 1)
    support = 1 + k * z_sorted > cumsum
    k_max = k[support][-1]
    tau = (cumsum[k_max - 1] - 1) / k_max
    return np.maximum(z - tau, 0)
```

```
z         = [2.0,  1.0,  0.1]
softmax   = [0.659, 0.242, 0.099]
sparsemax = [1.0,   0.0,   0.0]    # weaker classes cut to exactly 0
```

---

## 8. Gated and learned activations

### 8.1 GLU (Gated Linear Unit)

Instead of applying a fixed function to one projection, GLU computes **two** projections of the input and uses one to gate the other:

```
GLU(x) = (x·W + b) ⊙ sigmoid(x·V + c)
```

`⊙` is element-wise multiplication. The sigmoid branch decides, per element, how much of the linear branch to let through.

### 8.2 GLU variants: SwiGLU, GeGLU, ReGLU

Swap the sigmoid gate for another activation:

| Variant | Gate function | Used in |
|---|---|---|
| GLU | sigmoid | Original gated CNNs |
| ReGLU | ReLU | Research variants |
| GeGLU | GELU | Some T5 variants |
| **SwiGLU** | Swish / SiLU | **LLaMA, PaLM, Mistral**, many modern LLMs |

A SwiGLU feed-forward block in a transformer:

```python
def swiglu_ffn(x, W_gate, W_up, W_down):
    gate = silu(x @ W_gate)       # activation branch
    up   = x @ W_up               # linear branch
    return (gate * up) @ W_down   # gate, then project back
```

Gated blocks have three weight matrices instead of two, so the hidden size is usually reduced (often to about 2/3) to keep the parameter count comparable.

### 8.3 Maxout

```
Maxout(x) = max_k ( x·W_k + b_k )
```

Each unit computes `k` linear functions and outputs the largest. It *learns* its own piecewise-linear activation shape (ReLU is the special case of `max(x·W, 0)`). Powerful but multiplies the parameter count by `k`, so it's rarely used today.

```python
def maxout(x, Ws, bs):     # Ws: list of k weight matrices
    return np.max([x @ W + b for W, b in zip(Ws, bs)], axis=0)
```

---

## 9. Putting it together: an MLP forward pass

A 3-layer MLP classifying into 10 classes (e.g. MNIST digits), with the hidden activation swappable:

```python
ACTIVATIONS = {
    "relu": relu,
    "leaky_relu": leaky_relu,
    "elu": elu,
    "selu": selu,
    "gelu": gelu,
    "silu": silu,
    "mish": mish,
    "tanh": tanh,
}

def mlp_forward(x, params, act="relu"):
    f = ACTIVATIONS[act]
    W1, b1, W2, b2, W3, b3 = params
    h1 = f(x @ W1 + b1)          # hidden layer 1: linear → activation
    h2 = f(h1 @ W2 + b2)         # hidden layer 2: linear → activation
    logits = h2 @ W3 + b3        # output layer: linear only
    return softmax(logits)       # probabilities

# x: [batch, 784], W1: [784, 128], W2: [128, 64], W3: [64, 10]
```

```
input x
   │
   ▼
[Linear W1,b1] → hidden activation    (non-linearity)
   │
   ▼
[Linear W2,b2] → hidden activation    (non-linearity)
   │
   ▼
[Linear W3,b3] → logits               (NO hidden activation here)
   │
   ▼
Output activation → answer            (softmax / sigmoid / identity)
```

The **final linear layer has no hidden activation**. Logits must be allowed to be negative; clamping them with ReLU would destroy information the output function needs.

### Matching output activation to task

| Task | Output neurons | Output activation | Paired loss (training) |
|---|---|---|---|
| Regression | 1 per target | Identity (or Softplus if must be > 0) | MSE / MAE |
| Binary classification | 1 | Sigmoid | Binary cross-entropy |
| Multi-label classification | N | Sigmoid on each | Binary cross-entropy per label |
| Multi-class classification | N | Softmax | Cross-entropy |
| Bounded output in [-1, 1] | any | Tanh | Task-dependent |

> In practice, frameworks often fuse the output activation into the loss for stability (e.g. "cross-entropy with logits"). The model then outputs raw logits, and you apply sigmoid/softmax yourself at inference time if you need probabilities.

---

## 10. Comparison table

| Function | Formula | Range | Zero-centered | Smooth | Saturates | Cost | Typical use |
|---|---|---|---|---|---|---|---|
| Identity | `z` | (-∞, ∞) | yes | yes | no | lowest | Regression output |
| Step | `1 if z≥0 else 0` | {0, 1} | no | no | — | lowest | Historical |
| Sigmoid | `1/(1+e^-z)` | (0, 1) | no | yes | both sides | exp | Binary output, gates |
| Tanh | `tanh(z)` | (-1, 1) | yes | yes | both sides | exp | RNNs, bounded output |
| ReLU | `max(0,z)` | [0, ∞) | no | no | negative side | lowest | Default hidden layer |
| Leaky ReLU | `z or αz` | (-∞, ∞) | ~ | no | no | lowest | Avoid dead neurons, GANs |
| PReLU | `z or αz`, α learned | (-∞, ∞) | ~ | no | no | lowest | CNNs |
| ReLU6 | `clip(z,0,6)` | [0, 6] | no | no | both sides | lowest | Mobile / quantized |
| ELU | `z or α(e^z−1)` | (−α, ∞) | ~ | yes (α=1) | negative side | exp | Deep nets |
| SELU | `λ·ELU(z)` | (−λα, ∞) | yes* | no | negative side | exp | Self-normalizing MLPs |
| Softplus | `ln(1+e^z)` | (0, ∞) | no | yes | negative side | exp + log | Positive outputs |
| GELU | `z·Φ(z)` | ≈(−0.17, ∞) | ~ | yes | negative side | erf / tanh | Transformers |
| SiLU / Swish | `z·σ(z)` | ≈(−0.28, ∞) | ~ | yes | negative side | exp | CNNs, LLMs (SwiGLU) |
| Mish | `z·tanh(softplus z)` | ≈(−0.31, ∞) | ~ | yes | negative side | highest | Some detection models |
| Hard Sigmoid | `clip(z/6+0.5,0,1)` | [0, 1] | no | no | both sides | lowest | Mobile |
| Hard Swish | `z·HardSigmoid(z)` | ≈[−0.375, ∞) | ~ | no | negative side | lowest | MobileNetV3 |
| Softsign | `z/(1+|z|)` | (-1, 1) | yes | yes | both (slowly) | low | Rare |
| Softmax | `e^zi / Σe^zj` | (0, 1), sums to 1 | — | yes | — | exp + sum | Multi-class output |
| Sparsemax | simplex projection | [0, 1], sums to 1 | — | no | — | sort | Sparse attention / output |

`~` = approximately zero-centered (outputs can be negative, but the mean isn't guaranteed to be 0).
`*` = under the self-normalizing conditions described in §4.7.

---

## 11. Choosing an activation

**Hidden layers**

- **Default, simple MLP or CNN:** ReLU. Fast and works well in most cases.
- **Seeing dead neurons / training stalls:** Leaky ReLU or PReLU.
- **Transformer or modern architecture:** GELU, or SiLU; follow the reference architecture.
- **LLM feed-forward block:** SwiGLU.
- **Deep plain MLP without batch norm:** SELU (with LeCun-normal init and alpha dropout).
- **Mobile, edge, or 8-bit quantized inference:** ReLU6, Hard Swish, Hard Sigmoid.
- **Recurrent networks (LSTM/GRU):** tanh for state, sigmoid for gates.
- **Avoid in deep hidden stacks:** sigmoid and tanh (vanishing gradients), step (no gradient).

**Output layer**

- **One real number:** Identity.
- **Positive real number:** Softplus (or exp).
- **One yes/no answer:** 1 neuron + Sigmoid.
- **Several independent yes/no answers:** N neurons + Sigmoid each.
- **Exactly one of N classes:** N neurons + Softmax (Log-Softmax if you need log-probabilities).
- **Value in [-1, 1]:** Tanh.

**At inference time, above all:** use exactly the activation (and exact variant, e.g. GELU vs. tanh-approximated GELU) the model was trained with.
