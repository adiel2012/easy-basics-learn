# Mini Autograd: Learning How Machines Compute Gradients

This guide explains a small Python program called a **mini autograd**. It is about 150 lines long, and it is a real (tiny) version of the engine that trains neural networks.

You do not need any math beyond basic algebra to follow along. We will build up the ideas one step at a time.

---

## 1. The big idea in one sentence

> **Autograd automatically figures out how much each input affects the final answer.**

That "how much" is called a **gradient**.

---

## 2. What is a gradient? (plain English)

Imagine you are adjusting the volume knob on a speaker. If you turn it a little and the sound gets louder, the knob's effect is **positive**. If the sound gets quieter, the effect is **negative**. The size of the change tells you **how strongly** the knob matters.

A gradient is exactly that:

- It answers: *"If I nudge this input a tiny bit, how quickly does the output change, at its current value?"*
- For a tiny input change, the output change is approximately the gradient multiplied by the input change.
- A gradient tells you how sensitive the output is to a small change in an input, measured in that input's units.
  Example: for `f(x) = x²`, the gradient at `x = 3` is `6`. A tiny increase of `0.01` in `x` changes `f` by about `6 × 0.01 = 0.06`.
- A **positive** gradient means: increasing the input increases the output.
- A **negative** gradient means: increasing the input decreases the output.
- A **big** number means the input matters a lot. A **small** number means it barely matters.

Why care? If you know the gradient, you know which way to nudge things to make the output better. This is how neural networks learn.

---

## 3. A tiny example by hand

Let's compute something simple:

    c = a * b
    a = 2
    b = 3

So `c = 6`.

**Question:** If I increase `a` by a tiny amount, how much does `c` change?

Think: `c = a * b`. Each extra 1 in `a` adds `b` (which is 3) to `c`. So the gradient of `c` with respect to `a` is **3**. Similarly, the gradient with respect to `b` is **2** (the value of `a`).

That's all autograd does: it applies rules like this automatically, for any chain of operations, no matter how long.

---

## 4. The core idea: a "computation graph"

Every time the program does math, it remembers:

1. **What it computed** (the result), and
2. **What it was computed from** (the inputs).

If you chain several operations together, you get a graph of connected steps. For example:

    a ──┐
        ├── [ * ] ──► c ──┐
    b ──┘                 ├── [ + ] ──► d
    e ────────────────────┘

Here `c = a * b` and `d = c + e`. The `+` node receives both `c` and `e`.

Autograd records this graph while the forward calculation runs. Then, to find gradients, it walks **backwards** through the graph. This is called **backpropagation**.

---

## 5. Reading the code, piece by piece

### The `Value` class

```python
class Value:
    def __init__(self, data, _children=(), _op=""):
        self.data = float(data)      # the actual number
        self.grad = 0.0              # how much the final answer depends on this
        self._backward = lambda: None  # instructions for passing gradients back
        self._prev = set(_children)  # the inputs that created this value
        self._op = _op               # the operation name, e.g. "+" or "*"
```

Every number in the program is wrapped in a `Value`. Each one stores:

| Field | Meaning |
|---|---|
| `data` | The number itself |
| `grad` | The gradient (filled in during backprop) |
| `_prev` | The "parents" it came from (the graph links) |
| `_backward` | A small function that knows how to send gradients back |

### Addition: the simplest rule

```python
def __add__(self, other):
    other = self._wrap(other)
    out = Value(self.data + other.data, (self, other), "+")

    def _backward():
        self.grad += out.grad
        other.grad += out.grad

    out._backward = _backward
    return out
```

**In plain words:** If `out = x + y`, then nudging `x` by 1 changes `out` by 1, and nudging `y` by 1 also changes `out` by 1. So each input gets the full gradient of the output.

### Multiplication

```python
def __mul__(self, other):
    other = self._wrap(other)
    out = Value(self.data * other.data, (self, other), "*")

    def _backward():
        self.grad += other.data * out.grad
        other.grad += self.data * out.grad

    out._backward = _backward
    return out
```

**In plain words:** If `out = x * y`, nudging `x` changes `out` by `y`, and nudging `y` changes `out` by `x`. That is exactly the rule we used by hand in section 3.

### Powers

```python
def __pow__(self, exponent):
    out = Value(self.data ** exponent, (self,), f"**{exponent}")

    def _backward():
        self.grad += exponent * (self.data ** (exponent - 1)) * out.grad

    out._backward = _backward
    return out
```

**In plain words:** This uses the classic calculus rule: the derivative of `x^n` is `n * x^(n-1)`.

### Subtraction, division, negation

These are built from the ones above, so we do not need separate rules:

- `-x` is `x * -1`
- `x - y` is `x + (-y)`
- `x / y` is `x * y^(-1)`

This is a nice trick: a few basic rules cover everything.

### Activation functions

```python
def tanh(self):
    t = math.tanh(self.data)
    out = Value(t, (self,), "tanh")

    def _backward():
        self.grad += (1 - t * t) * out.grad

    out._backward = _backward
    return out
```

`tanh` squashes any number into the range -1 to 1. Neural networks use functions like this to add "bend" to their behaviour.

### Sigmoid

```python
def sigmoid(self):
    x = self.data
    # numerically stable form: avoids overflow in exp() for large |x|
    if x >= 0:
        s = 1.0 / (1.0 + math.exp(-x))
    else:
        e = math.exp(x)
        s = e / (1.0 + e)
    out = Value(s, (self,), "sigmoid")

    def _backward():
        self.grad += s * (1 - s) * out.grad

    out._backward = _backward
    return out
```

`sigmoid` squashes any number into the range 0 to 1, so its output can be read as a probability. It is used at the output of the XOR network later in this guide.

The code also includes `relu`, `exp`, and `log`, which follow the same pattern.

### The `backward()` method: the engine

```python
def backward(self):
    topo, visited = [], set()

    def build(v):
        if v not in visited:
            visited.add(v)
            for child in v._prev:
                build(child)
            topo.append(v)

    build(self)

    self.grad = 1.0
    for v in reversed(topo):
        v._backward()
```

This is the heart of the program. It does three things:

1. **Sort the graph (`build`).** It builds an order that places each input before the result computed from it.
2. **Start at the end.** The final output's gradient is set to `1.0`, because the output's change with respect to itself is 1.
3. **Walk backwards.** It reverses that order, so the final output is processed first and gradients flow toward the original inputs. Each value runs its `_backward()` function.

The order matters. When a value is processed, all operations that use it have already contributed their gradients. It can now pass its accumulated gradient farther backward.

---

## 6. Running the example

The bottom of the file tests the code on this function:

    f(a, b) = tanh(a*b + a²) / b

with `a = 2` and `b = -3`.

Run it:

```
python mini_autograd.py
```

You should see:

```
f = 0.32134...
df/da = -0.02355...
df/db = 0.06001...
numeric df/da ~ -0.02355...
numeric df/db ~ 0.06001...
```

The **numeric** lines are a check. They estimate the gradient by nudging the input by a tiny amount (`h = 0.000001`) and measuring the change. The autograd numbers match, which confirms the engine works.

---

## 7. Try it yourself

Open a Python shell in the same folder and try these:

```python
from mini_autograd import Value

x = Value(3.0)
y = Value(4.0)
z = x * y + x        # z = 3*4 + 3 = 15
z.backward()

print(z.data)   # 15.0
print(x.grad)   # 5.0  (y + 1)
print(y.grad)   # 3.0  (x)
```

Check the results by hand. If they match, you understand the core idea.

**Challenges:**

1. Compute the gradient of `w = (a + b) * (a + b)` for `a = 1`, `b = 2`. Predict first, then check.
2. Add a new operation, such as `sigmoid`, by following the `tanh` pattern.
3. Build a tiny neuron: `out = (w * x + bias).tanh()`, then call `backward()` and print the gradient for `w`.

---

## 8. Key takeaways

- A **gradient** tells you how much the output changes when an input changes.
- Each operation stores **how to pass gradients back** to its inputs.
- **`backward()`** sorts the operations and runs those rules from the end to the start.
- Neural networks use exactly this process to learn: compute the output, measure the error, get gradients, and adjust.

## 9. Glossary

| Term | Simple meaning |
|---|---|
| **Autograd** | Automatic gradient calculation |
| **Gradient** | How much an output changes when an input is nudged |
| **Computation graph** | The record of which operations produced which values |
| **Backpropagation** | Passing gradients backwards through the graph |
| **Forward pass** | Computing the output from the inputs |
| **Backward pass** | Computing gradients from the output back to the inputs |

---

## 10. Putting it all together: training a network on XOR

So far we have computed gradients for single expressions. Now we use them to **train** a small neural network, the same way real networks learn.

### The problem: XOR

XOR ("exclusive or") outputs 1 when exactly one input is 1:

| x1 | x2 | target |
|---|---|---|
| 0 | 0 | 0 |
| 0 | 1 | 1 |
| 1 | 0 | 1 |
| 1 | 1 | 0 |

This is a **classification** problem: each input belongs to class 0 or class 1. XOR is a famous test because **no single straight line can separate the 1s from the 0s**. The network needs a **hidden layer** to bend the decision boundary.

### The network

```
x1 ──┐                      ┌── output (sigmoid) ── probability of class 1
     ├── 4 hidden neurons ──┤
x2 ──┘   (tanh)             └
```

- Each **hidden neuron** computes `tanh(w1·x1 + w2·x2 + b)`.
- The **output neuron** combines the four hidden values with its own weights and then applies `sigmoid`. The sigmoid squeezes the result into (0, 1), so we can read it as a probability.

### The training loop

Training repeats the same three steps many times:

1. **Forward pass.** Run each of the four examples through the network. Compare each prediction to its target, and add up the squared errors into one number called the **loss**. A lower loss means better predictions.
2. **Backward pass.** Call `loss.backward()`. This fills in `grad` for every weight, telling us how the loss changes if that weight changes. Before this step we reset the old gradients to zero, because gradients accumulate by default.
3. **Update.** Move each weight a small step in the opposite direction of its gradient:

       weight = weight - learning_rate × gradient

   If a gradient is positive, a sufficiently small increase in the weight would increase the loss locally, so gradient descent moves the weight in the opposite direction. A step that is too large can still increase the loss, which is why the learning rate matters. If it is negative, we increase the weight. The **learning rate** controls how big each step is.

Repeat for many **epochs** (full passes over the data). The loss shrinks and the predictions improve.

### The script

Save this as `xor_train.py` in the **same folder as `mini_autograd.py`**, because it imports the `Value` class from there. It reuses the existing `+`, `*`, `tanh`, `sigmoid`, and `backward()` code without changes.

```python
"""
Train a tiny neural network to learn XOR, using the Value autograd engine
from mini_autograd.py.

XOR truth table:
    x1  x2  ->  y
     0   0  ->  0
     0   1  ->  1
     1   0  ->  1
     1   1  ->  0

A single straight line cannot separate these classes, so the network
needs a hidden layer.
"""
import random
from mini_autograd import Value

random.seed(1)


class Neuron:
    def __init__(self, n_in):
        self.w = [Value(random.uniform(-1, 1)) for _ in range(n_in)]
        self.b = Value(0.0)

    def __call__(self, x):
        # w1*x1 + w2*x2 + ... + b, then tanh to add non-linearity
        z = sum((wi * xi for wi, xi in zip(self.w, x)), self.b)
        return z.tanh()

    def params(self):
        return self.w + [self.b]


class MLP:
    """2 inputs -> hidden layer of tanh neurons -> 1 sigmoid output."""

    def __init__(self, n_in, n_hidden):
        self.hidden = [Neuron(n_in) for _ in range(n_hidden)]
        self.out_w = [Value(random.uniform(-1, 1)) for _ in range(n_hidden)]
        self.out_b = Value(0.0)

    def __call__(self, x):
        h = [n(x) for n in self.hidden]
        z = sum((wi * hi for wi, hi in zip(self.out_w, h)), self.out_b)
        return z.sigmoid()  # output in (0, 1): read it as a probability

    def params(self):
        ps = []
        for n in self.hidden:
            ps += n.params()
        return ps + self.out_w + [self.out_b]


X = [[0, 0], [0, 1], [1, 0], [1, 1]]
Y = [0, 1, 1, 0]

model = MLP(n_in=2, n_hidden=4)
params = model.params()

learning_rate = 0.5
epochs = 2000

for epoch in range(epochs):
    # 1. Forward pass: build the graph and compute the total loss
    loss = Value(0.0)
    for x, y in zip(X, Y):
        pred = model([Value(v) for v in x])
        loss = loss + (pred - y) ** 2  # squared error

    # 2. Backward pass: clear old gradients, then compute new ones
    for p in params:
        p.grad = 0.0
    loss.backward()

    # 3. Update: move each parameter a little against its gradient
    for p in params:
        p.data -= learning_rate * p.grad

    if epoch % 400 == 0 or epoch == epochs - 1:
        print(f"epoch {epoch:4d}   loss {loss.data:.5f}")

print()
print("x1  x2  target  output  predicted")
for x, y in zip(X, Y):
    out = model([Value(v) for v in x]).data
    print(f"{x[0]}   {x[1]}     {y}      {out:.3f}   {int(out >= 0.5)}")
```

### Running it

```
python xor_train.py
```

Expected output (the exact numbers depend on the random seed):

```
epoch    0   loss 1.00875
epoch  400   loss 0.00382
epoch  800   loss 0.00173
epoch 1200   loss 0.00111
epoch 1600   loss 0.00082
epoch 1999   loss 0.00064

x1  x2  target  output  predicted
0   0     0      0.008   0
0   1     1      0.987   1
1   0     1      0.987   1
1   1     0      0.015   0
```

Read the last table:

- **output** is the network's probability that the answer is 1.
- **predicted** turns that probability into a class: 1 if the output is at least 0.5, otherwise 0.
- When **predicted** matches **target** in every row, the network has learned XOR.

### Things to try

1. Change `learning_rate` to `0.05` and to `5.0`. Which one learns slowly, and which one overshoots?
2. Change `n_hidden` from `4` to `1`. Can a single hidden neuron learn XOR? Why or why not?
3. Change `random.seed(1)` to a different number. Does training still work every time?
4. Print `params[0].grad` before and after `loss.backward()` to see gradients appear.

### What you learned

This script is the whole idea behind neural network training:

- The **forward pass** builds a computation graph and produces a loss.
- The **backward pass** (`backward()`) finds how each weight affects that loss.
- The **update step** nudges each weight in the direction that lowers the loss.

Large models such as language models use the same loop. They have billions of weights and far more data, but the steps are the same ones you just ran.
