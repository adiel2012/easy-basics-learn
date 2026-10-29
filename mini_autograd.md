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

- It answers: *"If I nudge this input a tiny bit, how much does the output change?"*
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
        ├── (*) ──► c ──┐
    b ──┘               ├── (+) ──► d
                        │
    e ──────────────────┘

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

`tanh` squashes any number into the range -1 to 1. Neural networks use functions like this to add "bend" to their behaviour. The code also includes `relu`, `exp`, and `log`, which follow the same pattern.

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

1. **Sort the graph (`build`).** It lists every value in an order where each value comes *after* the values it depends on.
2. **Start at the end.** The final output's gradient is set to `1.0`, because the output's change with respect to itself is 1.
3. **Walk backwards.** It runs each value's `_backward()` function in reverse order, so gradients flow from the output back to the inputs.

The order matters. By the time we process a value, all the gradients flowing into it have already been added up.

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
