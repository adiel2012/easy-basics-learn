# Mini Autograd: Learning How Machines Compute Gradients

> **The one idea in this guide:** every operation knows its own small derivative rule. Autograd records the operations as a graph while it computes, then walks that graph backwards. At each step it multiplies the incoming gradient by the local derivative, and it adds the contributions from different paths. That is enough to train a neural network.

This guide explains a small Python program called a **mini autograd**. It is about 188 lines long, and it is a real (tiny) version of the engine that trains neural networks.

You only need basic algebra to follow along. Each calculus rule is explained in plain words when it first appears, and we build up the ideas one step at a time.

## 1. In simple words

> **Autograd automatically figures out how much each input affects the final answer.**

That "how much" is called a **gradient**.

### What is a gradient?

Imagine you are adjusting the volume knob on a speaker. If you turn it a little and the sound gets louder, the knob's effect is **positive**. If the sound gets quieter, the effect is **negative**. The size of the change tells you **how strongly** the knob matters.

A gradient is exactly that:

- It answers: *"If I nudge this input a tiny bit, how quickly does the output change, at its current value?"*
- For a tiny nudge, the output changes by about **gradient × nudge**.
  Example: for `f(x) = x²`, the gradient at `x = 3` is `6`. Increasing `x` by `0.01` changes `f` by about `6 × 0.01 = 0.06`. The exact change is `0.0601`, so the estimate is very close for tiny nudges.
- A **positive** gradient means a small increase in the input increases the output.
- A **negative** gradient means a small increase in the input decreases the output.
- A **larger** gradient (ignoring its sign) means the output changes faster. Its units are output units per input unit, so its size also depends on the units you measure in.

Why care? If you know the gradient, you know which way to nudge things to make the output better. This is how neural networks learn.

## 2. What you need to know first

- **Basic algebra:** adding, multiplying, and squaring numbers.
- **What a gradient means:** explained in section 1 above. For the rule that chains gradients together, see **[The Chain Rule](chain_rule.md)**.
- **A little Python:** classes and methods. Python lets a class decide what `+` and `*` mean for its objects by defining methods named `__add__` and `__mul__`. This guide uses that trick.
- **Optional:** **[Backpropagation](backpropagation.md)** works through the same calculations by hand for a small network. This guide shows how a program does them automatically.

## 3. The key terms

| Term | Simple meaning |
|---|---|
| **Autograd** | Automatic gradient calculation |
| **Gradient** | How quickly an output changes when an input is nudged slightly |
| **Local derivative** | The rate of change of one operation's output with respect to one of its inputs |
| **Computation graph** | The record of which operations produced which values |
| **Backpropagation** | Passing gradients backwards through the graph, using the chain rule |
| **Forward pass** | Computing the output from the inputs |
| **Backward pass** | Computing gradients from the output back to the inputs |
| **Loss** | One number that measures how wrong the predictions are |
| **Gradient descent** | Moving each weight a small step against its gradient to lower the loss |
| **Learning rate** | How big each gradient descent step is |
| **Epoch** | One full pass over the training data |
| **Frozen parameter** | A value the update step leaves unchanged |

## 4. How it works

### Step 1: a tiny example by hand

Let's compute something simple:

    c = a * b
    a = 2
    b = 3

So `c = 6`.

**Question:** If I increase `a` by a tiny amount, how much does `c` change?

Think: `c = a * b`. Each extra 1 in `a` adds `b` (which is 3) to `c`. So the gradient of `c` with respect to `a` is **3**. Similarly, the gradient with respect to `b` is **2** (the value of `a`).

That's all autograd does: it applies rules like this automatically, for any chain of the operations it supports.

### Step 2: the computation graph

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

### Step 3: the code, piece by piece

#### The `Value` class

```python
class Value:
    def __init__(self, data, _children=(), _op="", requires_grad=True):
        self.data = float(data)      # the actual number
        self.grad = 0.0              # how much the final answer depends on this
        self._backward = lambda: None  # instructions for passing gradients back
        self._prev = tuple(dict.fromkeys(_children))  # the inputs, unique, in a fixed order
        self._op = _op               # the operation name, e.g. "+" or "*"
        self.requires_grad = requires_grad  # False = frozen: the update step skips it (gradients are still computed)
```

**Note:** in this engine, `requires_grad` only controls whether the training loop updates a value. It does not stop the graph from being built or gradients from being calculated. In PyTorch, the flag with the same name controls whether operations on a tensor are tracked for automatic differentiation at all.

Every number in the program is wrapped in a `Value`. Each one stores:

| Field | Meaning |
|---|---|
| `data` | The number itself |
| `grad` | The gradient (filled in during backprop) |
| `_prev` | The inputs it was computed from (passed in as `_children`), with duplicates removed. These are the graph links. |
| `_backward` | A small function that knows how to send gradients back |
| `_op` | The operation's name, used only for display |
| `requires_grad` | Whether the update step may change this value |

Why `tuple(dict.fromkeys(...))` instead of a `set`? Both remove duplicates, so `x * x` lists `x` only once. But the order of a `set` of objects can change from one run to the next. That order decides the order in which gradients are added up, and adding floating-point numbers in a different order can change the last digits. Usually that does not matter, but in an unstable training run tiny differences can grow until two runs of the same script give different results. Keeping a fixed order removes this source of run-to-run differences. In our tests, repeated runs of the same script on the same computer then gave identical results. It does not guarantee bit-for-bit identical results on different hardware or Python versions.

#### Addition: the simplest rule

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

`_wrap` turns a plain number into a `Value`, so you can write `x + 1` instead of `x + Value(1)`.

#### Multiplication

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

**In plain words:** If `out = x * y`, the rate of change of `out` with respect to `x` is `y`, and with respect to `y` is `x`. For a small nudge, the change in `out` is approximately that rate multiplied by the nudge. That is exactly the rule we used by hand in Step 1.

#### Powers

```python
def __pow__(self, exponent):
    assert isinstance(exponent, (int, float)), "only int/float exponents"
    out = Value(self.data ** exponent, (self,), f"**{exponent}")

    def _backward():
        if exponent == 0:  # x ** 0 is the constant 1, so its gradient is 0
            return
        self.grad += exponent * (self.data ** (exponent - 1)) * out.grad

    out._backward = _backward
    return out
```

**In plain words:** This uses the classic calculus rule: the derivative of `x^n` is `n * x^(n-1)`. The exponent `n` must be a plain number, such as `2` or `-1`, not another `Value`. The `exponent == 0` check handles one edge case: `x ** 0` is always 1, so its gradient is 0, but the general formula would try to compute `0 ** -1` when `x = 0` and crash.

#### Subtraction, division, negation

These are built from the ones above, so we do not need separate rules:

- `-x` is `x * -1`
- `x - y` is `x + (-y)`
- `x / y` is `x * y^(-1)`

This is a nice trick: these three operations need no new backward rules. All the rules in this engine apply only where the operation is defined and has a slope. For example, division by zero is undefined, `log(x)` needs `x > 0`, and a fractional power of a negative number is not a real number. The engine does not check these cases itself. Usually Python raises an error (such as `ZeroDivisionError` or `ValueError`), but very large values can also overflow.

#### Activation functions

```python
def tanh(self):
    t = math.tanh(self.data)
    out = Value(t, (self,), "tanh")

    def _backward():
        self.grad += (1 - t * t) * out.grad

    out._backward = _backward
    return out
```

`tanh` squashes any number into a value between -1 and 1. Neural networks use functions like this to add "bend" to their behaviour.

#### Sigmoid

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

`sigmoid` squashes any number into a value between 0 and 1, so its output can be read as an estimated probability of class 1. This estimate is not necessarily a perfectly calibrated probability. It is used at the output of the XOR network later in this guide.

The two-branch form only prevents overflow inside `exp()`. It does not prevent **saturation**: for large inputs the output rounds to exactly 1 or 0 (at an input of 40 it is already exactly `1.0`), and then the sigmoid's local gradient `s × (1 − s)` is exactly 0. Nothing flows back through the sigmoid to the values before it, although other branches of the graph can still have gradients.

The code also includes `relu`, `exp`, and `log`, which follow the same pattern. At exactly 0, `relu` has no single slope; this engine uses a gradient of 0 there.

#### The `backward()` method: the engine

```python
def backward(self):
    # topological order via DFS
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

Why reverse order? A value may feed into several later operations. Processing results before their inputs means that, by the time a value's turn comes, every operation that uses it has already added its gradient contribution. The value then passes its complete, accumulated gradient farther backward.

**Important:** gradients accumulate. Each `_backward()` adds to `.grad` with `+=` and never overwrites it, so calling `backward()` twice on the same graph gives wrong gradients. It is worse than simple doubling: the intermediate nodes also keep their old gradients and pass them back again. In the backward pass example below, a second call leaves `a.grad` at 9 instead of the correct 3. Before recomputing gradients on an existing graph, reset `.grad` to zero on every node in that graph, including intermediate nodes. Resetting only the parameters is not enough, because `backward()` resets only the output's gradient (to 1). The `zero_grad()` method below resets every node for you. The XOR training loop later in this guide is safe: each epoch builds a brand-new graph, so its intermediate nodes start at zero, and the loop resets the reused parameters' gradients before calling `backward()`.

#### Resetting gradients with `zero_grad()`

```python
def zero_grad(self):
    # set .grad to 0 on this value and every value it was computed from
    visited = set()

    def clear(v):
        if v not in visited:
            visited.add(v)
            v.grad = 0.0
            for child in v._prev:
                clear(child)

    clear(self)
```

`zero_grad()` walks the same graph as `backward()` and sets every gradient back to 0, including the intermediate nodes. Use it when you want to call `backward()` on the same graph again:

```python
from mini_autograd import Value

a = Value(2.0)
b = Value(3.0)
e = Value(4.0)
d = a * b + e

d.backward()
d.backward()
print(a.grad)  # 9.0  (wrong: old gradients were added again)

d.zero_grad()
d.backward()
print(a.grad)  # 3.0  (correct)
```

The XOR training loop below does not need `zero_grad()`, because it builds a new graph every epoch.

#### Backward pass example

Take the graph `c = a * b`, then `d = c + e`, with `a = 2`, `b = 3`, and `e = 4`.

**Forward pass** (left to right): `c = 2 × 3 = 6`, then `d = 6 + 4 = 10`.

**Backward pass** (right to left), starting with `d.grad = 1`:

1. `d = c + e` passes the gradient to both inputs unchanged: `c.grad = 1` and `e.grad = 1`.
2. `c = a * b` multiplies the **incoming gradient** (`c.grad`, which came from downstream) by the **local derivative** (the other input, which is the factor from the multiplication itself): `a.grad = b × c.grad = 3 × 1 = 3` and `b.grad = a × c.grad = 2 × 1 = 2`. This multiplication of an incoming gradient by a local derivative is the chain rule in action.

Here is the same graph with each value and its gradient after the backward pass:

        a = 2   (grad 3) ──┐
                           ├── [ * ] ──► c = 6   (grad 1) ──┐
        b = 3   (grad 2) ──┘                                ├── [ + ] ──► d = 10   (grad 1)
        e = 4   (grad 1) ───────────────────────────────────┘

These are the values the backward functions compute. They follow the addition and multiplication rules described above.

In general, the gradient of the final result `L` with respect to an input `a` is the **incoming gradient** times the **local derivative**: `∂L/∂a = (∂L/∂c) × (∂c/∂a)`. The engine stores `∂L/∂c` in `c.grad`, and each operation supplies its own local derivative `∂c/∂a`. If `a` feeds into several operations, its gradient is the sum of these products over all paths.

### Step 4: check the gradients numerically

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

The **numeric** lines are a check. They estimate the gradient by nudging the input by a tiny amount (`h = 0.000001`) and measuring the change, using the forward-difference formula `gradient ≈ (f(x + h) − f(x)) / h`. A central difference, `(f(x + h) − f(x − h)) / 2h`, is usually more accurate, but the forward difference is accurate enough for this check. The autograd numbers closely match the numerical estimates, which is a good check that the gradient calculations are correct for this example. This checks one expression at one point. It does not test every backward rule, repeated inputs such as `x * x`, or graphs with several branches. Repeating the check for each operation and for a few small graphs would give much more confidence.

## 5. Try it yourself

### A first experiment

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

### Train a network on XOR

So far we have computed gradients for single expressions. Now we use them to **train** a small neural network, the same way real networks learn.

#### The problem: XOR

XOR ("exclusive or") outputs 1 when exactly one input is 1:

| x1 | x2 | target |
|---|---|---|
| 0 | 0 | 0 |
| 0 | 1 | 1 |
| 1 | 0 | 1 |
| 1 | 1 | 0 |

Here are the four points, with `●` for class 1 and `○` for class 0:

```
 x2
  1 │  ● (0,1)        ○ (1,1)
    │
  0 │  ○ (0,0)        ● (1,0)
    └─────────────────────────── x1
       0                1
```

The two `●` points sit on one diagonal and the two `○` points on the other. Any straight line that puts both `●` points on one side also puts at least one `○` point there.

This is a **classification** problem: each input belongs to class 0 or class 1. XOR is a famous test because **no single straight line can separate the 1s from the 0s**. The network needs a **hidden layer** to bend the decision boundary.

#### The network

```
x1 ──┐
     ├──► 4 hidden neurons (tanh) ──► 1 output neuron (sigmoid) ──► estimated probability of class 1
x2 ──┘
```

- Each **hidden neuron** receives both inputs and computes `tanh(w1·x1 + w2·x2 + b)`.
- The **output neuron** combines the four hidden values with its own weights and then applies `sigmoid`. The sigmoid squeezes the result into (0, 1), giving a number the network can use as an estimated probability of class 1. This estimate is not necessarily a perfectly calibrated probability.

#### The training loop

Training repeats the same three steps many times:

1. **Forward pass.** Run each of the four examples through the network. Compare each prediction to its target, and add up the squared errors into one number called the **loss**. A lower loss usually means better predictions. The script adds the four errors instead of averaging them. Averaging is also common; it only rescales the gradients, so the learning rate would need adjusting. Real classifiers usually use a different loss called cross-entropy, but squared error is simpler and works fine for showing how autograd trains a network.
2. **Backward pass.** Call `loss.backward()`. This fills in `grad` for every weight, telling us how the loss changes if that weight changes. Before this step we reset the old gradients to zero, because gradients accumulate by default.
3. **Update.** Move each weight a small step in the opposite direction of its gradient:

       weight = weight - learning_rate × gradient

   If a gradient is positive, a small increase in the weight would increase the loss, so we decrease the weight. If it is negative, we increase the weight. This is called **gradient descent**. The **learning rate** controls how big each step is. For example, if a weight is `2.0`, its gradient is `0.6`, and the learning rate is `0.1`, the new weight is `2.0 − 0.1 × 0.6 = 1.94`. The gradient is positive, so the weight goes down. The gradient only describes the loss very close to the current weights, so a step that is too large can still make the loss worse.

Repeat for many **epochs** (full passes over the data). Usually the loss shrinks and the predictions improve. The loss printed for an epoch is measured before that epoch's weight update, because the forward pass runs before the update step.

#### The script

Save this as `xor_train.py` in the **same folder as `mini_autograd.py`**, because it imports the `Value` class from there. It uses the engine's `+`, `-`, `*`, `**`, `tanh`, `sigmoid`, and `backward()` without changes.

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
        return z.sigmoid()  # output in (0, 1): an estimated probability of class 1

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

# Freezing: a parameter with requires_grad = False keeps its current value.
# Set freeze_hidden = True to train only the output neuron.
freeze_hidden = False
if freeze_hidden:
    for n in model.hidden:
        for p in n.params():
            p.requires_grad = False

trainable = sum(1 for p in params if p.requires_grad)
print(f"trainable parameters: {trainable} of {len(params)}")

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
        if p.requires_grad:  # frozen parameters are skipped
            p.data -= learning_rate * p.grad

    if epoch % 400 == 0 or epoch == epochs - 1:
        print(f"epoch {epoch:4d}   loss {loss.data:.5f}")

print()
print("x1  x2  target  output  predicted")
for x, y in zip(X, Y):
    out = model([Value(v) for v in x]).data
    print(f"{x[0]}   {x[1]}     {y}      {out:.3f}   {int(out >= 0.5)}")
```

#### Running it

```
python xor_train.py
```

Output from running the script as written (`random.seed(1)`). Other random seeds may produce different loss values and predictions. Some initializations may require more training or a different learning rate.

```
trainable parameters: 17 of 17
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

The printed losses are measured before each epoch's update, but the final table is computed after the last update, so the two use slightly different weights.

Read the last table:

- **output** is the network's estimated probability that the answer is 1. It is not guaranteed to be perfectly calibrated.
- **predicted** turns that probability into a class: 1 if the output is at least 0.5, otherwise 0.
- When **predicted** matches **target** in every row, the network has learned XOR.
- A lower loss usually means better predictions, but it does not guarantee correct classifications at every step.
- The network learns outputs near the correct labels on these four training examples. This alone does not show how well it would generalize to other data. Four examples can be memorized, so success here is not proof of broad learning.

## 6. Variations and common mistakes

### Freezing some parameters

Sometimes you want to keep a value fixed during training. Each `Value` has a `requires_grad` flag. When it is `False`, the value is **frozen**: the update step skips it, so its number never changes.

The flag does nothing by itself: the update loop checks it and skips frozen values. Gradients are still calculated for a frozen value, and they still flow through it to other values. Being part of the graph, receiving a gradient, and being updated are three separate things. In the example below, `x` is frozen, but `w` still gets its gradient.

A simple example, with `y = w * x`:

```python
from mini_autograd import Value

w = Value(2.0)                       # trainable
x = Value(3.0, requires_grad=False)  # frozen

y = w * x
y.backward()
print(w.grad)  # 3.0  (the gradient is still computed for both values)
print(x.grad)  # 2.0

lr = 0.1
for p in [w, x]:
    if p.requires_grad:   # skip frozen values
        p.data -= lr * p.grad

print(w.data)  # 1.7  (2.0 - 0.1 * 3.0)
print(x.data)  # 3.0  (frozen, unchanged)
```

The XOR script uses the same idea. Setting `freeze_hidden = True` freezes all 12 hidden-layer weights and biases, so only the 5 output parameters learn, and the script prints `trainable parameters: 5 of 17`. The hidden layer keeps its random starting values, and whether the output neuron can still solve XOR depends on them. With seed 1 it cannot: the loss only falls from about 1.01 to 0.97, and 3 of 4 cases are correct.

### Limits of this engine

This is a teaching engine, not a production library:

- It works on single numbers (scalars), not arrays or tensors, so it is slow for large networks.
- `backward()` walks the graph recursively. Very deep graphs can exceed Python's recursion limit.
- `backward()` expects a single-number output and starts it with a gradient of `1.0`.
- Each `Value` keeps references to its inputs, so a graph uses memory until it is no longer needed. The XOR loop builds a new graph each epoch and lets the old one go.
- Exponents in `**` must be plain numbers, not other `Value`s.
- Gradients accumulate, so you must reset them (for example with `zero_grad()`) before reusing a graph.
- `requires_grad` only freezes updates. It does not turn off gradient calculation.
- The snippets in this guide show the main rules. The complete `mini_autograd.py` also has helper methods such as `__radd__`, `__sub__`, `__neg__`, and `__truediv__`, so use the full file to run the examples.

### Common mistakes

- **Forgetting to reset gradients.** Gradients accumulate with `+=`. If you reuse a graph or reuse parameters without setting `.grad` back to zero, old gradients are added to new ones.
- **Calling `backward()` twice on the same graph.** The gradients come out wrong, and not just doubled, because intermediate nodes pass their old gradients back again. Call `zero_grad()` first, or build a fresh graph.
- **Choosing a learning rate that is too large.** Each step can overshoot, and with `10.0` the XOR network gets stuck and stops learning.
- **Reading the sigmoid output as an exact probability.** It is an estimate between 0 and 1, not necessarily a calibrated probability.
- **Thinking `requires_grad = False` stops gradients.** In this engine it only stops updates; gradients are still calculated and still flow through the value.
- **Expecting one hidden neuron to solve XOR.** One neuron can only draw one straight boundary, and XOR needs more than that.
- **Copying only the snippets.** The snippets show the main rules, but the scripts need the complete `mini_autograd.py`.

## 7. Remember this

- A **gradient** tells you how quickly the output changes when an input changes slightly.
- Each operation stores a **local derivative rule** for passing gradients back to its inputs.
- **`backward()`** orders the graph, then applies those rules from the output back to the inputs. Each input's gradient is the incoming gradient times the local derivative, added up over all paths.
- Training repeats three steps: the **forward pass** computes the loss, the **backward pass** computes gradients, and the **update step** nudges each weight against its gradient.
- Large neural networks use the same principles: record operations during the forward pass, send gradients backward with the chain rule, and update the parameters. Production frameworks extend this to tensors, fast vectorized operations, and more advanced training methods.

### Quick check

1. Compute the gradient of `w = (a + b) * (a + b)` for `a = 1`, `b = 2`. Predict first, then check.
2. Add a new operation that is not in the code yet, such as `softplus(x) = log(1 + exp(x))`, by following the `tanh` pattern. Hint: its derivative is the sigmoid.
3. Build a tiny neuron: `out = (w * x + bias).tanh()`, then call `backward()` and print the gradient for `w`.
4. Change `learning_rate` to `0.05`, `5.0`, and `10.0`. Which one learns slowly, which one learns faster, and which one breaks training?
5. Change `n_hidden` from `4` to `1`. Can a single hidden neuron learn XOR? Why or why not? Hint: as the inputs change, can the output of a single tanh neuron followed by a sigmoid ever go up and then down again?
6. Change `random.seed(1)` to a different number. Does training still work every time?
7. Print `params[0].grad` before and after `loss.backward()` to see gradients appear.

Try answering before reading on. For questions 4 to 7, change `xor_train.py` and run it.

---

### Check your answers

1. `w = (1 + 2)² = 9`, and both gradients are `6`, because `dw/da = 2(a + b) = 6` and the same for `b`. The sum `a + b` is used twice in the multiplication, and the engine adds both contributions.
2. Add this method to the `Value` class:

   ```python
   def softplus(self):
       x = self.data
       # stable form of log(1 + exp(x)): never calls exp() on a large positive number
       sp = max(x, 0.0) + math.log1p(math.exp(-abs(x)))
       out = Value(sp, (self,), "softplus")

       def _backward():
           # the local derivative is the sigmoid, computed in its stable form
           if x >= 0:
               s = 1.0 / (1.0 + math.exp(-x))
           else:
               e = math.exp(x)
               s = e / (1.0 + e)
           self.grad += s * out.grad

       out._backward = _backward
       return out
   ```

   The local derivative of softplus is the sigmoid. The simple formula `math.log(1 + math.exp(x))` also works, but it overflows for large `x` (around 710 and above). The version above rewrites it so `exp()` only ever sees zero or negative numbers, and it works even for `x = 1000` or `x = -1000`.
3. The gradient for `w` is `x × (1 − tanh²(w·x + bias))`: the incoming gradient through `tanh`, times `x` from the multiplication. For example, with `w = 0.5`, `x = 2`, and `bias = 0.1`, the output is about 0.80 and `w.grad` is about 0.72.

**Answers from running the script** (seed 1, 2000 epochs; other seeds give somewhat different numbers):

4. These results are for this exact script and seed, not general rules about learning rates. `0.05` learns slowly: the final loss is about 0.009, though all four cases are still correct. `5.0` learns faster: the final loss is about 0.00004. `10.0` breaks training: the loss jumps from about 1.0 to about 2.0 and stays there. The output is about 1 for every input, so only 2 of 4 cases are correct. The steps were so large that the sigmoid output got pushed to the edge of its range and stuck near 1. There its slope `s × (1 − s)` is almost zero, so the gradients are almost zero and training cannot recover.
5. No. With one hidden neuron the loss stalls near 0.67 and only 3 of 4 cases are correct. In tests with 20 different seeds, none solved XOR. The reason: the hidden neuron computes `tanh(w1·x1 + w2·x2 + b)`, and the output applies a sigmoid to a scaled version of that. Tanh and sigmoid are both increasing functions: a bigger input always gives a bigger output. So the final output only rises (or only falls, if the output weight is negative) as `w1·x1 + w2·x2` grows. The points where the output equals 0.5 therefore form a straight line (or there are none, and the network gives every input the same class). Neither can separate XOR's classes. With several hidden neurons, each one turns its own weighted sum of the inputs into a nonlinear feature. The output neuron combines these features, which lets the network form a curved decision boundary that a single neuron cannot.
6. In tests with seeds 0 to 49, every seed learned XOR. That does not guarantee that every seed, or every other setting, will work.
7. Before `loss.backward()`, every gradient is `0.0`, because the loop has just reset them. After it, each parameter holds a gradient that tells the update step which way to move it.

## 8. Going further

- **[The Chain Rule](chain_rule.md)**: the math rule that `backward()` applies at every step.
- **[Backpropagation](backpropagation.md)**: the same gradients worked out by hand for one full training step.
- **[MLP Inference: XOR](mlp_xor_inference.md)**: how a trained XOR network turns inputs into an answer.

### Learn more

- 📖 [micrograd README](https://github.com/karpathy/micrograd) (Andrej Karpathy): a tiny autograd engine with the same design as this one
- 📖 [How the backpropagation algorithm works](https://neuralnetworksanddeeplearning.com/chap2.html) (Michael Nielsen, chapter 2): backpropagation applied to whole networks
