# The Chain Rule: The Math Behind Backpropagation

> **The one idea in this guide:** when one quantity affects another **through a chain of steps**, **multiply the local derivatives** along the chain. When **several paths** connect an input to the output, **add** their contributions. Backpropagation applies these two rules efficiently to compute neural-network gradients.

## 1. In simple words

A derivative answers one question: **"If I nudge this input a little, how much does the output change?"**

Often the input doesn't affect the output directly. It goes through **several steps**: x changes y, y changes z, and z changes the loss. The chain rule says: **multiply the effects of the steps**.

**Analogy: currency exchange.** You convert dollars to euros, then euros to yen.

```
1 dollar → 0.9 euros        (rate: 0.9 euros per dollar)
1 euro   → 160 yen          (rate: 160 yen per euro)

1 dollar → 0.9 × 160 = 144 yen
```

(This assumes both exchange rates stay constant. That makes the analogy a simple linear example of the chain rule. Real derivatives also work for curved, nonlinear relationships: they measure the rate of change locally, at the current point.)

You never needed a "dollars to yen" table. You **multiplied the rates** along the chain. Derivatives are rates too ("how many units of output per unit of input"), so they chain the same way.

**Analogy 2: gears.** If gear A turns gear B 2 times per turn, and B turns C 3 times per turn, then A turns C **2 × 3 = 6** times per turn.

In a neural network, a weight affects the loss through a long chain of steps. Backpropagation is just the chain rule, applied efficiently to every weight.

## 2. What you need to know first

- **What a derivative means**: dy/dx is how much y changes per small change in x. If y = 3x, then dy/dx = 3.
- A few basic derivatives (all used below):

| Function | Derivative |
|---|---|
| y = a·x (a is a constant) | dy/dx = a |
| y = x + b | dy/dx = 1 |
| y = x² | dy/dx = 2x |
| y = x³ | dy/dx = 3x² |

For how the chain rule is used in a full training step, see **[Backpropagation](backpropagation.md)**.

## 3. The key terms

| Term | Meaning |
|---|---|
| **Derivative** | rate of change of an output with respect to an input |
| **Local derivative** | the derivative of **one step** of the chain, looking only at that step |
| **Chain** | a sequence of steps where each output feeds the next step |
| **Path** | one route from an input to the output through the steps |
| **Partial derivative (∂)** | a derivative when there are several inputs: change one, keep the others fixed. Written ∂z/∂x instead of dz/dx |
| **Computation graph** | a picture of all the steps and how values flow between them |

## 4. How it works

### Rule 1: one chain, multiply

If x affects y, and y affects z:

```
x ──▶ y ──▶ z

dz/dx = dz/dy · dy/dx
```

Read it as: (how z reacts to y) × (how y reacts to x).

**Example.** y = 3x and z = y². Find dz/dx at x = 2.

**Step 1: forward values**

```
x = 2
y = 3·2 = 6
z = 6² = 36
```

**Step 2: local derivatives**

```
dy/dx = 3                   (y = 3x)
dz/dy = 2y = 2·6 = 12       (z = y², evaluated at y = 6)
```

**Step 3: multiply**

```
dz/dx = dz/dy · dy/dx = 12 · 3 = 36
```

**Check without the chain rule:** z = (3x)² = 9x², so dz/dx = 18x = 18·2 = 36. ✅ Same.

**Check numerically:** nudge x from 2 to 2.001:

```
y = 3·2.001 = 6.003
z = 6.003² = 36.036009
change in z / change in x = (36.036009 − 36) / 0.001 = 36.009 ≈ 36  ✅
```

Notice that **dz/dy was evaluated at y = 6**, the value from the forward pass. Local derivatives always use the current values. That's why backpropagation needs the forward pass first.

### Rule 2: a longer chain, multiply all the links

```
x ──▶ a ──▶ b ──▶ c ──▶ L

dL/dx = dL/dc · dc/db · db/da · da/dx
```

However long the chain is, multiply every local derivative along it.

### Rule 3: several paths, add them

Sometimes an input reaches the output by **two or more routes**. Then you compute each path with Rule 1, and **add** the results.

```
        ┌──▶ a ──┐
x ──────┤        ├──▶ z
        └──▶ b ──┘

∂z/∂x = ∂z/∂a · ∂a/∂x  +  ∂z/∂b · ∂b/∂x
        └── path via a ──┘   └── path via b ──┘
```

**Example.** a = x + 1, b = 2x, z = a·b. Find dz/dx at x = 1.

**Step 1: forward values**

```
x = 1
a = 1 + 1 = 2
b = 2·1 = 2
z = 2·2 = 4
```

**Step 2: local derivatives**

```
∂a/∂x = 1          (a = x + 1)
∂b/∂x = 2          (b = 2x)
∂z/∂a = b = 2      (z = a·b: changing a changes z by b times as much)
∂z/∂b = a = 2      (and changing b changes z by a times as much)
```

**Step 3: each path, then add**

```
path via a:  ∂z/∂a · ∂a/∂x = 2 · 1 = 2
path via b:  ∂z/∂b · ∂b/∂x = 2 · 2 = 4

dz/dx = 2 + 4 = 6
```

**Check:** z = (x + 1)(2x) = 2x² + 2x, so dz/dx = 4x + 2 = 4·1 + 2 = 6. ✅

**Why add?** A small change in x changes **both** a and b. Each of those changes contributes to the change in z. To first order (that is, ignoring much smaller interaction terms that vanish as the nudge shrinks), these contributions add, which is why we add the path derivatives.

### Rule 4: go backwards and reuse

In a network, we want the derivative of the loss L with respect to **many** parameters. The trick is to start at L and walk **backwards**, reusing each result:

```
x ──▶ a ──▶ b ──▶ c ──▶ L

Step 1:  dL/dc                          (start at the end)
Step 2:  dL/db = dL/dc · dc/db          (reuse dL/dc)
Step 3:  dL/da = dL/db · db/da          (reuse dL/db)
Step 4:  dL/dx = dL/da · da/dx          (reuse dL/da)
```

Each link reuses a gradient that's already computed and multiplies it by the local derivative: ∂L/∂x = ∂L/∂y · ∂y/∂x, when y depends on x through that single connection. When **one operation has several inputs** (like z = a·b), each input gets its own gradient, calculated separately. When **several paths lead back to the same variable** (Rule 3), their contributions are added. So backpropagation isn't always a single chain of multiplications: it multiplies along each link and adds where paths merge. Without reuse, you'd recompute the same long products over and over for every parameter. **This reuse is what makes backpropagation fast**: it's the chain rule, computed from the output backwards.

## 5. Try it yourself: the chain rule inside a neural network

We use the network and the numbers from the [Backpropagation](backpropagation.md) guide: input x = (1, 0), correct answer y = 1.

### The forward values (already computed there)

```
a₁ = 0.6     h₁ = 0.6      (ReLU, a₁ > 0)
a₂ = 0.5     h₂ = 0.5      (ReLU, a₂ > 0)
z  = 0.04    p  = 0.510    L = −ln(p) ≈ 0.673

Weights used: w₁₁ = 0.5, w₁₂ = 0.3, v₁ = 0.4, v₂ = −0.6
```

### The computation graph

```
 w₁₁ ──▶ a₁ ──ReLU──▶ h₁ ──(×v₁)──┐
                                  ├──▶ z ──sigmoid──▶ p ──loss──▶ L
 w₁₂ ──▶ a₂ ──ReLU──▶ h₂ ──(×v₂)──┘
```

### The local derivatives

| Step | Formula | Local derivative | Value here |
|---|---|---|---|
| loss | L = −ln(p) (since y = 1) | ∂L/∂p = −1/p | −1/0.510 ≈ −1.961 |
| sigmoid | p = σ(z) | ∂p/∂z = p(1 − p) | 0.510 · 0.490 ≈ 0.2499 |
| output sum | z = v₁h₁ + v₂h₂ + c | ∂z/∂v₁ = h₁ | 0.6 |
| output sum | z = v₁h₁ + v₂h₂ + c | ∂z/∂h₁ = v₁ | 0.4 |
| ReLU | h₁ = max(0, a₁) | ∂h₁/∂a₁ = 1 if a₁ > 0 | 1 |
| hidden sum | a₁ = w₁₁x₁ + w₂₁x₂ + d₁ | ∂a₁/∂w₁₁ = x₁ | 1 |

### Example 1: a short chain, ∂L/∂v₁

Path: v₁ → z → p → L. Multiply the three local derivatives:

```
∂L/∂v₁ = ∂L/∂p · ∂p/∂z · ∂z/∂v₁
       = −1.961 · 0.2499 · 0.6
       = −0.490 · 0.6
       = −0.294
```

### Example 2: a long chain, ∂L/∂w₁₁

Path: w₁₁ → a₁ → h₁ → z → p → L. Multiply all five local derivatives:

```
∂L/∂w₁₁ = ∂L/∂p · ∂p/∂z · ∂z/∂h₁ · ∂h₁/∂a₁ · ∂a₁/∂w₁₁
        = −1.961 · 0.2499 · 0.4    · 1        · 1
        = −0.490 · 0.4 · 1 · 1
        = −0.196
```

**Notice the reuse (Rule 4).** Both examples start with ∂L/∂p · ∂p/∂z = −0.490. In the backpropagation guide this shared piece is called **δz**, and it's computed **once** and reused for every parameter. That's why the guide could write simply ∂L/∂v₁ = δz · h₁ and ∂L/∂w₁₁ = δz · v₁ · 1 · x₁.

### Example 3: two paths, ∂L/∂x₁

The input x₁ reaches the loss through **both** hidden neurons:

```
        ┌──(w₁₁)──▶ a₁ → h₁ ──(v₁)──┐
 x₁ ────┤                           ├──▶ z → p → L
        └──(w₁₂)──▶ a₂ → h₂ ──(v₂)──┘
```

So we use Rule 3: compute each path, then add.

**Path through h₁**

```
∂L/∂z · ∂z/∂h₁ · ∂h₁/∂a₁ · ∂a₁/∂x₁
= −0.490 · 0.4 · 1 · w₁₁
= −0.196 · 0.5
= −0.098
```

**Path through h₂**

```
∂L/∂z · ∂z/∂h₂ · ∂h₂/∂a₂ · ∂a₂/∂x₁
= −0.490 · (−0.6) · 1 · w₁₂
= 0.294 · 0.3
= 0.0882
```

**Add the paths**

```
∂L/∂x₁ = −0.098 + 0.0882 = −0.0098
```

The two paths almost cancel: one pushes the loss down, the other pushes it up.

**Check numerically:** nudging x₁ by a tiny amount (0.000001) and recomputing the loss gives a slope of −0.0098. ✅

**Why care about ∂L/∂x₁?** We can't change the input, but in a **deeper** network, x₁ would be the output of an **earlier layer**. ∂L/∂x₁ is exactly the error signal that gets passed back to that layer, and the chain continues from there.

### Check it in code

A **numerical gradient check** helps verify a chain-rule result: nudge one value slightly, recompute the loss, and divide the change in loss by the change in that value. The result is an approximation, so compare it with the analytical gradient using a reasonable step size. (It can also mislead at points where the function has a corner, such as ReLU at exactly 0.)

```python
import math

def loss(x1, w11=0.5, w12=0.3):
    # forward pass of the network from the backpropagation guide, with x2 = 0
    a1 = w11 * x1 + 0.1
    a2 = w12 * x1 + 0.2
    h1, h2 = max(0, a1), max(0, a2)
    z = 0.4 * h1 + (-0.6) * h2 + 0.1
    p = 1 / (1 + math.exp(-z))
    return -math.log(p)                 # y = 1

eps = 1e-6
print((loss(1, w11=0.5 + eps) - loss(1)) / eps)   # ≈ -0.196  (∂L/∂w11, Example 2)
print((loss(1 + eps) - loss(1)) / eps)            # ≈ -0.0098 (∂L/∂x1,  Example 3)
```

This **forward-difference** estimate uses a small but finite change, so it's approximate. For a smooth function, it approaches the true derivative as the step size shrinks, until the computer's floating-point rounding becomes significant.

## 6. Common mistakes

- **Forgetting a path.** If a value reaches the output in two ways, you must **add** both paths. Missing one gives a wrong gradient, like computing ∂L/∂x₁ = −0.098 instead of −0.0098 in Example 3.
- **Evaluating at the wrong point.** Local derivatives use the **current** forward values. For z = y², dz/dy = 2y must use y = 6 from the forward pass, not some other y.
- **Multiplying across paths instead of adding.** Within one path, **multiply**. Between paths, **add**.
- **Recomputing instead of reusing.** Correct but slow. Going backwards and reusing results (Rule 4) is the whole point of backpropagation.
- **Ignoring ReLU's derivative.** It's 1 when the input is positive and 0 when it's negative. (At exactly 0 it isn't defined in ordinary calculus; implementations choose a convention, commonly 0.) A zero derivative anywhere along a path makes **that path's contribution** zero. Other paths can still contribute to the total gradient.

## 7. Remember this

- **One chain → multiply** the local derivatives: dz/dx = dz/dy · dy/dx.
- **Several paths → add** the paths.
- Local derivatives are evaluated at the **forward-pass values**.
- **Backpropagation = the chain rule, computed from the output backwards, reusing each result.**

### Quick check

1. y = 2x + 1 and z = y³. What is dz/dx at x = 1?
2. 1 dollar = 0.9 euros and 1 euro = 160 yen, and both rates stay constant. If you have 0.01 more dollars, how many more yen do you get?
3. z = a·b with a = x and b = x. Using Rule 3, what is dz/dx? Check with z = x².
4. In Example 2, why does the factor ∂h₁/∂a₁ equal 1? What would ∂L/∂w₁₁ be if a₁ had been negative?

Try answering before reading on.

---

### Check your answers

1. Forward: y = 2·1 + 1 = 3. Local: dy/dx = 2, dz/dy = 3y² = 3·3² = 27. Multiply: dz/dx = 27 · 2 = **54**.
2. Chain: 0.01 · 0.9 · 160 = 0.01 · 144 = **1.44 more yen** (with constant rates).
3. Path via a: ∂z/∂a · ∂a/∂x = b · 1 = x. Path via b: ∂z/∂b · ∂b/∂x = a · 1 = x. Add: **dz/dx = 2x**. Check: z = x², dz/dx = 2x. ✅
4. Because a₁ = 0.6 > 0, and ReLU's derivative is 1 for positive inputs. If a₁ were negative, that factor would be 0, so the whole product, and ∂L/∂w₁₁, would be **0**.

## 8. Going further

- **[Backpropagation](backpropagation.md)**: the chain rule used to compute all 9 gradients and update the weights in one full training step.
- **[MLP Inference: XOR](mlp_xor_inference.md)**: the forward pass that provides the values the chain rule needs.

### Learn more

- 📖 [How the backpropagation algorithm works](https://neuralnetworksanddeeplearning.com/chap2.html) (Michael Nielsen, chapter 2): the chain rule applied to whole networks
- 📖 [micrograd README](https://github.com/karpathy/micrograd) (Andrej Karpathy): a tiny engine that applies the chain rule automatically over a computation graph
