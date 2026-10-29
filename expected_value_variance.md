# Expected Value and Variance

## In simple words

When something is random, two numbers tell you most of what you need to know about it.

1. The **expected value** is where it lands **on average**.
2. The **variance** is how far it tends to **stray** from that average.

Think of two archers. Both aim at the same spot, and on average their arrows land right on it. That is the same expected value. But one archer's arrows land in a tight cluster, and the other's are scattered all over the target. That is a different variance.

Machine learning uses these two numbers everywhere. A training loss is an **average**. A minibatch gradient is a **noisy** guess, and its noise is a variance. Weight initialization and dropout are both tricks for keeping the average and the spread under control.

## What you need to know first

- A **probability** is a number from 0 to 1. All the probabilities of the possible outcomes add up to 1.
- A **weighted average**: multiply each value by its weight, then add.
- Squaring and square roots: 2.5² = 6.25 and √6.25 = 2.5.

## The key terms

| Term | Name | Die example |
|---|---|---|
| X | **Random variable**: a number decided by chance | The number a fair die shows: 1 to 6 |
| P(X = x) | **Probability** of one value | P(X = 3) = 1/6 |
| E[X] or μ | **Expected value** (mean): the long-run average | 3.5 |
| Var(X) or σ² | **Variance**: the average *squared* distance from the mean | 35/12 ≈ 2.917 |
| σ | **Standard deviation**: the square root of the variance, back in normal units | ≈ 1.708 |
| Cov(X, Y) | **Covariance**: whether two random numbers move together | 0 for two separate dice |
| Independent | Knowing one tells you nothing about the other | Two separate dice |

## How it works

We use one fair six-sided die the whole way through. Every outcome, 1 to 6, has probability 1/6.

### Step 1: Expected value

Multiply each value by its probability, then add:

E[X] = 1·(1/6) + 2·(1/6) + 3·(1/6) + 4·(1/6) + 5·(1/6) + 6·(1/6) = 21/6 = **3.5**

Two things to notice:

- **The expected value doesn't have to be possible.** No die ever shows 3.5.
- **It is a long-run average.** Roll many times and average the results. As the number of rolls grows, the average **tends** to get closer to 3.5. This is the **law of large numbers**. It describes a trend: any one batch of rolls can still miss.

Analogy: put a weight of 1/6 at each number from 1 to 6 on a ruler. The expected value is the point where the ruler **balances**.

### Step 2: The average of a function is not the function of the average

What is the average of the **square** of a roll? Square each value first, then average:

E[X²] = (1 + 4 + 9 + 16 + 25 + 36) / 6 = 91/6 ≈ **15.17**

That is **not** 3.5² = 12.25.

In general, E[g(X)] ≠ g(E[X]). They only match when g is a straight line, like 2X + 1. For a function that curves upward, like X², squaring first and averaging second always gives the bigger answer. This rule is called **Jensen's inequality**. The gap, 15.17 − 12.25 = 2.92, will turn out to be the variance.

### Step 3: Averages add up

Expected values follow three simple rules:

- **Scale and shift:** E[aX + b] = a·E[X] + b
- **Add:** E[X + Y] = E[X] + E[Y]
- The adding rule works **even when X and Y are connected**. You never need to check independence.

Examples:

- Roll two dice and add them: 3.5 + 3.5 = **7**. No need to list all 36 combinations.
- Double one roll and add 1: 2 × 3.5 + 1 = **8**.

**Counting trick.** A variable that is 1 when something happens and 0 when it doesn't is called an **indicator**. Its average is just the probability of the thing happening. If a classifier is right 90% of the time, the expected number correct on 100 examples is 100 × 0.9 = **90**, even if its mistakes are connected to each other. Test accuracy is exactly this: an average of indicators.

**Why this matters in ML.** The loss over a dataset is an average of per-example losses. Because averages add, the **gradient of the average is the average of the gradients**. That is what lets us compute gradients one example at a time and combine them.

### Step 4: Variance, the spread

The average says nothing about spread. A die and a coin that always shows 3.5 have the same mean but behave very differently.

To measure spread, look at how far each value is from the mean. Square those distances, then average them.

| x | x − 3.5 | (x − 3.5)² |
|---|---|---|
| 1 | −2.5 | 6.25 |
| 2 | −1.5 | 2.25 |
| 3 | −0.5 | 0.25 |
| 4 | 0.5 | 0.25 |
| 5 | 1.5 | 2.25 |
| 6 | 2.5 | 6.25 |
| **Sum** | **0** | **17.5** |

The plain distances add to **0**, because values above and below the mean cancel out. That is why we square them. Squaring makes every distance positive and also punishes big misses more than small ones.

Var(X) = 17.5 / 6 = 35/12 ≈ **2.917**

**A shortcut.** Variance also equals "the average of the square, minus the square of the average":

Var(X) = E[X²] − (E[X])² = 15.17 − 12.25 ≈ **2.917** ✓

This is the gap from step 2. It also shows that E[X²] can never be smaller than (E[X])², because variance can never be negative.

### Step 5: Standard deviation

Variance is in **squared units**. If X is in meters, its variance is in square meters, which is hard to picture. Take the square root to get back to normal units:

σ = √2.917 ≈ **1.708**

Roughly speaking, a typical roll lands about 1.7 away from 3.5.

### Step 6: Rules for variance

Variance follows **different** rules from expected value. This is where most mistakes happen.

| Change | Rule | Die example |
|---|---|---|
| Add a constant | Var(X + b) = Var(X) | Adding 10 to every roll: still 2.917 |
| Multiply by a constant | Var(aX) = **a²**·Var(X) | Doubling a roll: 4 × 2.917 = 11.667 |
| Add two **independent** variables | Var(X + Y) = Var(X) + Var(Y) | Two dice: 2.917 + 2.917 = 5.833 |
| Subtract two **independent** variables | Var(X − Y) = Var(X) + Var(Y) | Still 5.833, not 0 |
| Add two **connected** variables | Var(X + Y) = Var(X) + Var(Y) + 2·Cov(X, Y) | See step 7 |

Look closely at two dice versus one doubled die. Both have an average of 7. But two dice have variance **5.833**, and one doubled die has variance **11.667**. With two separate dice, a high roll on one is often balanced by a low roll on the other. With one doubled die, nothing balances it.

That balancing effect is why **averaging reduces noise**, which step 8 makes exact.

### Step 7: Covariance

**Covariance** says whether two random numbers tend to move together:

Cov(X, Y) = E[XY] − E[X]·E[Y]

- **Positive:** when one is high, the other tends to be high.
- **Negative:** when one is high, the other tends to be low.
- **Zero:** no straight-line tendency either way. Independent variables always have zero covariance.

Rearranged, the same formula gives E[XY] = E[X]·E[Y] + Cov(X, Y). So averages of products **only multiply when the covariance is zero**.

**Careful: zero covariance doesn't mean independent.** Let X be −1, 0 or 1, each with probability 1/3, and let Y = X². Y is completely decided by X. But E[XY] = E[X³] = 0 and E[X] = 0, so Cov(X, Y) = 0 − 0 = 0. Covariance only notices straight-line patterns, and this one is a U shape.

**Correlation** is covariance divided by both standard deviations. That puts it between −1 and 1, so it is easier to read.

### Step 8: The average of many samples

Take n independent rolls and average them. This sample average is random too, because different rolls give a different average. Using the rules above:

- **Its expected value is still μ.** On average, it hits the right answer. We call it **unbiased**.
- **Its variance is σ² / n.** The more samples, the less it wobbles.

For the die, averaging 4 rolls gives a variance of 2.917 / 4 ≈ 0.729. Averaging 16 rolls gives 2.917 / 16 ≈ 0.182.

The standard deviation of the average is σ / √n. This is called the **standard error**. Because of the square root, **halving the noise takes four times as many samples**.

**Why this matters in ML.** A minibatch gradient is the average of per-example gradients over a batch of size B. It is an unbiased guess of the full gradient, and its noise is roughly proportional to 1 / B. Going from batch size 64 to 256 costs four times as much per step but only halves the noise's standard deviation. (This is exact when examples are picked independently, with replacement. Picking without replacement makes the noise slightly smaller, but the difference is tiny when the batch is much smaller than the dataset.)

### Step 9: Estimating variance from data: divide by n − 1

To estimate the variance from samples, take each sample's distance from the **sample average**, square it, add them up, and divide by **n − 1**, not n.

Why? The samples sit a little closer to their own average than to the true mean, because that average was computed from the same samples. Dividing by n gives an answer that is too small on average, by a factor of (n − 1) / n. Dividing by n − 1 fixes it. This is called **Bessel's correction**.

Libraries don't agree on which one to use by default:

| Library | Function | Divides by |
|---|---|---|
| Python | `statistics.variance` | n − 1 |
| Python | `statistics.pvariance` | n |
| NumPy | `np.var` | n, unless you pass `ddof=1` |
| PyTorch | `torch.var` | n − 1, unless you pass `correction=0` |

For big samples the difference is tiny. For small samples it matters.

## Try it yourself

Use pen and paper first, then check with the Python code below.

1. **A loaded die.** A die shows 6 with probability 1/2, and each other face with probability 1/10. Find its expected value and variance.
2. **Difference of two dice.** You roll two dice and subtract: X − Y. What are its mean and variance?
3. **Connected dice.** Roll two separate dice, X and Z, and let Y = X + Z. Find Cov(X, Y) and Var(X + Y).
4. **Dropout.** Dropout keeps an activation x = 2 with probability 0.5 and divides it by 0.5, so it becomes 4. Otherwise it sets it to 0. What are the output's mean and variance?
5. **Bessel's correction.** You take many samples of 3 die rolls each, and for each sample you compute the variance by dividing by n. What do those estimates average to?

<details>
<summary>Answers</summary>

**1.** E[X] = 6 × 0.5 + (1 + 2 + 3 + 4 + 5) × 0.1 = 3 + 1.5 = **4.5**.
E[X²] = 36 × 0.5 + (1 + 4 + 9 + 16 + 25) × 0.1 = 18 + 5.5 = 23.5.
Var(X) = 23.5 − 4.5² = 23.5 − 20.25 = **3.25**.

**2.** Mean: 3.5 − 3.5 = **0**. Variance: the dice are independent, so the variances add: 2.917 + 2.917 = **5.833**. Subtracting adds noise, just like adding does.

**3.** Cov(X, Y) = Cov(X, X) + Cov(X, Z) = Var(X) + 0 = **2.917**.
Var(Y) = 2.917 + 2.917 = 5.833.
Var(X + Y) = 2.917 + 5.833 + 2 × 2.917 = **14.58**.
Check it another way: X + Y = 2X + Z, and Var(2X + Z) = 4 × 2.917 + 2.917 = 14.58 ✓. Forgetting the covariance term would give 8.75, which is wrong.

**4.** Mean: 0.5 × 4 + 0.5 × 0 = **2**, the same as the input. That is the whole point of dividing by 0.5.
E[output²] = 0.5 × 16 + 0.5 × 0 = 8, so Var = 8 − 2² = **4**.

**5.** About (n − 1)/n × 2.917 = 2/3 × 2.917 ≈ **1.94**. That is too small. Dividing by n − 1 instead averages to about 2.917.

</details>

### Check it in Python

This script needs only the standard library. Save it as `expected_value_variance.py` and run `python expected_value_variance.py`.

```python
import random
import statistics

random.seed(0)

# 1. Exact expected value and variance of a fair die
faces = [1, 2, 3, 4, 5, 6]
p = 1 / 6
mean = sum(x * p for x in faces)
mean_sq = sum(x * x * p for x in faces)
var = mean_sq - mean ** 2
print(f"die: E[X] = {mean:.4f}, E[X^2] = {mean_sq:.4f}, Var(X) = {var:.4f}, SD = {var ** 0.5:.4f}")

# 2. Simulate: the average of many rolls approaches E[X]
for n in [10, 1_000, 100_000]:
    rolls = [random.randint(1, 6) for _ in range(n)]
    print(f"n = {n:>6}: sample mean = {statistics.fmean(rolls):.4f}, "
          f"sample variance = {statistics.variance(rolls):.4f}")

# 3. Sum of two dice: means add, and (because independent) variances add
sums = [random.randint(1, 6) + random.randint(1, 6) for _ in range(100_000)]
print(f"two dice: mean = {statistics.fmean(sums):.4f} (exact 7), "
      f"variance = {statistics.variance(sums):.4f} (exact {2 * var:.4f})")

# 4. Variance of a sample mean shrinks like sigma^2 / n
for n in [1, 4, 16, 64]:
    means = [statistics.fmean(random.randint(1, 6) for _ in range(n)) for _ in range(20_000)]
    print(f"batch size {n:>2}: Var(mean) = {statistics.variance(means):.4f}, "
          f"predicted sigma^2/n = {var / n:.4f}")

# 5. Dropout with rescaling keeps the expected value
x, drop = 2.0, 0.5
keep = 1 - drop
outs = [(x / keep if random.random() < keep else 0.0) for _ in range(100_000)]
print(f"dropout: mean = {statistics.fmean(outs):.4f} (input {x}), "
      f"variance = {statistics.variance(outs):.4f} (predicted {x * x * drop / keep:.4f})")

# 6. Two common distributions: Bernoulli(p) and continuous Uniform(0, 1)
b = [1 if random.random() < 0.3 else 0 for _ in range(100_000)]
print(f"Bernoulli(0.3): mean = {statistics.fmean(b):.4f} (exact 0.3), "
      f"variance = {statistics.variance(b):.4f} (exact {0.3 * 0.7:.4f})")
u = [random.random() for _ in range(100_000)]
print(f"Uniform(0, 1): mean = {statistics.fmean(u):.4f} (exact 0.5), "
      f"variance = {statistics.variance(u):.4f} (exact {1 / 12:.4f})")
```

Output (Python 3, seed 0):

```text
die: E[X] = 3.5000, E[X^2] = 15.1667, Var(X) = 2.9167, SD = 1.7078
n =     10: sample mean = 3.5000, sample variance = 1.1667
n =   1000: sample mean = 3.4450, sample variance = 3.0440
n = 100000: sample mean = 3.4894, sample variance = 2.9128
two dice: mean = 6.9916 (exact 7), variance = 5.8103 (exact 5.8333)
batch size  1: Var(mean) = 2.9114, predicted sigma^2/n = 2.9167
batch size  4: Var(mean) = 0.7265, predicted sigma^2/n = 0.7292
batch size 16: Var(mean) = 0.1807, predicted sigma^2/n = 0.1823
batch size 64: Var(mean) = 0.0449, predicted sigma^2/n = 0.0456
dropout: mean = 2.0005 (input 2.0), variance = 4.0000 (predicted 4.0000)
Bernoulli(0.3): mean = 0.3014 (exact 0.3), variance = 0.2106 (exact 0.2100)
Uniform(0, 1): mean = 0.4988 (exact 0.5), variance = 0.0837 (exact 0.0833)
```

Your numbers may differ slightly on another Python version. Things to notice:

- **10 rolls happen to average exactly 3.5, but their variance (1.17) is far from 2.917.** A small sample can look right on one number by luck. The 1,000-roll average (3.445) is actually further from 3.5. Only the overall trend improves.
- **Each time the batch grows 4 times bigger, the variance of the average drops about 4 times,** just as σ² / n predicts.
- **The dropout mean matches the input,** and the variance is 4, matching answer 4 above.

## Variations and common mistakes

### Common distributions

A few random variables show up again and again. Their means and variances are worth knowing.

| Distribution | What it does | Mean | Variance | Where it shows up |
|---|---|---|---|---|
| **Bernoulli(p)** | 1 with probability p, else 0 | p | p(1 − p) | Dropout masks, yes/no labels, "was this prediction right?" |
| **Fair die** | 1 to 6, equally likely | 3.5 | 35/12 | This guide |
| **Uniform(a, b)** | Any number between a and b, all equally likely | (a + b) / 2 | (b − a)² / 12 | Uniform weight initialization |
| **Normal(μ, σ²)** | The bell curve | μ | σ² | Gaussian weight initialization, noise |

For the Bernoulli: X² = X when X is only ever 0 or 1, so E[X²] = p, and Var = p − p² = p(1 − p). It is largest at p = 0.5, the most uncertain coin.

For Uniform(−a, a): the mean is 0, and the variance is (2a)² / 12 = a² / 3. So to get a target variance v, pick a = √(3v).

For continuous variables like these, the sums become integrals, but every rule in this guide still holds.

### Common mistakes

- **Assuming E[g(X)] = g(E[X]).** Only true for straight lines. The average of the squares is not the square of the average.
- **Assuming E[XY] = E[X]·E[Y].** Only true when the covariance is zero. Independence is enough.
- **Writing Var(aX) = a·Var(X).** The scale comes out **squared**: a².
- **Writing Var(X − Y) = Var(X) − Var(Y).** For independent variables, the variances **add**.
- **Adding variances of connected variables.** Without independence, you need the 2·Cov term.
- **Mixing up E[X²] and Var(X).** They are only equal when the mean is 0. This difference is exactly why ReLU networks need a different initialization (see below).
- **Mixing up variance and standard deviation.** They are in different units. Error bars use standard deviation.
- **Mixing up n and n − 1.** Check what your library does.

## Where it shows up in neural networks

### Mean squared error: the best single guess is the mean

Suppose you must guess a random target Y with one fixed number c, and you are scored by squared error. The expected score splits into two parts:

E[(Y − c)²] = Var(Y) + (E[Y] − c)²

The first part doesn't depend on your guess. The second is zero when c = E[Y]. So **the mean is the best single guess under squared error**, and the variance is the error no fixed guess can remove. Applied to each input separately, this is why a model trained with MSE learns to predict the **average** target for that input. The same split is the starting point of the **bias–variance decomposition**.

### Dropout keeps the average the same

During training, dropout zeros each activation with probability p and keeps it with probability q = 1 − p. Kept activations are divided by q. For an activation x:

- Mean: q × (x / q) + p × 0 = **x**, the same as with dropout turned off.
- Variance: x² × p / q. With p = 0.5, that is x², which is a lot of noise. That noise is part of how dropout prevents overfitting.

The average **activation** is preserved, but by step 2, the average of anything curved computed from it later is not. So training and inference still behave a little differently.

### Weight initialization keeps the variance steady

A neuron adds up n inputs, each times a weight: z = w₁x₁ + w₂x₂ + … + wₙxₙ. Suppose the weights have mean 0 and variance Var(w), and everything is independent.

**One product.** Because the weight's mean is 0, Var(w·x) = Var(w) × E[x²]. Notice the input appears as **E[x²]**, not Var(x).

**The whole sum.** The n products are independent, so their variances add:

Var(z) = n × Var(w) × E[x²]

If n × Var(w) is bigger than 1, the numbers grow at every layer and blow up. If it is smaller than 1, they shrink toward zero. Either way, a deep network stops training well.

| Activation | Inputs have mean 0? | Rule that keeps it steady | Name |
|---|---|---|---|
| None or tanh | Yes, so E[x²] = Var(x) | Var(w) = 1 / n | **Xavier** (Glorot) |
| ReLU | No: ReLU is never negative | Var(w) = 2 / n | **He** (Kaiming) |

**Why ReLU needs 2 / n.** ReLU keeps the positive half of its input and zeros the negative half. If the input is symmetric around 0, each half contributes the same amount to its average square, so ReLU cuts E[x²] in **half**. Doubling the weight variance makes up for it.

Xavier initialization actually uses 2 / (n_in + n_out). That is a compromise between keeping the forward pass steady, which needs 1 / n_in, and keeping the backward pass steady, which needs 1 / n_out.

You can watch this happen. In a stack of 20 random layers of width 256, a quick NumPy run gave these averages of the squared activations at the end:

| Weight variance | No activation | ReLU after each layer |
|---|---|---|
| 0.5 / n | about 0.000001 (vanished) | about 0 (vanished) |
| 1 / n | **about 1.2 (steady)** | about 0 (vanished) |
| 2 / n | about 1,000,000 (exploded) | **about 0.9 (steady)** |

These rules rest on the independence and symmetry assumptions above. They are a good starting point at the beginning of training, not a guarantee for every network.

## Remember this

- **Expected value** = the probability-weighted average = the balance point.
- **Variance** = the average squared distance from the mean = E[X²] − (E[X])².
- **Standard deviation** = √variance, back in normal units.
- Averages **always add**. Variances only add when the variables are **independent**.
- Scaling by a multiplies the variance by **a²**.
- The average of n samples has variance **σ² / n**. Halving the noise takes 4 times the samples.
- E[g(X)] ≠ g(E[X]) unless g is a straight line.

## Going further

- **Bayesian inference.** A posterior is a whole distribution, and its mean and variance summarize your updated belief and how sure you are. See [Bayesian Inference](bayesian_inference.md).
- **Cross-entropy and KL divergence.** Both are expected values: the average surprise, weighted by the true probabilities. See [Cross-Entropy](cross_entropy.md) and [KL Divergence](kl_divergence.md).
- **Backpropagation.** The gradients it computes for a minibatch are the noisy averages from step 8. See [Backpropagation](backpropagation.md).
- **Normalization layers.** Batch norm and layer norm estimate a mean and variance from data, then rescale activations to mean 0 and variance 1.
