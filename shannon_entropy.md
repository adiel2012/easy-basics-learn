# Shannon Entropy

> **Learning path:** **Shannon Entropy** → [Conditional Entropy](conditional_entropy.md) → [Mutual Information](mutual_information.md) → [Information Gain](information_gain.md) → [Cross-Entropy](cross_entropy.md) → [KL Divergence](kl_divergence.md) → [Bayesian Inference](bayesian_inference.md)

Shannon entropy is one number that measures how uncertain a random process is, on average. Claude Shannon introduced it in 1948, and it is the foundation of information theory.

Strictly speaking, entropy describes a whole probability distribution. **Surprise** describes one particular outcome. Entropy is the average surprise over all outcomes.

## In simple words

Imagine a friend tells you the result of something random. How much new information do you get?

- If the result is always the same (a coin that always lands heads), you learn nothing new. There is zero uncertainty.
- If every result is equally likely (a fair coin), you cannot guess well. There is maximum uncertainty.

Entropy is the number that captures this. The more even and unpredictable the outcomes, the higher the entropy. The more one outcome dominates, the lower it is.

A good analogy is the game "20 Questions." If the possibilities are split evenly, you need many yes/no questions to find the answer. If one answer is almost certain, you need almost none. Entropy measures the average information needed to identify an outcome. In an idealized binary-question scheme, it provides a lower bound on the average number of yes/no answers needed. The actual number may be larger.

## What you need to know first

- **Probability**: a number from 0 to 1 for how likely something is. The probabilities of all possible outcomes add up to 1.
- **Logarithm base 2**: log₂(x) asks "2 to what power gives x?" For example, log₂(2) = 1, log₂(4) = 2, log₂(8) = 3, and log₂(1) = 0. For numbers between 0 and 1 the answer is negative: log₂(1/2) = −1.
- **Bit**: the unit of information. One bit answers one yes/no question.

## The key terms

| Symbol | Meaning |
|---|---|
| X | A random variable: the thing that produces outcomes (a coin flip, a letter in a text) |
| x | One possible outcome |
| p(x) | The probability of outcome x |
| Σ | "Add up over all the outcomes" |
| H(X) | The entropy of X, measured in bits |

## How it works

Define the surprise of one outcome x as:

I(x) = −log₂ p(x)

Entropy is the expected (average) surprise:

H(X) = E[I(X)] = −Σ p(x) · log₂ p(x)

A rare outcome can have a high surprise, even when the whole distribution has low entropy. For example, if heads has probability 0.99 and tails has probability 0.01, tails has a surprise of about 6.644 bits. Yet the entropy of the flip is only about 0.081 bits.

For each outcome, do three steps:

1. Take its probability, p(x).
2. Compute −log₂ p(x). This is the **surprise** of that outcome. Rare outcomes give a big surprise. A certain outcome (p = 1) gives a surprise of 0.
3. Multiply the surprise by how often it happens, p(x).

Add up the results. Entropy is the **average surprise**.

### Example 1: A fair coin

There are two outcomes, each with p = 1/2.

- Surprise of each outcome: −log₂(1/2) = 1 bit
- H = (1/2)(1) + (1/2)(1) = **1 bit**

One flip of a fair coin gives exactly one bit of uncertainty: one yes/no question's worth.

### Example 2: A biased coin (heads 90% of the time)

| Outcome | p | Surprise −log₂ p | p × surprise |
|---|---|---|---|
| Heads | 0.9 | 0.152 | 0.137 |
| Tails | 0.1 | 3.322 | 0.332 |
| **Total** | | | **H ≈ 0.47 bits** |

This is less than half a bit. You are rarely surprised, so each flip teaches you less.

### Example 3: A fair six-sided die

There are six outcomes, each with p = 1/6. Each surprise is log₂ 6 ≈ 2.585 bits, and the average is the same.

- H = log₂ 6 ≈ **2.585 bits**

That is more than a coin, because there are more equally likely outcomes.

### Example 4: Three letters with different chances

A source produces the letters A, B, and C with probabilities 1/2, 1/4, and 1/4.

| Letter | p | Surprise | p × surprise |
|---|---|---|---|
| A | 1/2 | 1 | 0.5 |
| B | 1/4 | 2 | 0.5 |
| C | 1/4 | 2 | 0.5 |
| **Total** | | | **H = 1.5 bits** |

Now check what this number means with a code. Encode A as `0`, B as `10`, and C as `11`.

- A takes 1 bit and appears half the time.
- B and C take 2 bits each and together appear half the time.
- Average length = (1/2)(1) + (1/2)(2) = **1.5 bits per letter**

This matches the entropy exactly. For binary prefix codes that encode one letter at a time, this code achieves the minimum possible average length of 1.5 bits per letter.

## Calculating it by hand

You need three things: a table of probabilities, a way to find each surprise (log₂), and some multiplying and adding. No calculator is needed for most problems.

### Step 1: Know the surprise of a fraction

For probabilities that are powers of 1/2, the surprise is the number of times you **halve 1** to reach that probability. Halving once gives 1/2, twice gives 1/4, three times gives 1/8. So the surprise of 1/2 is 1 bit, the surprise of 1/4 is 2 bits, and so on. Other probabilities have fractional surprise values. You can calculate them with logarithms, as shown in Step 2.

Memorize these:

| p | Surprise −log₂ p (bits) |
|---|---|
| 1/2 = 0.5 | 1 |
| 1/4 = 0.25 | 2 |
| 1/8 = 0.125 | 3 |
| 1/16 = 0.0625 | 4 |

Some other useful values:

| p | Surprise (bits) |
|---|---|
| 3/4 = 0.75 | 0.415 |
| 1/3 ≈ 0.333 | 1.585 |
| 1/5 = 0.2 | 2.322 |
| 1/6 ≈ 0.167 | 2.585 |
| 1/10 = 0.1 | 3.322 |

### Step 2: Split hard fractions into easy logs

Any fraction a/b has surprise:

−log₂(a/b) = log₂ b − log₂ a

So you only need the logs of whole numbers. For example, the surprise of 3/10 is:

log₂ 10 − log₂ 3 = 3.322 − 1.585 = **1.737 bits**

Here is a short list of whole-number logs to know:

| n | log₂ n |
|---|---|
| 2 | 1 |
| 3 | 1.585 |
| 4 | 2 |
| 5 | 2.322 |
| 6 | 2.585 |
| 8 | 3 |
| 10 | 3.322 |

**Tip:** If your numbers are decimals, write them as fractions first. 0.3 is 3/10 and 0.2 is 1/5.

If you have a regular calculator with a `ln` button, use log₂ x = ln(x) ÷ ln(2), where ln(2) ≈ 0.693.

### Step 3: Build a table and fill it in

For every outcome, fill in four boxes:

1. **Outcome**: its name
2. **p**: its probability
3. **Surprise**: −log₂ p, from Step 1 or Step 2
4. **p × surprise**: multiply the two boxes above

Then add the last column. The total is the entropy H.

### Worked example: p = (1/2, 3/10, 1/5)

Check first that the probabilities add up: 1/2 + 3/10 + 1/5 = 5/10 + 3/10 + 2/10 = 1. ✓

| Outcome | p | Surprise | p × surprise |
|---|---|---|---|
| A | 1/2 | 1 | 0.500 |
| B | 3/10 | 1.737 | 0.521 |
| C | 1/5 | 2.322 | 0.464 |
| **Total** | | | **H ≈ 1.49 bits** |

Notice that the table works exactly as before. You are just doing more multiplications.

### Step 4: Check your answer

Two quick checks catch most mistakes:

- **The answer cannot be negative.** If you get a negative number, you probably forgot the minus sign in Step 1.
- **The answer cannot be bigger than log₂ n**, where n is the number of outcomes. With 3 outcomes, the maximum is log₂ 3 = 1.585. Our answer, 1.49, is below that. ✓

The maximum is reached only when all outcomes are equally likely.

### Practice by hand

Try these before looking at the answers.

1. A fair coin: p = (1/2, 1/2).
2. A biased coin: p = (3/4, 1/4).
3. Three equally likely outcomes: p = (1/3, 1/3, 1/3).

<details>
<summary>Answers</summary>

1. Each surprise is 1, each product is 0.5, and the total is **1 bit**.
2. Surprises are 0.415 and 2. Products are 0.75 × 0.415 = 0.311 and 0.25 × 2 = 0.5. Total: **H ≈ 0.811 bits**.
3. Each surprise is 1.585, each product is (1/3) × 1.585 = 0.528, and the three products add to **H ≈ 1.585 bits**. This equals log₂ 3, the maximum for three outcomes.

</details>

## Try it yourself

A loaded four-sided die has these probabilities:

| Face | 1 | 2 | 3 | 4 |
|---|---|---|---|---|
| p | 1/2 | 1/4 | 1/8 | 1/8 |

Work out the surprise of each face, multiply by its probability, and add the results.

<details>
<summary>Check your answer</summary>

- Face 1: 1 bit × 1/2 = 0.5
- Face 2: 2 bits × 1/4 = 0.5
- Face 3: 3 bits × 1/8 = 0.375
- Face 4: 3 bits × 1/8 = 0.375
- **Total: H = 1.75 bits**

</details>

## Variations

- **Base of the logarithm.** Base 2 gives bits. The natural logarithm (base e) gives *nats*, which is common in machine learning formulas. Only the unit changes.
- **Binary entropy.** For a yes/no event with probability p, H(p) = −p log₂ p − (1 − p) log₂(1 − p). It is highest, 1 bit, when p = 0.5.
- **Entropy of real data.** For a dataset, you can estimate entropy by treating observed frequencies as probabilities. For the text `aaab`, the empirical probabilities are p(a) = 3/4 and p(b) = 1/4, giving an empirical entropy of approximately 0.811 bits per letter. Observed frequencies are only estimates of the true probabilities, and small samples can be misleading.

**Common mistakes:**

- **Thinking entropy is just the number of outcomes.** It depends on how evenly the probabilities are spread. A 10-sided die where one face has 99% probability has low entropy.
- **Forgetting the minus sign.** Logs of numbers below 1 are negative. The minus sign makes the total positive.
- **Using 0 × log 0.** Treat this term as 0. An outcome that never happens adds nothing.
- **Mixing bits and nats.** Always check which logarithm base a formula uses.

## Remember this

- Entropy = average surprise. It is the ideal average number of bits per outcome.
- A certain outcome has entropy 0. An evenly split outcome has the highest entropy.
- Fair coin: 1 bit. Fair die: log₂ 6 ≈ 2.585 bits.
- For a source that produces independent outcomes with the same known probability distribution, entropy is the lower bound on the expected number of bits per outcome for lossless compression. With suitable coding of long sequences, the average number of bits per outcome can approach this bound. A code for single outcomes may need more bits on average.

## Going further

- **Data compression.** Huffman codes and ZIP-style compressors try to get close to the entropy limit.
- **Next:** [Conditional Entropy](conditional_entropy.md) asks how much uncertainty is left after you learn a clue.
- **Cross-entropy loss.** Machine learning models compare predicted probabilities with the true answer using a close relative of this formula. See [Cross-Entropy](cross_entropy.md).
- **Tokenization.** Entropy helps explain why some text is easier to predict than other text. See [Tokenization](tokenization.md).
- **Thermodynamics.** Gibbs entropy has the same mathematical form, with physical units added.
