# Mutual Information

> **Learning path:** [Shannon Entropy](shannon_entropy.md) → [Conditional Entropy](conditional_entropy.md) → **Mutual Information** → [Information Gain](information_gain.md) → [Cross-Entropy](cross_entropy.md) → [KL Divergence](kl_divergence.md) → [Bayesian Inference](bayesian_inference.md)

## In simple words

Mutual information measures how much two things **tell you about each other**.

In the conditional entropy guide, seeing an umbrella left 0.722 bits of the original 1 bit of uncertainty about rain. The clue removed 0.278 bits. That removed amount is the mutual information.

- Unrelated things, such as shoe size and favorite color: the mutual information is 0.
- One thing completely decides the other, such as a value and its exact copy: the mutual information equals the full uncertainty.

Analogy: two friends each keep a diary. If they spend every day together, reading one diary tells you most of what's in the other. That is high mutual information. If they have never met, reading one tells you nothing about the other. That is zero mutual information.

## What you need to know first

- Shannon entropy. See [Shannon Entropy](shannon_entropy.md).
- Conditional entropy and joint entropy, H(X | Y) and H(X, Y). See [Conditional Entropy](conditional_entropy.md).

## The key terms

| Symbol | Meaning |
|---|---|
| P(x, y) | One cell of the joint table |
| P(x), P(y) | The **marginals**: the row totals and the column totals |
| I(X; Y) | The mutual information between X and Y, in bits. The semicolon is just notation. |

## How it works

The main idea is **before minus after**:

I(X; Y) = H(X) − H(X | Y)

There are three ways to compute it, and they all give the same answer:

| Formula | Read it as |
|---|---|
| H(X) − H(X \| Y) | Uncertainty about X removed by learning Y |
| H(Y) − H(Y \| X) | Uncertainty about Y removed by learning X |
| H(X) + H(Y) − H(X, Y) | The overlap between the two |

The second row shows that mutual information is **symmetric**: X tells you exactly as much about Y as Y tells you about X.

### A picture of how the pieces fit

```
|<------ H(X) ------->|
|  H(X | Y)  | I(X;Y) |  H(Y | X)  |
             |<------ H(Y) ------->|
|<----------- H(X, Y) ------------>|
```

- H(X) is the left two pieces. H(Y) is the right two pieces.
- The middle piece, where they overlap, is the mutual information.
- The whole bar is the joint entropy H(X, Y).

This picture explains every formula above. For example, H(X) + H(Y) counts the middle piece twice, so subtracting the whole bar H(X, Y) leaves exactly one copy of the middle.

### Example: Rain and umbrella

Use the table from the conditional entropy guide:

| | Umbrella | No umbrella | Row total |
|---|---|---|---|
| **Rain** | 0.4 | 0.1 | 0.5 |
| **No rain** | 0.1 | 0.4 | 0.5 |
| **Column total** | 0.5 | 0.5 | 1 |

We already know H(Rain) = 1, H(Umbrella) = 1, H(Rain | Umbrella) = 0.722, and H(Rain, Umbrella) = 1.722.

- **Before minus after:** 1 − 0.722 = **0.278 bits**
- **Overlap:** 1 + 1 − 1.722 = **0.278 bits** ✓

**What this means:** seeing the umbrella removes 0.278 bits out of 1 bit, about a quarter of the uncertainty about rain.

### The two extremes

| Situation | Joint table | H(X) | H(Y) | H(X, Y) | I(X; Y) |
|---|---|---|---|---|---|
| Two fair coins flipped separately | Every cell is 0.25 | 1 | 1 | 2 | **0** |
| Y is an exact copy of X | 0.5 on the diagonal, 0 elsewhere | 1 | 1 | 1 | **1** |

- When the variables are independent, the bars don't overlap, so I = 0.
- When one is a copy of the other, they overlap completely, so I = H(X).

## Try it yourself

Use the table from the conditional entropy exercise:

| | Y = 0 | Y = 1 |
|---|---|---|
| **X = 0** | 0.3 | 0.2 |
| **X = 1** | 0.1 | 0.4 |

You already found H(X | Y) = 0.875 and H(X, Y) = 1.846, and you know H(Y) = 0.971.

1. Find H(X) from the row totals.
2. Compute I(X; Y) with "before minus after."
3. Check your answer with the overlap formula.
4. Which pair shares more information: this one, or rain and umbrella?

<details>
<summary>Answers</summary>

1. Row totals: P(X=0) = 0.5 and P(X=1) = 0.5, so H(X) = **1 bit**.
2. I = 1 − 0.875 = **0.125 bits**.
3. I = 1 + 0.971 − 1.846 = **0.125 bits** ✓
4. Rain and umbrella share more: 0.278 bits compared with 0.125 bits.

</details>

## Variations and common mistakes

- **Zero correlation does not mean zero mutual information.** Correlation only detects straight-line relationships. Example: X is −1, 0, or 1, each with probability 1/3, and Y = X². The correlation is 0. Yet Y is completely decided by X, so I(X; Y) = H(Y) = H(1/3, 2/3) = 0.918 bits.
- **It is not a correlation coefficient.** It has no sign, and it is not limited to the range −1 to 1. Its largest possible value is the smaller of H(X) and H(Y).
- **Using the wrong numbers for H(X, Y).** The joint entropy uses all the cells of the table, not the row or column totals.
- **Comparing across problems.** A larger number can just mean the variables have more uncertainty to share. One common fix divides by the smaller of H(X) and H(Y) to get a score from 0 to 1.

## Remember this

- Mutual information = before − after = H(X) − H(X | Y).
- It is symmetric, and it is 0 only when the variables are independent.
- Picture two overlapping bars: the overlap is the mutual information.

## Going further

- **Next:** [Information Gain](information_gain.md) uses mutual information to choose the best question to ask.
- **KL divergence view:** I(X; Y) = D_KL(P(X, Y) ‖ P(X)P(Y)). This measures how far the real table is from the table you would get if X and Y were independent. See [KL Divergence](kl_divergence.md).
- **Representation learning** uses mutual information to study what a model's internal features keep from its input.
