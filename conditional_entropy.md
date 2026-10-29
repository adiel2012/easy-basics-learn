# Conditional Entropy

## In simple words

Conditional entropy is the average uncertainty that **remains** about one thing after you learn another. It is entropy with a condition attached.

Example: today's weather tells you something about tomorrow's. The uncertainty about tomorrow's weather, after you know today's, is the conditional entropy H(tomorrow | today). It is usually smaller than the uncertainty without knowing today.

## What you need to know first

- Shannon entropy. See [Shannon Entropy](shannon_entropy.md).
- Joint probabilities P(x, y) and conditional probabilities P(x | y) = P(x, y) / P(y).

## The key terms

| Symbol | Meaning |
|---|---|
| H(X \| Y) | Average uncertainty about X after learning Y |
| H(X, Y) | Uncertainty about both X and Y together |
| H(Y) | Uncertainty about Y alone |

## How it works

The formula is:

H(X | Y) = Σ_y P(y) · H(X | Y = y)

This means: for each value of y, compute the entropy of X in that case, then take the weighted average.

An equivalent and often faster formula:

H(X | Y) = H(X, Y) − H(Y)

Important facts:

- 0 ≤ H(X | Y) ≤ H(X). On average, learning Y does not increase uncertainty.
- If X and Y are independent, H(X | Y) = H(X).
- If Y completely determines X, H(X | Y) = 0.

Steps:

1. Write the joint table.
2. For each value of Y, find the conditional probabilities P(x | y) and their entropy.
3. Weight each entropy by P(y) and add them up.

### Example: Rain and umbrella

Using the table from the mutual information guide:

| | Umbrella | No umbrella | Total |
|---|---|---|---|
| Rain | 0.4 | 0.1 | 0.5 |
| No rain | 0.1 | 0.4 | 0.5 |

- **Given an umbrella** (P = 0.5): P(rain) = 0.4 / 0.5 = 0.8, so H = 0.722 bits.
- **Given no umbrella** (P = 0.5): P(rain) = 0.1 / 0.5 = 0.2, so H = 0.722 bits.

H(Rain | Umbrella) = 0.5 × 0.722 + 0.5 × 0.722 = **0.722 bits**

Check: H(X, Y) − H(Y) = 1.722 − 1 = 0.722. ✓

Without the umbrella information, H(Rain) = 1 bit. Knowing the umbrella reduces the uncertainty by 0.278 bits.

## Try it yourself

Joint probabilities: P(X=0, Y=0) = 0.3, P(X=0, Y=1) = 0.2, P(X=1, Y=0) = 0.1, P(X=1, Y=1) = 0.4.

Compute H(X | Y).

<details>
<summary>Answer</summary>

1. Marginals: P(Y=0) = 0.4 and P(Y=1) = 0.6.
2. Given Y = 0: P(X=0 | Y=0) = 0.3/0.4 = 0.75, so H = 0.811 bits.
3. Given Y = 1: P(X=0 | Y=1) = 0.2/0.6 = 1/3, so H = 0.918 bits.
4. H(X | Y) = 0.4 × 0.811 + 0.6 × 0.918 = **0.875 bits**.

Check: H(X, Y) = 0.521 + 0.464 + 0.332 + 0.529 = 1.846 and H(Y) = 0.971. Then 1.846 − 0.971 = 0.875. ✓

</details>

## Variations and common mistakes

- **Forgetting the weights.** Each conditional entropy must be weighted by P(y).
- **Not symmetric.** H(X | Y) is generally different from H(Y | X). They are related by H(X | Y) − H(Y | X) = H(X) − H(Y).
- **Thinking conditioning always reduces uncertainty.** On average it does not increase it. For one specific value of y, the uncertainty can go up.
- **Chain rule.** H(X, Y) = H(Y) + H(X | Y). A language model uses this idea when it adds the uncertainty of each new token. See [LLM Inference](llm_inference.md).

## Remember this

- Conditional entropy = uncertainty left after learning Y.
- H(X | Y) = H(X, Y) − H(Y).
- On average it is never larger than H(X).

## Going further

- **Mutual information** is the uncertainty removed by conditioning: I(X; Y) = H(X) − H(X | Y). See [Mutual Information](mutual_information.md).
- **Information gain** uses this same subtraction to choose decision tree splits. See [Information Gain](information_gain.md).
