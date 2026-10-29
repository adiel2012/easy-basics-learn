# Mutual Information

## In simple words

Mutual information measures how much knowing one thing tells you about another. If they are unrelated, the answer is zero. If knowing one completely determines the other, the answer is as large as the uncertainty itself.

Example: knowing that it is raining tells you a lot about whether someone carries an umbrella. Knowing someone's shoe size tells you almost nothing about their favorite color.

## What you need to know first

- Shannon entropy. See [Shannon Entropy](shannon_entropy.md).
- Conditional entropy. See [Conditional Entropy](conditional_entropy.md).
- Joint probability tables, where each cell is P(x, y).

## The key terms

| Symbol | Meaning |
|---|---|
| P(x, y) | The joint probability that X = x and Y = y |
| P(x), P(y) | The marginal probabilities: the row and column totals |
| I(X; Y) | The mutual information, in bits |

## How it works

Three equivalent formulas:

I(X; Y) = H(X) − H(X | Y)
I(X; Y) = H(X) + H(Y) − H(X, Y)
I(X; Y) = I(Y; X)

The mutual information is **symmetric**: knowing X tells you as much about Y as knowing Y tells you about X.

Steps using a joint table:

1. Add the rows and columns to get H(X) and H(Y).
2. Compute H(X, Y) from the four joint probabilities.
3. Compute I = H(X) + H(Y) − H(X, Y).

### Example: Rain and umbrella

| | Umbrella | No umbrella | Total |
|---|---|---|---|
| Rain | 0.4 | 0.1 | 0.5 |
| No rain | 0.1 | 0.4 | 0.5 |
| **Total** | 0.5 | 0.5 | 1 |

- H(Rain) = 1 bit and H(Umbrella) = 1 bit, because each is split 50/50.
- H(X, Y) uses the four values 0.4, 0.1, 0.1, 0.4:
  - 0.4 contributes 0.529 each, twice: 1.058
  - 0.1 contributes 0.332 each, twice: 0.664
  - H(X, Y) = 1.722 bits

I = 1 + 1 − 1.722 = **0.278 bits**

Check with the other formula: given an umbrella, P(rain) = 0.8, so H = 0.722 bits. Then I = 1 − 0.722 = 0.278. ✓

### The zero case

If X and Y are independent, every joint probability equals the product of the marginals, and I = 0. For two fair coins, each joint probability is 1/4, so H(X, Y) = 2 bits, and I = 1 + 1 − 2 = 0.

## Try it yourself

1. A perfect copy: P(0, 0) = 0.5, P(1, 1) = 0.5, and the other two cells are 0. Compute I(X; Y).
2. Independent: all four joint probabilities are 0.25. Compute I(X; Y).

<details>
<summary>Answers</summary>

1. H(X) = 1, H(Y) = 1, and H(X, Y) = 1 (two outcomes, each 0.5). So I = 1 + 1 − 1 = **1 bit**. Knowing X tells you Y completely.
2. H(X) = 1, H(Y) = 1, and H(X, Y) = 2. So I = **0 bits**. The variables tell you nothing about each other.

</details>

## Variations and common mistakes

- **Zero correlation is not always zero mutual information.** Correlation measures only straight-line relationships. Mutual information detects any dependence, including curved ones.
- **Mutual information is not a correlation coefficient.** It is not limited to −1 to 1, and it has no sign.
- **Mixing up the formulas.** Use the joint probabilities for H(X, Y), not the marginals.
- **Scale.** The value depends on the entropy of the variables, so it is hard to compare across very different problems without normalizing.

## Remember this

- Mutual information = how much knowing one variable reduces uncertainty about the other.
- It is 0 for independent variables, and symmetric in X and Y.
- Information gain is a mutual information.

## Going further

- **Information gain** uses mutual information to choose decision tree splits. See [Information Gain](information_gain.md).
- **KL divergence:** I(X; Y) = D_KL(P(X, Y) ‖ P(X)P(Y)). See [KL Divergence](kl_divergence.md).
- **Representation learning** uses mutual information to study what a model's internal features keep from the input.
