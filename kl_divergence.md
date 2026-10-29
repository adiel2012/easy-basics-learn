# KL Divergence

## In simple words

KL divergence measures the **extra cost** of using a guessed probability distribution Q when the true distribution is P. It is measured in bits.

- If Q matches P exactly, the extra cost is zero.
- If Q is wrong, the extra cost is positive.

It is never negative. It is also **not a true distance**, because it is not symmetric: the cost of guessing Q when the truth is P is not the same as guessing P when the truth is Q.

Analogy: you pack for a sunny trip (Q), but it rains (P). The extra trouble of packing the wrong way is the KL divergence.

## What you need to know first

- Shannon entropy and cross-entropy. See [Shannon Entropy](shannon_entropy.md) and [Cross-Entropy](cross_entropy.md).
- Logarithms base 2, and how to compute a ratio inside a log.
- Probabilities add to 1.

## The key terms

| Symbol | Meaning |
|---|---|
| P(x) | The true probability of outcome x |
| Q(x) | The guessed probability of outcome x |
| D_KL(P ‖ Q) | The KL divergence from Q to P, in bits |

## How it works

The formula is:

D_KL(P ‖ Q) = Σ P(x) · log₂ ( P(x) / Q(x) )

Each term asks: how much more does the truth weight this outcome than the guess does? The term is weighted by the true probability P(x).

There is a useful identity:

D_KL(P ‖ Q) = H(P, Q) − H(P)

So KL divergence is cross-entropy minus the unavoidable uncertainty. Other key facts:

- It is always **≥ 0**.
- It is **0 only when P = Q**.
- It is **not symmetric**: D_KL(P ‖ Q) ≠ D_KL(Q ‖ P) in general.

### Example 1: The same numbers as the cross-entropy guide

P = (1/2, 1/2), Q = (3/4, 1/4).

| Outcome | P | Q | P / Q | P × log₂(P / Q) |
|---|---|---|---|---|
| 1 | 1/2 | 3/4 | 2/3 | (1/2)(−0.585) = −0.2925 |
| 2 | 1/2 | 1/4 | 2 | (1/2)(1) = 0.5 |
| **Total** | | | | **D_KL = 0.2075 bits** |

This matches the extra cost found in the cross-entropy guide.

### Example 2: Reversing the direction

Now swap the roles: D_KL(Q ‖ P).

| Outcome | Q | P | Q / P | Q × log₂(Q / P) |
|---|---|---|---|---|
| 1 | 3/4 | 1/2 | 3/2 | (3/4)(0.585) = 0.439 |
| 2 | 1/4 | 1/2 | 1/2 | (1/4)(−1) = −0.25 |
| **Total** | | | | **D_KL = 0.189 bits** |

The answer is 0.189, not 0.2075. The two directions differ, which is why KL divergence is not a true distance.

## Try it yourself

1. P = (1, 0) and Q = (1/2, 1/2). Compute D_KL(P ‖ Q).
2. P = (1/2, 1/2) and Q = (1/2, 1/2). Compute D_KL(P ‖ Q).
3. P = (1/4, 3/4) and Q = (1/2, 1/2). Compute D_KL(P ‖ Q).

<details>
<summary>Answers</summary>

1. Only the first outcome counts, because 0 × log(0/…) = 0: 1 × log₂(1 / (1/2)) = **1 bit**.
2. The distributions match, so **0 bits**.
3. (1/4) log₂((1/4)/(1/2)) + (3/4) log₂((3/4)/(1/2)) = −0.25 + 0.439 = **0.189 bits**.

</details>

## Variations and common mistakes

- **Not a distance.** It is asymmetric and does not satisfy the triangle inequality. Call it a *divergence*.
- **Order matters.** Put the true distribution P first in D_KL(P ‖ Q).
- **Zero in the wrong place.** If Q(x) = 0 but P(x) > 0, the KL divergence is infinite.
- **Zero terms.** Treat 0 × log(0/…) as 0.
- **Jensen–Shannon divergence** averages the KL divergences in both directions, which makes it symmetric.

## Remember this

- KL divergence = extra bits you pay for using Q when the truth is P.
- It is always ≥ 0, and equals 0 only when P = Q.
- It is not symmetric.

## Going further

- **Cross-entropy** = entropy + KL divergence. See [Cross-Entropy](cross_entropy.md).
- **Mutual information** can be written as a KL divergence between a joint distribution and the product of its marginals. See [Mutual Information](mutual_information.md).
- **Variational autoencoders** use KL divergence to keep learned distributions close to a simple prior.
