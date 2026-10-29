# Cross-Entropy

## In simple words

A weather app says there is a 75% chance of rain. You use its numbers, but the real weather follows its own true chances. Cross-entropy measures how much surprise you get **on average** when you judge outcomes with the app's numbers.

If the app's numbers are close to the truth, the average surprise is small. If the app is confidently wrong, the average surprise is large.

Think of a map that is slightly out of date. Cross-entropy is the average extra trouble the wrong map causes, measured in bits.

## What you need to know first

- Shannon entropy, the average surprise of a distribution. See [Shannon Entropy](shannon_entropy.md).
- Logarithms base 2, and how to compute surprise −log₂ p.
- Probabilities for all outcomes add up to 1.

## The key terms

| Symbol | Meaning |
|---|---|
| P(x) | The **true** probability of outcome x |
| Q(x) | The probability the **model** assigns to outcome x |
| H(P, Q) | The cross-entropy, in bits |

## How it works

The formula is:

H(P, Q) = −Σ P(x) · log₂ Q(x)

Compare it with Shannon entropy, H(P) = −Σ P(x) · log₂ P(x). The only change is that the log now uses the model's probability Q. The outcomes are still weighted by how often they truly happen, P.

There is a key identity:

H(P, Q) = H(P) + D_KL(P ‖ Q)

- H(P) is the uncertainty that cannot be removed.
- D_KL is the extra cost of using the wrong probabilities. See [KL Divergence](kl_divergence.md).

So cross-entropy is never smaller than H(P). It equals H(P) only when Q = P.

### Example 1: Checking the numbers by hand

True probabilities: P = (1/2, 1/2). Model: Q = (3/4, 1/4).

| Outcome | P | Q | −log₂ Q | P × (−log₂ Q) |
|---|---|---|---|---|
| 1 | 1/2 | 3/4 | 0.415 | 0.2075 |
| 2 | 1/2 | 1/4 | 2 | 1.0 |
| **Total** | | | | **H(P, Q) = 1.2075 bits** |

The true entropy is H(P) = 1 bit. The extra cost is 0.2075 bits, which is the KL divergence.

### Example 2: A classifier

A photo really shows a cat. The true distribution is P = (cat: 1, not cat: 0). Only the cat outcome counts, so the loss is just −log₂ Q(cat).

- Model A says "cat" with 0.8: loss = −log₂ 0.8 = **0.322 bits**
- Model B says "cat" with 0.1: loss = −log₂ 0.1 = **3.322 bits**

Confident and correct is cheap. Confident and wrong is expensive. Training tries to make the average of these losses as small as possible.

## Try it yourself

1. The true distribution is P = (1/4, 3/4). The model says Q = (1/2, 1/2). Compute H(P, Q).
2. The true label is class B, so P = (0, 1). The model says Q = (0.2, 0.8). Compute the loss.

<details>
<summary>Answers</summary>

1. −(1/4) log₂(1/2) − (3/4) log₂(1/2) = 1/4 + 3/4 = **1 bit**. As a check, H(P) = 0.811, so the extra cost is 0.189 bits, which matches the KL divergence.
2. Only class B counts: −log₂ 0.8 = **0.322 bits**.

</details>

## Variations and common mistakes

- **Natural logarithm.** Machine learning libraries often use ln, which gives *nats* instead of bits. The idea is the same.
- **Not symmetric.** H(P, Q) is generally different from H(Q, P). The true distribution goes first.
- **Zero probability.** If Q(x) = 0 for an outcome that can happen (P(x) > 0), the loss is infinite. This is why models use softmax, which never gives exactly zero.
- **Forgetting the weights.** Each outcome's surprise is weighted by its true probability P, not by the model's Q.
- **Thinking a lower loss always means a perfect model.** Lower is better, but the loss is still limited by the uncertainty H(P) in the data.

## Remember this

- Cross-entropy = average surprise when you predict with Q but outcomes follow P.
- Its smallest possible value is H(P), reached when Q = P.
- When used as a training loss, lower is better.

## Going further

- **KL divergence** is the extra part of cross-entropy. See [KL Divergence](kl_divergence.md).
- **Backpropagation** computes the gradient of this loss so the weights can be adjusted. See [Backpropagation](backpropagation.md).
- **Language models** are trained by minimizing the cross-entropy of the next token. See [LLM Inference](llm_inference.md).
