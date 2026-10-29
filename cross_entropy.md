# Cross-Entropy

> **Learning path:** [Shannon Entropy](shannon_entropy.md) → [Conditional Entropy](conditional_entropy.md) → [Mutual Information](mutual_information.md) → [Information Gain](information_gain.md) → **Cross-Entropy** → [KL Divergence](kl_divergence.md) → [Bayesian Inference](bayesian_inference.md)

## In simple words

Suppose it really rains on **half** of all days. Your weather app, however, always says "75% chance of rain."

- On rainy days, the app said 75%, so you are only a little surprised.
- On dry days, the app said only 25% chance of dry weather, so you are very surprised.

Cross-entropy is your **average surprise** when you use the app's numbers while the weather follows its true chances. For this app, it is 1.2075 bits.

If the app had said the true 50%, your average surprise would be 1 bit, which is just the Shannon entropy of the weather. The extra 0.2075 bits is the price of using wrong numbers.

That is the whole idea: **cross-entropy = unavoidable surprise + extra surprise from wrong predictions.**

## What you need to know first

- Shannon entropy, the average surprise of a distribution. See [Shannon Entropy](shannon_entropy.md).
- How to compute a surprise, −log₂ p.

## The key terms

| Symbol | Meaning |
|---|---|
| P(x) | The **true** probability of outcome x: how often it really happens |
| Q(x) | The **predicted** probability of outcome x: what the model or app says |
| H(P) | The Shannon entropy of the truth. It is the same thing as H(X) in the Shannon guide. |
| H(P, Q) | The cross-entropy, in bits |

## Numbers you'll need

| Value | Result |
|---|---|
| −log₂ 0.5 | 1 |
| −log₂ 0.25 | 2 |
| −log₂ 0.75 | 0.415 |
| −log₂ 0.8 | 0.322 |
| −log₂ 0.1 | 3.322 |

## How it works

The formula is:

H(P, Q) = −Σ P(x) · log₂ Q(x)

Each probability has one job:

| | How often each outcome counts (the weight) | How surprised you are (inside the log) |
|---|---|---|
| Shannon entropy, H(P) | P, the truth | −log₂ P, the truth |
| Cross-entropy, H(P, Q) | P, the truth | −log₂ Q, the prediction |

So the only change from Shannon entropy is that the surprise now comes from the **prediction**. The weights still come from the truth, because the truth decides how often each outcome actually happens.

There is a key identity:

H(P, Q) = H(P) + D_KL(P ‖ Q)

- H(P) is the uncertainty that no prediction can remove.
- D_KL(P ‖ Q) is the extra cost of wrong predictions. It is the topic of the [KL Divergence](kl_divergence.md) guide.

This means cross-entropy is never smaller than H(P). It equals H(P) only when the prediction is perfect, Q = P.

### Example 1: The weather app, by hand

Truth: P = (rain 1/2, dry 1/2). App: Q = (rain 3/4, dry 1/4).

| Outcome | P (weight) | Q | Surprise −log₂ Q | P × surprise |
|---|---|---|---|---|
| Rain | 1/2 | 3/4 | 0.415 | 0.2075 |
| Dry | 1/2 | 1/4 | 2 | 1.0 |
| **Total** | | | | **H(P, Q) = 1.2075 bits** |

The true entropy is H(P) = 1 bit, so the extra cost is 1.2075 − 1 = 0.2075 bits.

### Example 2: A classifier looking at one photo

A photo really shows a cat. So the truth is P = (cat 1, not cat 0).

Write out the formula:

H(P, Q) = −1 · log₂ Q(cat) − 0 · log₂ Q(not cat)

The second term is multiplied by 0, so it disappears. Only the correct answer counts:

**Loss = −log₂ Q(correct answer)**

Here is the loss for different predictions:

| Model's probability for "cat" | 0.99 | 0.8 | 0.5 | 0.1 | 0.01 |
|---|---|---|---|---|---|
| Loss (bits) | 0.014 | 0.322 | 1 | 3.322 | 6.644 |

Notice the pattern: **each time the probability on the right answer is halved, the loss grows by exactly 1 bit.** Confident and right is cheap. Confident and wrong is very expensive.

### Example 3: A whole dataset

Training uses the **average** loss over many examples. A model gives these probabilities to the correct answer on four photos:

| Photo | Probability on the correct answer | Loss (bits) |
|---|---|---|
| 1 | 0.8 | 0.322 |
| 2 | 0.5 | 1 |
| 3 | 0.5 | 1 |
| 4 | 0.25 | 2 |
| **Average** | | **4.322 / 4 = 1.08 bits** |

Training adjusts the model to make this average smaller.

## Try it yourself

1. The truth is P = (1/4, 3/4). The model says Q = (1/2, 1/2). Compute H(P, Q).
2. The correct answer is class B, so P = (0, 1). The model says Q = (0.2, 0.8). Compute the loss.
3. A model puts probability 0.5 on the correct answer for one photo and 0.25 for another. What is the average loss?

<details>
<summary>Answers</summary>

1. Every surprise is −log₂(1/2) = 1, so H(P, Q) = (1/4)(1) + (3/4)(1) = **1 bit**. As a check, H(P) = 0.811, so the extra cost is 0.189 bits.
2. Only class B counts: −log₂ 0.8 = **0.322 bits**.
3. The losses are 1 and 2, so the average is **1.5 bits**.

</details>

## Variations and common mistakes

- **Natural logarithm.** Machine learning libraries usually use ln instead of log₂. The answer is then in *nats*, where 1 nat ≈ 1.443 bits. The idea is the same.
- **Order matters.** H(P, Q) is usually different from H(Q, P). The truth goes first.
- **Zero probability.** If the model gives probability 0 to something that really happens, its surprise is infinite, and so is the loss. This is one reason models use softmax, which never outputs exactly 0.
- **Using the wrong weights.** Each surprise is weighted by the true probability P, not the predicted Q.
- **Expecting a loss of 0.** If the labels themselves are uncertain (H(P) > 0), even a perfect model cannot go below H(P).

## Remember this

- Cross-entropy = average surprise when you predict with Q but the world follows P.
- Cross-entropy = H(P) + extra cost. Its smallest value, H(P), is reached when Q = P.
- For a single labeled example, the loss is −log₂ of the probability on the correct answer.

## Going further

- **Next:** [KL Divergence](kl_divergence.md) looks at just the extra cost.
- **Training.** Backpropagation computes how each weight affects this loss, so the weights can be adjusted. See [Backpropagation](backpropagation.md).
- **Language models** are trained by minimizing the cross-entropy of the next token. See [LLM Inference](llm_inference.md).
