# KL Divergence

> **Learning path:** [Shannon Entropy](shannon_entropy.md) → [Conditional Entropy](conditional_entropy.md) → [Mutual Information](mutual_information.md) → [Information Gain](information_gain.md) → [Cross-Entropy](cross_entropy.md) → **KL Divergence** → [Bayesian Inference](bayesian_inference.md)

## In simple words

In the cross-entropy guide, the weather app had an average surprise of 1.2075 bits. Of that, 1 bit could not be avoided, because the weather really is uncertain. The other 0.2075 bits came only from the app's wrong numbers.

**KL divergence is that extra part.** It measures how much you lose by using a predicted distribution Q when the truth is P.

- If the prediction is perfect, the KL divergence is 0.
- If the prediction is wrong, the KL divergence is positive.
- It is never negative.

One surprising fact: **the direction matters.** If the truth is 50/50 and you predict 75/25, you lose 0.208 bits. If the truth is 75/25 and you predict 50/50, you lose 0.189 bits. The cost depends on which one is the truth.

## What you need to know first

- Shannon entropy. See [Shannon Entropy](shannon_entropy.md).
- Cross-entropy. See [Cross-Entropy](cross_entropy.md).
- Logs of fractions: log₂(a/b) = log₂ a − log₂ b.

## The key terms

| Symbol | Meaning |
|---|---|
| P(x) | The true probability of outcome x |
| Q(x) | The predicted probability of outcome x |
| D_KL(P ‖ Q) | The KL divergence. The double bar just separates the two. **The truth goes first.** |

## Numbers you'll need

| Value | Result | Why |
|---|---|---|
| log₂ 2 | 1 | |
| log₂(1/2) | −1 | |
| log₂(3/2) | 0.585 | log₂ 3 − log₂ 2 = 1.585 − 1 |
| log₂(2/3) | −0.585 | log₂ 2 − log₂ 3 = 1 − 1.585 |

## How it works

There are two ways to compute it.

**Way 1, by subtraction** (easiest if you already know the cross-entropy):

D_KL(P ‖ Q) = H(P, Q) − H(P)

**Way 2, directly:**

D_KL(P ‖ Q) = Σ P(x) · log₂ ( P(x) / Q(x) )

For each outcome, compare the truth with the prediction (P / Q), take log₂, and weight it by how often the outcome really happens (P).

**Watch out:** in Way 2, a single row can be **negative**. That happens when the prediction gave an outcome more probability than it deserves. The **total** is never negative.

Key facts:

- It is always ≥ 0.
- It is 0 only when P = Q.
- It is not symmetric: D_KL(P ‖ Q) is usually different from D_KL(Q ‖ P).

### Example 1: The weather app

Truth P = (1/2, 1/2). Prediction Q = (3/4, 1/4).

**Way 1:** from the cross-entropy guide, H(P, Q) = 1.2075 and H(P) = 1, so D_KL = **0.2075 bits**.

**Way 2:**

| Outcome | P | Q | P / Q | log₂(P / Q) | P × log₂(P / Q) |
|---|---|---|---|---|---|
| Rain | 1/2 | 3/4 | 2/3 | −0.585 | −0.2925 |
| Dry | 1/2 | 1/4 | 2 | 1 | 0.5 |
| **Total** | | | | | **0.2075 bits** |

Both ways agree. Notice the first row is negative, but the total is positive.

### Example 2: Swap the roles

Now the truth is (3/4, 1/4) and the prediction is (1/2, 1/2). This is D_KL(Q ‖ P) from Example 1.

| Outcome | Truth | Prediction | Truth / Prediction | log₂ | Truth × log₂ |
|---|---|---|---|---|---|
| Rain | 3/4 | 1/2 | 3/2 | 0.585 | 0.439 |
| Dry | 1/4 | 1/2 | 1/2 | −1 | −0.25 |
| **Total** | | | | | **0.189 bits** |

| Truth | Prediction | Cost |
|---|---|---|
| (1/2, 1/2) | (3/4, 1/4) | 0.2075 bits |
| (3/4, 1/4) | (1/2, 1/2) | 0.189 bits |

Same two distributions, different costs. This is why KL divergence is called a *divergence*, not a distance.

## Try it yourself

1. Truth P = (1, 0). Prediction Q = (1/2, 1/2). Compute D_KL(P ‖ Q).
2. Truth P = (1/2, 1/2). Prediction Q = (1/2, 1/2). Compute D_KL(P ‖ Q).
3. Truth P = (1/4, 3/4). Prediction Q = (1/2, 1/2). Compute D_KL(P ‖ Q). Then check it with Way 1, using the cross-entropy practice problem 1.

<details>
<summary>Answers</summary>

1. The second row has P = 0, so it counts as 0. The first row: 1 × log₂(1 / (1/2)) = 1 × log₂ 2 = **1 bit**.
2. The prediction is perfect, so **0 bits**.
3. (1/4) × log₂(1/2) + (3/4) × log₂(3/2) = −0.25 + 0.439 = **0.189 bits**. Check: H(P, Q) − H(P) = 1 − 0.811 = 0.189 ✓

</details>

## Variations and common mistakes

- **Calling it a distance.** It is not symmetric, so it is not a true distance. Always say which distribution is the truth.
- **Putting the arguments in the wrong order.** In D_KL(P ‖ Q), P is the truth and goes first.
- **Panicking at a negative row.** Single rows can be negative. Only the total must be ≥ 0. If your **total** is negative, you made a mistake.
- **A zero in the wrong place.** Suppose the truth is (1/2, 1/2) and the prediction is (1, 0). The prediction says the second outcome is impossible, but it happens half the time. The KL divergence is infinite.
- **Rows where P = 0.** Treat 0 × log(0 / …) as 0. An outcome that never happens costs nothing.
- **Why machine learning minimizes cross-entropy.** Cross-entropy = H(P) + D_KL(P ‖ Q), and H(P) does not depend on the model. So making cross-entropy smaller is the same as making the KL divergence smaller.

## Remember this

- KL divergence = the extra bits you pay for predicting Q when the truth is P.
- KL divergence = cross-entropy − entropy.
- It is ≥ 0, equal to 0 only for a perfect prediction, and the direction matters.

## Going further

- **Next:** [Bayesian Inference](bayesian_inference.md) shows how to update a prediction when new evidence arrives.
- **Mutual information** is a KL divergence: it compares the real joint table with the table you would get if the variables were independent. See [Mutual Information](mutual_information.md).
- **Variational autoencoders** use KL divergence to keep a learned distribution close to a simple one.
