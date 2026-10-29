# Bayesian Inference

## In simple words

Bayesian inference is a way to **update a belief** when new evidence arrives. You start with a belief you already have (the **prior**), look at the evidence, and combine the two into a new belief (the **posterior**).

Example: a doctor thinks a patient is probably healthy. A test comes back positive. Bayes' rule tells the doctor how much the belief should change, given how accurate the test is and how common the disease is.

The key idea is that the posterior depends on **both** the evidence and the prior. A rare condition needs strong evidence before it becomes likely.

## What you need to know first

- Conditional probability: P(A | B) is the probability of A if B is known.
- Multiplication and division of probabilities.
- Probabilities of a hypothesis and its opposite add up to 1.

## The key terms

| Term | Meaning | Medical example |
|---|---|---|
| P(H) | The **prior**: belief before the evidence | Chance of the disease before the test (1%) |
| P(E \| H) | The **likelihood**: chance of the evidence if H is true | Chance of a positive test if sick (90%) |
| P(E) | The **evidence**: total chance of seeing the evidence | Chance of a positive test at all |
| P(H \| E) | The **posterior**: belief after the evidence | Chance of being sick after a positive test |

## How it works

Bayes' rule:

P(H | E) = P(E | H) · P(H) / P(E)

The total probability of the evidence is:

P(E) = P(E | H) · P(H) + P(E | not H) · P(not H)

Steps:

1. Start with the prior P(H).
2. Multiply it by the likelihood P(E | H). This is the probability of the evidence if the hypothesis is true.
3. Compute P(E) by adding up all the ways the evidence can happen.
4. Divide to get the posterior.

### Example: A medical test

- Disease prevalence: 1%, so P(sick) = 0.01.
- Test sensitivity: P(positive | sick) = 0.90.
- False positive rate: P(positive | healthy) = 0.05.

A patient tests positive. What is P(sick | positive)?

- Sick and positive: 0.90 × 0.01 = 0.009
- Healthy and positive: 0.05 × 0.99 = 0.0495
- P(positive) = 0.009 + 0.0495 = 0.0585
- P(sick | positive) = 0.009 / 0.0585 = **0.154, about 15%**

Surprising? Picture 1,000 people. About 10 are sick, and 9 of them test positive. About 990 are healthy, and about 50 of them test positive. Out of about 59 positive tests, only 9 are sick people. The disease is rare, so most positive results are false alarms.

## Try it yourself

A disease affects 2% of people. The test catches 95% of sick people, and 10% of healthy people test positive. A patient tests positive. What is P(sick | positive)?

<details>
<summary>Answer</summary>

- Sick and positive: 0.95 × 0.02 = 0.019
- Healthy and positive: 0.10 × 0.98 = 0.098
- P(positive) = 0.019 + 0.098 = 0.117
- P(sick | positive) = 0.019 / 0.117 = **0.162, about 16%**

**Going further:** The same patient takes a second, independent test and is positive again. Use the 16% as the new prior:

- Sick and positive: 0.95 × 0.162 = 0.154
- Healthy and positive: 0.10 × 0.838 = 0.084
- P(sick | two positives) = 0.154 / (0.154 + 0.084) = **0.65, about 65%**

Each new piece of evidence updates the belief again.

</details>

## Variations and common mistakes

- **Base rate neglect.** Ignoring the prior is the most common mistake. The rare disease example above is a classic case.
- **Confusing P(E | H) with P(H | E).** The chance of a positive test if sick is not the same as the chance of being sick if positive. This confusion is called the prosecutor's fallacy in legal settings.
- **The prior matters less with lots of evidence.** After many observations, different starting beliefs tend to lead to similar posteriors.
- **Sequential updates.** The posterior from one step becomes the prior for the next, as long as the pieces of evidence are independent.

## Remember this

- Posterior ∝ likelihood × prior.
- Rare conditions need strong evidence before they become likely.
- New evidence updates the belief step by step.

## Going further

- **Information gain and conditional entropy** also measure how evidence changes uncertainty. See [Information Gain](information_gain.md) and [Conditional Entropy](conditional_entropy.md).
- **Probabilistic models** such as VAEs use Bayesian ideas together with KL divergence. See [KL Divergence](kl_divergence.md).
