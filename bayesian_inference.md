# Bayesian Inference

> **Learning path:** [Shannon Entropy](shannon_entropy.md) → [Conditional Entropy](conditional_entropy.md) → [Mutual Information](mutual_information.md) → [Information Gain](information_gain.md) → [Cross-Entropy](cross_entropy.md) → [KL Divergence](kl_divergence.md) → **Bayesian Inference**

## In simple words

Bayesian inference is a way to **update a belief** when new evidence arrives.

1. You start with a belief you already have. This is the **prior**.
2. You see some evidence.
3. You combine the two into a new belief. This is the **posterior**.

Example: a doctor thinks a patient is probably healthy, because the disease is rare. Then a test comes back positive. How worried should the doctor be? Bayes' rule gives the answer. It depends on **both** how accurate the test is **and** how rare the disease is.

The surprising lesson: when something is rare, even a good test can give mostly false alarms.

## What you need to know first

- Conditional probability: P(A | B) means "the probability of A, if we know B."
- Multiplying and dividing probabilities.
- Turning a percentage into a count: 1% of 10,000 people is 100 people.

## The key terms

| Term | Name | Medical example |
|---|---|---|
| P(H) | **Prior**: belief before the evidence | Chance of having the disease before testing: 1% |
| P(E \| H) | **Likelihood**: chance of the evidence if H is true | Chance of a positive test if sick: 90% |
| P(E \| not H) | Chance of the evidence if H is false | Chance of a positive test if healthy (a false alarm): 5% |
| P(E) | **Evidence**: total chance of seeing the evidence | Chance of a positive test for anyone |
| P(H \| E) | **Posterior**: belief after the evidence | Chance of being sick after a positive test |

## How it works

There are two methods. They always give the same answer. **Start with method 1.** It is the easiest to understand and to do by hand.

### Method 1: Count people

Imagine 10,000 people and fill in a table.

1. **Prior:** 1% are sick, so 100 are sick and 9,900 are healthy.
2. **Sick people:** 90% test positive, so 90 positive and 10 negative.
3. **Healthy people:** 5% test positive, so 495 positive and 9,405 negative.
4. **Add up the columns.**

| | Test positive | Test negative | Total |
|---|---|---|---|
| **Sick** | 90 | 10 | 100 |
| **Healthy** | 495 | 9,405 | 9,900 |
| **Total** | 585 | 9,415 | 10,000 |

5. **Answer the question.** Look only at the "Test positive" column. Of 585 positive tests, 90 come from sick people:

P(sick | positive) = 90 / 585 = **0.154, about 15%**

Now you can see why: there are so many healthy people that their few false alarms (495) outnumber the real cases (90).

### Method 2: Bayes' rule

The same calculation as a formula:

P(H | E) = P(E | H) · P(H) / P(E)

where the total chance of the evidence is:

P(E) = P(E | H) · P(H) + P(E | not H) · P(not H)

For the medical test:

- Sick and positive: 0.90 × 0.01 = 0.009
- Healthy and positive: 0.05 × 0.99 = 0.0495
- P(positive) = 0.009 + 0.0495 = 0.0585
- P(sick | positive) = 0.009 / 0.0585 = **0.154** ✓

Compare with method 1: 0.009 is 90 out of 10,000, and 0.0585 is 585 out of 10,000. The formula is just the table, written with probabilities.

## Try it yourself

A disease affects 2% of people. The test catches 95% of sick people, and 10% of healthy people get a false positive.

1. A patient tests positive. What is P(sick | positive)? Use the 10,000-people table.
2. The same patient takes a second, independent test, and it is positive again. What is the chance now?

<details>
<summary>Answers</summary>

**1.** With 10,000 people:

| | Test positive | Test negative | Total |
|---|---|---|---|
| **Sick** | 190 | 10 | 200 |
| **Healthy** | 980 | 8,820 | 9,800 |
| **Total** | 1,170 | 8,830 | 10,000 |

P(sick | positive) = 190 / 1,170 = **0.162, about 16%**

**2.** Start from the 1,170 people who tested positive: 190 sick and 980 healthy. These are the new prior.

- Sick: 95% of 190 test positive again, which is 180.5.
- Healthy: 10% of 980 test positive again, which is 98.

P(sick | two positives) = 180.5 / (180.5 + 98) = 180.5 / 278.5 = **0.65, about 65%**

Each new piece of evidence updates the belief again. Half a person is fine here. These are averages, not real people.

</details>

## Variations and common mistakes

- **Base rate neglect.** Ignoring the prior is the most common mistake. Many people guess that a positive result on a 90%-accurate test means a 90% chance of being sick. The table shows it is only 15% when the disease is rare.
- **Swapping the condition.** P(positive | sick) = 90% is not the same as P(sick | positive) = 15%. In court cases this mistake is called the *prosecutor's fallacy*.
- **A fast mental method: odds.** Write the prior as odds: 1% is 1 to 99. Multiply by the **likelihood ratio**, 90% / 5% = 18. The new odds are 18 to 99, so the probability is 18 / (18 + 99) = 18 / 117 = 0.154. Same answer.
- **The prior matters less with more evidence.** After many observations, people who started with different priors end up with similar posteriors.
- **Sequential updates need independence.** The posterior from one step can become the prior for the next, but only if the two pieces of evidence are independent given the true state.

## Connecting back to entropy

How uncertain is the doctor at each step? Use the Shannon entropy of the belief (calculator values):

| Moment | P(sick) | Uncertainty H |
|---|---|---|
| Before the test | 0.01 | 0.081 bits |
| After a **positive** test | 0.154 | 0.62 bits |
| After a **negative** test | 10 / 9,415 ≈ 0.001 | 0.012 bits |

A positive test **raised** the doctor's uncertainty: from "almost surely healthy" to "maybe sick." A negative test lowered it. Averaged over both results (58.5% weight on positive, 94.15% on negative), the uncertainty after the test is about 0.048 bits, lower than the 0.081 bits before.

This is exactly the rule from the [Conditional Entropy](conditional_entropy.md) guide: one particular clue can raise uncertainty, but on average a clue never does. The test's [information gain](information_gain.md) is 0.081 − 0.048 ≈ 0.033 bits.

## Remember this

- Posterior ∝ likelihood × prior.
- When in doubt, draw the 10,000-people table.
- Rare conditions need strong evidence before they become likely.
- New evidence updates the belief, step by step.

## Going further

- **Information theory.** Conditional entropy and information gain measure how much a piece of evidence reduces uncertainty on average, as shown above. See [Conditional Entropy](conditional_entropy.md) and [Information Gain](information_gain.md).
- **Probabilistic machine learning.** Models such as variational autoencoders combine Bayesian updating with [KL Divergence](kl_divergence.md).
