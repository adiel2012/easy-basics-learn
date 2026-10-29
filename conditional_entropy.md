# Conditional Entropy

> **Learning path:** [Shannon Entropy](shannon_entropy.md) → **Conditional Entropy** → [Mutual Information](mutual_information.md) → [Information Gain](information_gain.md) → [Cross-Entropy](cross_entropy.md) → [KL Divergence](kl_divergence.md) → [Bayesian Inference](bayesian_inference.md)

## In simple words

Shannon entropy measures how uncertain you are about something. Conditional entropy asks a follow-up question: **how uncertain are you still, after you learn a clue?**

Example: will it rain today? You are not sure. Then you look outside and see people carrying umbrellas. You are still not certain, but you are less unsure than before. The uncertainty that is left over is the conditional entropy.

- If the clue tells you everything, nothing is left over. The conditional entropy is 0.
- If the clue is useless, all the uncertainty is left over. The conditional entropy equals the original entropy.

## What you need to know first

- Shannon entropy, and how to compute it with a table. See [Shannon Entropy](shannon_entropy.md).
- A **joint probability table**: each cell is the chance that two things happen together.
- **Conditional probability**: the chance of one thing once you know another. You get it by dividing a cell by its column total.

## The key terms

| Symbol | Read it as | Meaning |
|---|---|---|
| P(x, y) | "probability of x and y" | One cell of the joint table |
| P(y) | "probability of y" | A column total |
| P(x \| y) | "probability of x given y" | A cell divided by its column total |
| H(X, Y) | "joint entropy of X and Y" | Uncertainty about X and Y together |
| H(X \| Y) | "entropy of X given Y" | Uncertainty about X that is left after learning Y |

## Numbers you'll need

These all come from the splitting trick in the Shannon entropy guide: −log₂(a/b) = log₂ b − log₂ a, with log₂ 3 = 1.585 and log₂ 5 = 2.322.

| Value | Result | How to get it |
|---|---|---|
| −log₂ 0.8 | 0.322 | 0.8 = 4/5, so log₂ 5 − log₂ 4 = 2.322 − 2 |
| −log₂ 0.4 | 1.322 | 0.4 = 2/5, so log₂ 5 − log₂ 2 = 2.322 − 1 |
| −log₂ 0.2 | 2.322 | 0.2 = 1/5, so log₂ 5 |
| −log₂ 0.1 | 3.322 | 0.1 = 1/10, so log₂ 10 |
| H(0.8, 0.2) | 0.722 | 0.8 × 0.322 + 0.2 × 2.322 = 0.258 + 0.464 |

## How it works

We'll use one example the whole way through.

### Step 0: Write the joint table

X is "does it rain today?" and Y is "does a person carry an umbrella?"

| | Umbrella | No umbrella | Row total |
|---|---|---|---|
| **Rain** | 0.4 | 0.1 | 0.5 |
| **No rain** | 0.1 | 0.4 | 0.5 |
| **Column total** | 0.5 | 0.5 | 1 |

Read the top-left cell as: "on 40% of days, it rains **and** the person carries an umbrella."

Without any clue, rain is 50/50, so H(Rain) = 1 bit.

### Step 1: Split the table by the clue

Look at each column on its own. Divide each cell by the column total, so the column adds up to 1.

**Column "Umbrella"** (total 0.5):

- P(rain | umbrella) = 0.4 / 0.5 = 0.8
- P(no rain | umbrella) = 0.1 / 0.5 = 0.2

**Column "No umbrella"** (total 0.5):

- P(rain | no umbrella) = 0.1 / 0.5 = 0.2
- P(no rain | no umbrella) = 0.4 / 0.5 = 0.8

### Step 2: Find the entropy of each column

Each column is now an ordinary distribution, so use the Shannon formula.

- Given an umbrella: H(0.8, 0.2) = 0.722 bits
- Given no umbrella: H(0.2, 0.8) = 0.722 bits

### Step 3: Take the weighted average

Weight each column's entropy by how often that column happens, then add.

H(Rain | Umbrella) = 0.5 × 0.722 + 0.5 × 0.722 = **0.722 bits**

As a formula:

H(X | Y) = Σ_y P(y) · H(X | Y = y)

**What this means:** before the clue, the uncertainty was 1 bit. After the clue, 0.722 bits are left. The clue removed 0.278 bits. That removed amount has its own name, mutual information, and it is the topic of the next guide.

## A shortcut: joint entropy

There is a second way to get the same answer.

**Joint entropy**, H(X, Y), is the entropy of the whole table. Treat each of the four cells as one outcome:

| Cell | p | −log₂ p | p × (−log₂ p) |
|---|---|---|---|
| Rain, umbrella | 0.4 | 1.322 | 0.529 |
| Rain, no umbrella | 0.1 | 3.322 | 0.332 |
| No rain, umbrella | 0.1 | 3.322 | 0.332 |
| No rain, no umbrella | 0.4 | 1.322 | 0.529 |
| **Total** | | | **H(X, Y) = 1.722 bits** |

Then subtract the uncertainty of the clue:

H(X | Y) = H(X, Y) − H(Y) = 1.722 − 1 = **0.722 bits** ✓

**Why this works:** the uncertainty of the whole table = the uncertainty of the clue + the uncertainty left after the clue. This is called the **chain rule**:

H(X, Y) = H(Y) + H(X | Y)

Use whichever method you like. The column method is easier to understand. The shortcut is often faster.

## Try it yourself

| | Y = 0 | Y = 1 |
|---|---|---|
| **X = 0** | 0.3 | 0.2 |
| **X = 1** | 0.1 | 0.4 |

Compute H(X | Y) with the column method. Then check your answer with the shortcut.

Helpful numbers: H(3/4, 1/4) = 0.811, H(1/3, 2/3) = 0.918, H(0.4, 0.6) = 0.971, and −log₂ 0.3 = 1.737.

<details>
<summary>Answer</summary>

**Column method:**

1. Column totals: P(Y=0) = 0.3 + 0.1 = 0.4, and P(Y=1) = 0.2 + 0.4 = 0.6.
2. Column Y = 0: 0.3/0.4 = 0.75 and 0.1/0.4 = 0.25, so H = 0.811 bits.
3. Column Y = 1: 0.2/0.6 = 1/3 and 0.4/0.6 = 2/3, so H = 0.918 bits.
4. H(X | Y) = 0.4 × 0.811 + 0.6 × 0.918 = 0.324 + 0.551 = **0.875 bits**.

**Shortcut check:**

- H(X, Y) = 0.3 × 1.737 + 0.2 × 2.322 + 0.1 × 3.322 + 0.4 × 1.322 = 0.521 + 0.464 + 0.332 + 0.529 = 1.846
- H(Y) = H(0.4, 0.6) = 0.971
- H(X | Y) = 1.846 − 0.971 = **0.875 bits** ✓

</details>

## Variations and common mistakes

- **Forgetting the weights.** Do not just average the column entropies. Weight each one by its column total. In the exercise, a plain average gives (0.811 + 0.918) / 2 = 0.865, which is wrong.
- **Dividing by the wrong total.** For H(X | Y), divide by the **column** totals (the Y values). Dividing by the row totals gives H(Y | X), which is a different number.
- **Not symmetric.** H(X | Y) and H(Y | X) are usually different. They are related by H(X | Y) − H(Y | X) = H(X) − H(Y).
- **A clue can raise uncertainty in one case.** On average, a clue never increases uncertainty: H(X | Y) ≤ H(X). But one particular clue value can. Snow is rare in some city, so you are nearly sure it won't snow. Learning "it's January" makes snow a real possibility, so for that clue your uncertainty goes up. Averaged over all months, it still goes down. [Bayesian Inference](bayesian_inference.md) shows the same effect with a medical test.

## Remember this

- Conditional entropy = the uncertainty left after a clue.
- Column method: split into columns, find each column's entropy, then take the weighted average.
- Shortcut: H(X | Y) = H(X, Y) − H(Y).
- It is always between 0 and H(X).

## Going further

- **Next:** [Mutual Information](mutual_information.md) measures the uncertainty the clue **removed**: H(X) − H(X | Y).
- **Language models.** The chain rule extends to many words: H(sentence) = H(word 1) + H(word 2 | word 1) + H(word 3 | words 1–2) + … Each step is a conditional entropy. See [LLM Inference](llm_inference.md).
