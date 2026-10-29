# Information Gain

> **Learning path:** [Shannon Entropy](shannon_entropy.md) → [Conditional Entropy](conditional_entropy.md) → [Mutual Information](mutual_information.md) → **Information Gain** → [Cross-Entropy](cross_entropy.md) → [KL Divergence](kl_divergence.md) → [Bayesian Inference](bayesian_inference.md)

## In simple words

Information gain is mutual information put to work.

You want to predict something, such as "will this customer buy?" You can learn one clue first, such as their age, whether they own a dog, or their favorite color. **Which clue is most useful?** The one that removes the most uncertainty. Information gain is the number that measures this.

Think of 20 Questions. "Is it an animal?" is a better first question than "Is it a purple giraffe?", because it cuts the possibilities down much more on average.

Decision trees use information gain to decide which question to ask first.

## What you need to know first

- Shannon entropy. See [Shannon Entropy](shannon_entropy.md).
- The column method for conditional entropy: split, find each group's entropy, take the weighted average. See [Conditional Entropy](conditional_entropy.md).
- Information gain is the same quantity as mutual information. See [Mutual Information](mutual_information.md).

## The key terms

| Term | Meaning |
|---|---|
| Target, X | What you want to predict, such as "buys?" |
| Feature, Y | A clue you could learn first, such as "age group" |
| Group | All the rows with the same feature value, such as all the young people |
| H(X) | Uncertainty about the target before learning the feature |
| H(X \| Y) | Average uncertainty about the target after learning the feature |
| IG | Information gain: IG = H(X) − H(X \| Y) |

## Numbers you'll need

| Value | Result | Where it comes from |
|---|---|---|
| H(1/2, 1/2) | 1 | A 50/50 split |
| H(3/4, 1/4) | 0.811 | Shannon entropy guide, practice problem 2 |
| H(2/3, 1/3) | 0.918 | (2/3)(0.585) + (1/3)(1.585) = 0.390 + 0.528 |
| H(1, 0) | 0 | A group where everyone gives the same answer |

## How it works

The formula is:

IG = H(X) − H(X | Y)

With real data you usually have **counts**, not probabilities. The recipe is:

1. **Before:** count the target answers in the whole dataset and compute H(X).
2. **Split:** put the rows into groups by the feature's value.
3. **Each group:** compute the entropy of the target inside each group.
4. **After:** take the weighted average of the group entropies. Each weight is the group size divided by the total. This gives H(X | Y).
5. **Gain:** subtract. IG = before − after.

Information gain is never negative. On average, learning a clue cannot make you more uncertain.

### Example: Which question should we ask first?

Here are eight customers:

| Person | Age | Owns a dog | Likes blue | Buys? |
|---|---|---|---|---|
| 1 | Young | Yes | Yes | Yes |
| 2 | Young | Yes | No | Yes |
| 3 | Young | Yes | No | Yes |
| 4 | Young | No | Yes | No |
| 5 | Old | Yes | Yes | Yes |
| 6 | Old | No | Yes | No |
| 7 | Old | No | No | No |
| 8 | Old | No | No | No |

**Step 1, before:** 4 people buy and 4 don't, so H(Buys) = **1 bit**.

**Steps 2 to 5, for each feature:**

| Feature | Group | Size | Buy / Don't | Group entropy | Weighted (size/8 × entropy) |
|---|---|---|---|---|---|
| Age | Young | 4 | 3 / 1 | 0.811 | 0.406 |
| | Old | 4 | 1 / 3 | 0.811 | 0.406 |
| | | | | **After: 0.811** | **IG = 1 − 0.811 = 0.189** |
| Owns a dog | Yes | 4 | 4 / 0 | 0 | 0 |
| | No | 4 | 0 / 4 | 0 | 0 |
| | | | | **After: 0** | **IG = 1 − 0 = 1** |
| Likes blue | Yes | 4 | 2 / 2 | 1 | 0.5 |
| | No | 4 | 2 / 2 | 1 | 0.5 |
| | | | | **After: 1** | **IG = 1 − 1 = 0** |

**Result:**

- "Owns a dog" removes all the uncertainty (IG = 1 bit). A decision tree asks this first.
- "Age" helps a little (IG = 0.189 bits).
- "Likes blue" tells you nothing (IG = 0 bits). Every group is still 50/50.

This is a toy example. Real features almost never split the data perfectly.

## Try it yourself

Six people. The target is "likes tea": 3 yes and 3 no. You can ask one of two questions.

**Question A: "Are you a morning person?"**

| Group | People | Likes tea | Doesn't |
|---|---|---|---|
| Morning person | 3 | 2 | 1 |
| Not a morning person | 3 | 1 | 2 |

**Question B: "Do you drink coffee?"**

| Group | People | Likes tea | Doesn't |
|---|---|---|---|
| Drinks coffee | 2 | 0 | 2 |
| Doesn't drink coffee | 4 | 3 | 1 |

Compute the information gain for each question. Which one should a decision tree ask first?

Hint: for question B the groups have different sizes, so the weights are 2/6 and 4/6.

<details>
<summary>Answers</summary>

**Before:** H(tea) = 1 bit (3 yes, 3 no).

**Question A:**

- Each group is split (2/3, 1/3), so each group's entropy is 0.918.
- After = (3/6)(0.918) + (3/6)(0.918) = 0.918
- IG = 1 − 0.918 = **0.082 bits**

**Question B:**

- Coffee drinkers: (0, 2), so the entropy is 0.
- Non-coffee drinkers: (3, 1), so the entropy is 0.811.
- After = (2/6)(0) + (4/6)(0.811) = 0.541
- IG = 1 − 0.541 = **0.459 bits**

**Ask question B first.** It removes more uncertainty.

</details>

## Variations and common mistakes

- **Forgetting the weights.** Each group's entropy must be weighted by its size. This matters most when the groups have different sizes, as in question B.
- **Features with many values.** Imagine a "person ID" feature. Each person forms a group of one, every group has entropy 0, and the information gain is the full 1 bit. But the feature is useless for new customers. The **gain ratio** (used in the C4.5 algorithm) fixes this by dividing IG by the entropy of the feature itself. For person ID that entropy is log₂ 8 = 3, so the gain ratio is 1/3. For "owns a dog" it is 1, so the gain ratio is 1/1 = 1, and the dog question still wins.
- **Mixing up target and feature.** H(X) uses the target column only.
- **Expecting big numbers.** Small gains are normal. A useful feature may remove only a fraction of a bit.
- **One question is not a whole tree.** A decision tree repeats this choice inside each group, until the groups are pure or too small to split.

## Remember this

- IG = uncertainty before − uncertainty after.
- Compute it for every feature, then ask the question with the largest gain.
- It is never negative, and it equals the mutual information between the target and the feature.

## Going further

- **Next:** [Cross-Entropy](cross_entropy.md) measures how good a model's predicted probabilities are, rather than how good a question is.
- **Decision trees** are built by applying information gain over and over.
- **Mutual information** is the same quantity under another name. See [Mutual Information](mutual_information.md).
