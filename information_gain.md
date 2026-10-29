# Information Gain

## In simple words

Information gain measures how much a piece of information **reduces uncertainty**. Before you ask a question, you are uncertain about the answer. After you learn the answer, you are less uncertain. The difference is the information gain.

Think of 20 Questions. The best question is the one that cuts the possibilities down the most. Information gain is the number that measures how well a question does this.

Decision trees use it to decide which feature to split on first.

## What you need to know first

- Shannon entropy. See [Shannon Entropy](shannon_entropy.md).
- Conditional entropy, the uncertainty left after learning a feature. See [Conditional Entropy](conditional_entropy.md).
- Splitting a group into smaller groups, and weighting each group by its size.

## The key terms

| Symbol | Meaning |
|---|---|
| X | The target: what you want to predict (for example, "buys the product?") |
| Y | The feature: the information you might learn (for example, "age group") |
| H(X) | The uncertainty about X before learning Y |
| H(X \| Y) | The average uncertainty about X after learning Y |
| IG | The information gain, IG = H(X) − H(X \| Y) |

## How it works

The formula is:

IG = H(X) − H(X | Y)

Steps:

1. Compute H(X) from the overall group.
2. Split the group by the value of Y.
3. For each value, compute the entropy of X inside that group.
4. Weight each group's entropy by the group's share of the total, then add them. This gives H(X | Y).
5. Subtract H(X | Y) from H(X).

Information gain is never negative. On average, learning something cannot make you more uncertain.

### Example: Who buys the product?

Eight people. The target X is "buys?" with 4 yes and 4 no, so H(X) = 1 bit.

Feature Y is age group:

| Group | People | Buy | Don't buy | Entropy of X in group |
|---|---|---|---|---|
| Young | 4 | 3 | 1 | H(3/4, 1/4) = 0.811 |
| Old | 4 | 1 | 3 | H(1/4, 3/4) = 0.811 |

H(X | Y) = (4/8)(0.811) + (4/8)(0.811) = 0.811 bits

IG = 1 − 0.811 = **0.189 bits**

A feature that split the people perfectly, such as "owns a dog" if every dog owner buys and no one else does, would have IG = 1 bit. A decision tree would choose that feature first.

## Try it yourself

Six people. The target is "likes tea": 3 yes and 3 no. The feature is "morning person?":

| Group | People | Likes tea | Doesn't like tea |
|---|---|---|---|
| Morning person | 3 | 2 | 1 |
| Not a morning person | 3 | 1 | 2 |

Compute IG.

<details>
<summary>Answer</summary>

- H(X) = 1 bit (3 yes, 3 no)
- Each group has the split (2/3, 1/3), so H = 0.918 bits.
- H(X | Y) = (1/2)(0.918) + (1/2)(0.918) = 0.918 bits
- IG = 1 − 0.918 = **0.082 bits**

This feature helps only a little.

</details>

## Variations and common mistakes

- **Forgetting the weights.** Each group's entropy must be weighted by its size.
- **Gain ratio.** Information gain favors features with many values. C4.5 decision trees correct this with the gain ratio, which divides IG by the entropy of the feature itself.
- **Confusing the target and feature.** H(X) uses the target only. Do not compute it from the feature.
- **Expecting a large gain.** Small gains are normal. A useful feature may remove only a fraction of a bit.

## Remember this

- IG = uncertainty before − uncertainty after.
- It is never negative.
- Choose the question that reduces uncertainty the most.

## Going further

- **Mutual information.** Information gain is the same as the mutual information between the target and the feature. See [Mutual Information](mutual_information.md).
- **Decision trees** repeat this choice at every branch.
- **Conditional entropy** is the core of the calculation. See [Conditional Entropy](conditional_entropy.md).
