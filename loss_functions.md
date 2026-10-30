# Loss Functions

A loss function measures how far a model's prediction is from the target. Training adjusts the parameters $\theta$ to minimize a **training objective**, which is usually the average loss plus an optional regularization penalty:

$$
J(\theta) = \underbrace{\frac{1}{N}\sum_{i=1}^{N} L\big(y_i, f(x_i; \theta)\big)}_{\text{prediction error}} + \underbrace{\lambda\, R(\theta)}_{\text{regularization}}
$$

The first term measures how wrong the predictions are; the second penalizes undesirable parameter configurations (see [`regularization.md`](regularization.md)). This file is about the first term.

The choice of loss defines *what the model is optimizing for*. Minimizing expected squared error yields the conditional mean, minimizing expected absolute error yields a conditional median, and quantile loss yields any chosen quantile. Picking the wrong one means the model optimizes the wrong quantity, however well it trains.

### Loss, Metric, and Regularization

These three are easy to confuse:

| Concept | Purpose | Example |
|---|---|---|
| Loss function | Provides the training objective | Cross-entropy, MSE |
| Evaluation metric | Measures practical performance | Accuracy, F1, AUC |
| Regularization | Penalizes undesirable model complexity | L1, L2 penalties |

A loss must be something the optimizer can work with, which for neural networks usually means differentiable (or at least subdifferentiable). A metric has no such requirement: accuracy is a step function of the parameters, so it gives zero gradient almost everywhere. That is why we train on a differentiable **surrogate** such as cross-entropy and *monitor* the metric we actually care about (see [section 8](#8-practical-notes-and-pitfalls)).

### Where This Fits

This guide completes a short learning sequence:

1. [`mlp_xor_inference.md`](mlp_xor_inference.md): how a neural network makes predictions.
2. [`backpropagation.md`](backpropagation.md): how gradients tell the network how to change its parameters.
3. **Loss functions** (this file): what the network is actually trying to optimize.

The loss is where backpropagation starts: its derivative with respect to the network's output is the first gradient passed backward (see the [worked example](#from-loss-to-gradient-a-worked-example) in section 2).

## Contents

1. [Regression losses](#1-regression-losses)
2. [Classification losses](#2-classification-losses)
3. [Ranking, metric, and contrastive losses](#3-ranking-metric-and-contrastive-losses)
4. [Generative and probabilistic losses](#4-generative-and-probabilistic-losses)
5. [Segmentation and sequence losses](#5-segmentation-and-sequence-losses)
6. [Framework reference](#6-framework-reference)
7. [Choosing a loss function](#7-choosing-a-loss-function)
8. [Practical notes and pitfalls](#8-practical-notes-and-pitfalls)

Notation: $y$ is the target, $\hat{y}$ the prediction, $e = y - \hat{y}$ the residual, $z$ a raw logit, $\sigma$ the sigmoid, and $p$ a predicted probability.

---

## 1. Regression Losses

### Mean Squared Error (MSE, L2 loss)

$$
L = (y - \hat{y})^2 \qquad \frac{\partial L}{\partial \hat{y}} = -2(y - \hat{y})
$$

- **Optimal prediction:** minimizing expected squared error yields the conditional mean $\mathbb{E}[y \mid x]$.
- **Gradient grows with the error**, so large mistakes dominate training. This makes MSE sensitive to outliers.
- Convex and differentiable everywhere in $\hat{y}$; easy to optimize.
- Equivalent to maximum likelihood under Gaussian noise with fixed variance.
- **RMSE** ($\sqrt{\text{MSE}}$) is usually reported instead because it has the same units as $y$.

### Mean Absolute Error (MAE, L1 loss)

$$
L = |y - \hat{y}| \qquad \frac{\partial L}{\partial \hat{y}} = -\operatorname{sign}(y - \hat{y})
$$

- **Optimal prediction:** minimizing expected absolute error yields a conditional median (it need not be unique when the distribution has flat regions).
- Gradient magnitude is constant, so outliers cannot dominate. The downside is that the gradient does not shrink near the optimum, so learning-rate decay helps convergence.
- Not differentiable at $e = 0$; frameworks use a subgradient.
- Equivalent to maximum likelihood under Laplace noise.

### Huber Loss

$$
L_\delta(e) =
\begin{cases}
\tfrac{1}{2}e^2 & |e| \le \delta \\[2pt]
\delta\big(|e| - \tfrac{1}{2}\delta\big) & |e| > \delta
\end{cases}
\qquad
\frac{\partial L}{\partial \hat{y}} = -\operatorname{clip}(e, -\delta, \delta)
$$

- Quadratic near zero and linear in the tails. Combines MSE's smooth behavior near zero with MAE's robustness to large errors.
- Continuously differentiable ($C^1$), but not twice differentiable at the transition points $e = \pm\delta$, where the second derivative jumps from 1 to 0.
- The gradient with respect to the prediction is the negative **clipped** residual. Its magnitude is capped at $\delta$, so large errors cannot produce arbitrarily large gradients.
- $\delta$ should be on the scale of a "normal" residual. A common choice is $\delta = 1$ on standardized targets.
- **Smooth L1** (common in object detection) is Huber divided by $\delta$ (PyTorch calls the threshold `beta`). The two have different scaling conventions, so `HuberLoss(delta=d)` and `SmoothL1Loss(beta=d)` differ by a factor of $d$.

### Log-Cosh Loss

$$
L = \log\cosh(\hat{y} - y) \qquad \frac{\partial L}{\partial \hat{y}} = \tanh(\hat{y} - y)
$$

- Approximately $\tfrac{1}{2}e^2$ for small errors and $|e| - \log 2$ for large ones.
- Like Huber but infinitely differentiable and with no $\delta$ to tune. Useful for second-order methods such as gradient-boosted trees.
- Its transition scale is fixed at $|e| \approx 1$, so it behaves sensibly only when targets are on a sensible scale (standardize them).
- **Optimal prediction:** the value $\hat{y}$ satisfying $\mathbb{E}[\tanh(\hat{y} - Y) \mid X = x] = 0$. This is a robust location estimate; it is generally neither the conditional mean nor the median, nor a fixed blend of the two.

### Quantile (Pinball) Loss

$$
L_\tau(e) = \max\big(\tau e,\; (\tau - 1)e\big), \qquad \tau \in (0, 1)
$$

- **Optimal prediction:** the conditional $\tau$-quantile. With $\tau = 0.5$ it is half the MAE.
- Under-prediction costs $\tau$ per unit and over-prediction costs $1 - \tau$, so a high $\tau$ pushes predictions upward.
- Train at $\tau = 0.05$ and $\tau = 0.95$ to *estimate* a central 90% prediction interval. Check its coverage on validation data: estimated quantiles are not automatically calibrated, so the interval may cover more or less than 90% of unseen targets.
- **Asymmetric costs.** If under-predicting costs $c_u$ per unit and over-predicting costs $c_o$ (e.g. running short on electricity is worse than over-supplying), the cost-minimizing prediction is the $\tau$-quantile with $\tau = c_u / (c_u + c_o)$. Quantile loss *is* the general linear asymmetric loss, so no separate loss is needed.

### Gaussian Negative Log-Likelihood (heteroscedastic regression)

When the model predicts both a mean $\mu$ and a variance $s^2$:

$$
L = \frac{1}{2}\left(\log s^2 + \frac{(y - \mu)^2}{s^2}\right)
$$

- The model learns per-example uncertainty: it can reduce the squared-error term on hard examples by predicting a larger variance, at a cost of $\log s^2$.
- Predict $\log s^2$ (not $s^2$ directly) for numerical stability.

### Losses for Positive and Count Targets

| Loss | Formula | Use |
|---|---|---|
| Poisson | $\hat{y} - y\log\hat{y}$ | Counts (arrivals, clicks) |
| Mean Squared Log Error | $\big(\log(1 + y) - \log(1 + \hat{y})\big)^2$ | Targets spanning orders of magnitude; penalizes relative error |
| MAPE | $\lvert e \rvert / \lvert y \rvert$ | Business reporting; undefined at $y = 0$, see below |
| Tweedie | Family between Poisson and Gamma | Zero-inflated positive targets (insurance claims, sales) |

**Poisson details.** The full Poisson negative log-likelihood is

$$
L = \hat{y} - y\log\hat{y} + \log(y!)
$$

where $\hat{y} > 0$ is the predicted rate and $y$ is a nonnegative integer count. The $\log(y!)$ term depends only on the target, so dropping it does not change the gradients with respect to the model parameters. Implementation conventions differ: some frameworks expect the rate $\hat{y}$ (which must be kept positive, e.g. with a softplus or exp output), others expect the log-rate. PyTorch's `PoissonNLLLoss` defaults to `log_input=True`, i.e. it expects $\log\hat{y}$.

**Why MAPE is tricky.** MAPE weights each error inversely by the size of the target, so a miss of 5 on a target of 10 counts as much as a miss of 500 on a target of 1000. Small actual values can therefore dominate the metric. Over-predictions can produce arbitrarily large percentage errors while under-predictions are capped at 100% (for positive predictions), so optimizing MAPE tends to encourage systematic under-prediction.

### Regression Loss Comparison

| Loss | Predicts | Outlier robust | Smooth | Extra hyperparameter |
|---|---|---|---|---|
| MSE | Mean | No | Yes | None |
| MAE | Median | Yes | No (at 0) | None |
| Huber | Robust location estimate (mean-like for small $\lvert e \rvert$, median-like for large) | Yes | $C^1$ (not twice differentiable at $\pm\delta$) | $\delta$ |
| Log-cosh | Robust location estimate (generally neither mean nor median) | Yes | Infinitely differentiable | None |
| Quantile | $\tau$-quantile | Yes | No (at 0) | $\tau$ |

---

## 2. Classification Losses

### Binary Cross-Entropy (Log Loss)

With label $y \in \{0, 1\}$, logit $z$, and $p = \sigma(z)$:

$$
L = -\big[y \log p + (1 - y)\log(1 - p)\big] \qquad \frac{\partial L}{\partial z} = p - y
$$

- The gradient with respect to the logit is just *prediction minus target*. That simplicity is a large part of why the sigmoid–cross-entropy pairing trains so well.
- Penalizes confident mistakes heavily: predicting $p = 0.01$ for a positive costs $\log 100 \approx 4.6$.
- **Multi-label classification** applies binary cross-entropy independently to each label.

**Stable form from logits** (what frameworks compute internally):

$$
L = \max(z, 0) - z\,y + \log\big(1 + e^{-|z|}\big)
$$

### From Loss to Gradient: A Worked Example

This is where the loss meets [`backpropagation.md`](backpropagation.md). Take one example with target $y = 1$ where the model currently predicts $p = \sigma(z) = 0.2$:

| Quantity | Value |
|---|---|
| Loss | $-\log 0.2 \approx 1.61$ |
| Output gradient $\partial L / \partial z = p - y$ | $0.2 - 1 = -0.8$ |
| Gradient-descent step $z \leftarrow z - \eta\,(-0.8)$ | $z$ increases by $0.8\eta$ |

The negative gradient means *increasing* the logit lowers the loss, so the update pushes $z$ up and $p$ toward 1. Backpropagation then carries this $-0.8$ backward through the network with the chain rule to get the gradient for every weight. The size of the push is proportional to how wrong the prediction was: at $p = 0.9$ the output gradient would be only $-0.1$.

### Weighted Binary Cross-Entropy

$$
L = -\big[w_+\, y \log p + w_-(1 - y)\log(1 - p)\big]
$$

- Up-weight the rare class. A common starting point is $w_+ = N_{\text{neg}} / N_{\text{pos}}$, then tune it.
- In PyTorch this is the `pos_weight` argument of `BCEWithLogitsLoss`.

### Categorical Cross-Entropy

With logits $z \in \mathbb{R}^C$, softmax probabilities $p_c = e^{z_c} / \sum_k e^{z_k}$, and one-hot target $y$:

$$
L = -\sum_{c=1}^{C} y_c \log p_c = -z_k + \log\sum_{c} e^{z_c} \qquad \frac{\partial L}{\partial z_c} = p_c - y_c
$$

where $k$ is the true class.

- The second form (**log-sum-exp**) is how it is computed stably.
- **Sparse** categorical cross-entropy is the same loss with integer class labels instead of one-hot vectors.
- With **label smoothing** $\varepsilon$, the target becomes $y_c = (1 - \varepsilon)\,\mathbb{1}[c = k] + \varepsilon / C$. The model is no longer trained toward perfectly confident predictions, which can improve generalization. It also changes the probability targets themselves, so the trained model's probabilities no longer estimate the true class probabilities and calibration can get better or worse. See [`regularization.md`](regularization.md#label-smoothing).

### Focal Loss

$$
L = -\alpha_t (1 - p_t)^\gamma \log p_t, \qquad p_t = \begin{cases} p & y = 1 \\ 1 - p & y = 0 \end{cases}, \qquad \alpha_t = \begin{cases} \alpha & y = 1 \\ 1 - \alpha & y = 0 \end{cases}
$$

- $p_t$ is the probability assigned to the true class, and $\alpha_t$ balances the positive and negative classes.
- The factor $(1 - p_t)^\gamma$ shrinks the loss on examples the model already gets right. With $\gamma = 2$, an example at $p_t = 0.9$ is down-weighted by 100×.
- Designed for extreme imbalance, such as dense object detection where most anchors are background. Common defaults are $\gamma = 2$ and $\alpha = 0.25$. Note that $\alpha = 0.25$ *down*-weights the rare positives: the focusing term already suppresses the many easy negatives, so $\alpha$ ends up compensating in the other direction. Tune the two together.
- With $\gamma = 0$ it reduces to $\alpha$-weighted cross-entropy; with $\gamma = 0$ and $\alpha = 0.5$ it is plain cross-entropy scaled by $\tfrac{1}{2}$.

### Hinge Loss

With label $y \in \{-1, +1\}$ and raw score $f(x)$:

$$
L = \max\big(0,\; 1 - y f(x)\big)
$$

- The loss of linear SVMs. It is zero once an example is on the correct side with margin at least 1, so only margin violators (the "support vectors") contribute gradient.
- Produces scores, not calibrated probabilities.
- **Squared hinge** $\max(0, 1 - yf(x))^2$ is differentiable and penalizes violations more strongly.
- **Multi-class hinge** (Crammer–Singer): $\max\big(0,\; 1 + \max_{c \ne k} f_c - f_k\big)$.

### KL Divergence

$$
D_{\text{KL}}(P \,\|\, Q) = \sum_x P(x)\log\frac{P(x)}{Q(x)} = H(P, Q) - H(P)
$$

- Cross-entropy minus the target's own entropy. When $P$ is a fixed one-hot label, $H(P) = 0$ and minimizing KL is identical to minimizing cross-entropy.
- KL matters when the target is itself a distribution, e.g. **knowledge distillation**, where a student matches the softened outputs of a teacher at temperature $T$:

$$
L = T^2 \cdot D_{\text{KL}}\Big(\operatorname{softmax}(z^{\text{teacher}}/T) \,\Big\|\, \operatorname{softmax}(z^{\text{student}}/T)\Big)
$$

- Asymmetric: $D_{\text{KL}}(P\|Q) \neq D_{\text{KL}}(Q\|P)$. When fitting an approximation $Q$ to a target $P$, forward KL $D_{\text{KL}}(P\|Q)$ tends to favor covering all the regions where $P$ has mass, while reverse KL $D_{\text{KL}}(Q\|P)$ often favors concentrating on individual modes. These are tendencies in common approximation settings, not universal guarantees.

### Brier Score

MSE applied to predicted probabilities:

$$
L = (p - y)^2 \qquad \text{(multi-class: } L = \textstyle\sum_{c=1}^{C} (p_c - y_c)^2\text{)}
$$

- A **strictly proper scoring rule**: in expectation it is minimized by reporting the true probability, just like log loss.
- Bounded between 0 and 1 (binary case), so it does not penalize confident mistakes as aggressively as log loss. Predicting $p = 0.01$ for a positive costs $0.98$ under Brier but $4.6$ under log loss.
- Mostly used as an **evaluation metric for calibration and probabilistic forecasts**, where its boundedness makes it robust to a few confidently wrong predictions. As a training loss it suffers from the vanishing-gradient problem described next.

### Why Cross-Entropy Instead of MSE for Classification?

With a sigmoid output, the MSE (Brier) gradient with respect to the logit is $2(p - y)\,p(1 - p)$. When the model is confidently wrong, $p(1 - p) \approx 0$ and learning stalls. Cross-entropy's gradient $p - y$ has no such factor, so confidently wrong predictions get the largest updates.

---

## 3. Ranking, Metric, and Contrastive Losses

These losses shape an **embedding space** or a **ranking** rather than predict a label.

### Contrastive (Pairwise) Loss

For an embedding distance $d$ between two items and $y = 1$ if they are similar:

$$
L = y\, d^2 + (1 - y)\max(0,\; m - d)^2
$$

Pulls similar pairs together and pushes dissimilar pairs apart until they are at least margin $m$ away.

### Triplet Loss

$$
L = \max\big(0,\; d(a, p) - d(a, n) + \alpha\big)
$$

- Anchor $a$ must be closer to positive $p$ than to negative $n$ by margin $\alpha$.
- **Mining matters more than the formula.** Random triplets are mostly already satisfied and give zero gradient. Use semi-hard or batch-hard negative mining.

### InfoNCE / NT-Xent (Contrastive Self-Supervised Learning)

For an anchor $q$, one positive $k^+$, and a batch of negatives, with similarity $s(\cdot,\cdot)$ (usually cosine) and temperature $\tau$:

$$
L = -\log \frac{\exp\big(s(q, k^+)/\tau\big)}{\sum_{j} \exp\big(s(q, k_j)/\tau\big)}
$$

- Cross-entropy where the "correct class" is the positive pair. Used in SimCLR, MoCo, and CLIP.
- The number and quality of negatives, along with the temperature $\tau$ (often 0.05–0.2), strongly affect the learned representations. Large batch sizes help because every other item in the batch serves as a negative.

### Pairwise Ranking Losses

| Loss | Formula | Used in |
|---|---|---|
| RankNet / BPR | $-\log \sigma(s_i - s_j)$ for $i$ preferred over $j$ | Recommenders, learning to rank |
| Margin ranking | $\max(0,\; m - (s_i - s_j))$ | Retrieval |
| Bradley–Terry reward loss | $-\log\sigma\big(r(y_w) - r(y_l)\big)$ | Reward models for RLHF |
| LambdaRank / ListNet | Weights pairs by their effect on NDCG / uses whole list | Search ranking |

---

## 4. Generative and Probabilistic Losses

### Negative Log-Likelihood (NLL)

$$
L = -\log p(y \mid x; \theta)
$$

The unifying principle behind most losses in this file. Choosing an output distribution determines the loss:

| Assumed distribution of $y$ | Resulting loss |
|---|---|
| Gaussian, fixed variance | MSE |
| Gaussian, learned variance | Gaussian NLL |
| Laplace | MAE |
| Bernoulli | Binary cross-entropy |
| Categorical | Categorical cross-entropy |
| Poisson | Poisson loss |

Language models are trained with **token-level cross-entropy** (NLL of the next token). **Perplexity** is $\exp(\text{mean NLL})$.

### VAE: Evidence Lower Bound (ELBO)

$$
\mathcal{L}_{\text{VAE}} = \underbrace{-\,\mathbb{E}_{q(z\mid x)}\big[\log p(x \mid z)\big]}_{\text{reconstruction}} + \beta \cdot \underbrace{D_{\text{KL}}\big(q(z \mid x)\,\|\,p(z)\big)}_{\text{latent regularizer}}
$$

- Minimize this (the negative ELBO). $\beta = 1$ is the standard VAE; $\beta > 1$ ($\beta$-VAE) encourages disentangled latents.
- For a Gaussian encoder and standard normal prior, the KL term has a closed form: $\tfrac{1}{2}\sum_j\big(\mu_j^2 + \sigma_j^2 - \log\sigma_j^2 - 1\big)$.

### GAN Losses

**Original (non-saturating) GAN:**

$$
L_D = -\mathbb{E}_{x}\big[\log D(x)\big] - \mathbb{E}_{z}\big[\log(1 - D(G(z)))\big], \qquad L_G = -\mathbb{E}_{z}\big[\log D(G(z))\big]
$$

The generator uses $-\log D(G(z))$ rather than $\log(1 - D(G(z)))$ because the latter has vanishing gradients early in training.

**Wasserstein GAN (WGAN-GP):** the critic $D$ outputs an unbounded score and is kept approximately 1-Lipschitz with a gradient penalty:

$$
L_D = \mathbb{E}_{z}\big[D(G(z))\big] - \mathbb{E}_{x}\big[D(x)\big] + \lambda\,\mathbb{E}_{\hat{x}}\Big[\big(\lVert\nabla_{\hat{x}} D(\hat{x})\rVert_2 - 1\big)^2\Big], \qquad L_G = -\mathbb{E}_{z}\big[D(G(z))\big]
$$

The critic objective approximates the negative Wasserstein-1 (Earth Mover's) distance, while the gradient penalty *encourages* the critic to satisfy the 1-Lipschitz constraint rather than enforcing it exactly. This gives more informative gradients and more stable training than the original objective.

### Diffusion Models

The common simplified objective trains a network $\epsilon_\theta$ to predict the noise added at a random timestep $t$:

$$
L = \mathbb{E}_{x_0, \epsilon, t}\Big[\big\lVert \epsilon - \epsilon_\theta(x_t, t)\big\rVert^2\Big]
$$

This is plain MSE, derived from a weighted variational bound.

---

## 5. Segmentation and Sequence Losses

### Dice Loss

$$
L = 1 - \frac{2\sum_i p_i g_i + \epsilon}{\sum_i p_i + \sum_i g_i + \epsilon}
$$

- $p_i$ are predicted probabilities and $g_i$ ground-truth labels per pixel; $\epsilon$ prevents division by zero.
- Measures overlap directly, so it is insensitive to how many background pixels there are. Useful when the object is small.
- Often combined with cross-entropy (`CE + Dice`) because Dice alone can give unstable gradients.
- **Per-image vs. per-batch.** Dice can be computed separately for each image and then averaged, or once over all pixels in the batch. These are different objectives: per-image Dice gives every image equal weight (including images with tiny objects), while batch Dice is dominated by images with large objects. The PyTorch example in section 6 uses per-image Dice.
- **The smoothing constant matters.** $\epsilon$ makes empty masks well defined (an empty prediction on an empty mask scores a perfect 1), but its value also shifts the loss and gradients. For small objects, a large $\epsilon$ such as 1.0 can noticeably soften the penalty for missing them.

### IoU (Jaccard) and Box-Regression Losses

- **Soft IoU loss:** $1 - \frac{\sum p_i g_i}{\sum p_i + \sum g_i - \sum p_i g_i}$.
- **GIoU, DIoU, CIoU:** variants for bounding boxes that still give gradients when boxes do not overlap and also account for center distance and aspect ratio.

### Tversky Loss

Generalizes Dice by weighting false positives and false negatives separately:

$$
L_{\text{Tversky}} = 1 - \frac{TP + \epsilon}{TP + \alpha\,FP + \beta\,FN + \epsilon}
$$

with soft counts $TP = \sum_i p_i g_i$, $FP = \sum_i p_i (1 - g_i)$, and $FN = \sum_i (1 - p_i) g_i$.

- $\alpha = \beta = 0.5$ recovers Dice; $\alpha = \beta = 1$ gives soft IoU.
- $\beta > \alpha$ penalizes missed foreground more than false alarms, trading precision for recall. This helps when objects are small and easy to miss (e.g. lesions); $\alpha = 0.3, \beta = 0.7$ is a common starting point.

### CTC Loss

Connectionist Temporal Classification sums the probability of every alignment between an input sequence and a shorter label sequence. Used for speech recognition and handwriting/OCR when frame-level alignments are unknown.

### Perceptual Loss

Computes the distance between feature maps of a pretrained network (often VGG) rather than between raw pixels. Produces sharper results than pixel MSE in super-resolution and style transfer.

---

## 6. Framework Reference

| Loss | PyTorch | TensorFlow / Keras |
|---|---|---|
| MSE | `nn.MSELoss` | `losses.MeanSquaredError` |
| MAE | `nn.L1Loss` | `losses.MeanAbsoluteError` |
| Huber | `nn.HuberLoss(delta=...)` | `losses.Huber(delta=...)` |
| Smooth L1 | `nn.SmoothL1Loss(beta=...)` | — |
| Log-cosh | — (one line: `torch.log(torch.cosh(e))`) | `losses.LogCosh` |
| Gaussian NLL | `nn.GaussianNLLLoss` | — |
| Poisson | `nn.PoissonNLLLoss` | `losses.Poisson` |
| Binary CE (from logits) | `nn.BCEWithLogitsLoss` | `losses.BinaryCrossentropy(from_logits=True)` |
| Categorical CE (from logits) | `nn.CrossEntropyLoss` (integer labels or probabilities) | `losses.CategoricalCrossentropy(from_logits=True)` / `SparseCategoricalCrossentropy` |
| Focal | `torchvision.ops.sigmoid_focal_loss` | `losses.BinaryFocalCrossentropy` |
| Hinge | `nn.MultiMarginLoss` | `losses.Hinge` |
| KL divergence | `nn.KLDivLoss` (expects **log-probabilities** as input) | `losses.KLDivergence` |
| Margin ranking | `nn.MarginRankingLoss` | — |
| Triplet | `nn.TripletMarginLoss` | — (TF Addons) |
| Cosine embedding | `nn.CosineEmbeddingLoss` | `losses.CosineSimilarity` |
| CTC | `nn.CTCLoss` | `tf.nn.ctc_loss` |

**Note:** PyTorch's `nn.CrossEntropyLoss` applies log-softmax internally. Pass raw logits, not softmax outputs.

**Similar names do not always mean identical objectives.** Framework implementations can differ in reduction, normalization, target format, and parameter conventions. Some examples from the table above:

- PyTorch `MultiMarginLoss` is a multi-class hinge that *sums* over wrong classes, unlike the Crammer–Singer *max* form; Keras `Hinge` is the binary hinge with targets in $\{-1, +1\}$ (or $\{0, 1\}$, converted internally).
- Keras `CosineSimilarity` returns the negative cosine similarity of two vectors; PyTorch `CosineEmbeddingLoss` takes a $\pm 1$ label per pair and a margin for dissimilar pairs.
- `HuberLoss` and `SmoothL1Loss` differ by a factor of $\delta$ (see section 1).
- PyTorch `KLDivLoss` expects log-probabilities as input, and its default `reduction="mean"` averages over every element rather than per sample; use `reduction="batchmean"` for the mathematical KL.

Check the documentation before swapping one framework's loss for another's.

### Minimal PyTorch Examples

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

# Multi-class classification with label smoothing
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
loss = criterion(logits, targets)             # logits: (N, C), targets: (N,) int64

# Imbalanced binary classification
pos_weight = torch.tensor([n_neg / n_pos])
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
loss = criterion(logits.squeeze(-1), targets.float())

# Quantile loss
def quantile_loss(pred, target, tau):
    e = target - pred
    return torch.mean(torch.maximum(tau * e, (tau - 1) * e))

# Dice loss (binary segmentation)
def dice_loss(logits, target, eps=1.0):
    p = torch.sigmoid(logits).flatten(1)
    g = target.flatten(1)
    inter = (p * g).sum(1)
    return 1 - ((2 * inter + eps) / (p.sum(1) + g.sum(1) + eps)).mean()

# InfoNCE between two batches of paired embeddings
def info_nce(q, k, tau=0.1):
    q, k = F.normalize(q, dim=1), F.normalize(k, dim=1)
    logits = q @ k.T / tau                      # (N, N), positives on the diagonal
    labels = torch.arange(q.size(0), device=q.device)
    return F.cross_entropy(logits, labels)
```

---

## 7. Choosing a Loss Function

| Problem | Default | Consider instead when… |
|---|---|---|
| Regression | MSE | Outliers present → Huber or MAE; need uncertainty → Gaussian NLL |
| Regression over orders of magnitude | MSLE, or MSE on $\log(1 + y)$ | Relative error is what matters |
| Prediction intervals | Quantile loss at two quantiles | Check coverage on validation data; recalibrate if off |
| Asymmetric over/under-prediction costs | Quantile loss with $\tau = c_u / (c_u + c_o)$ | — |
| Count data | Poisson | Many zeros → Tweedie or zero-inflated models |
| Binary classification | BCE with logits | Imbalance → `pos_weight` or focal loss |
| Multi-class classification | Cross-entropy | Overconfidence → label smoothing |
| Multi-label classification | BCE per label | Imbalanced labels → per-label weights or asymmetric loss |
| Semantic segmentation | Cross-entropy | Small objects / imbalance → CE + Dice; recall matters more → Tversky |
| Embedding / retrieval | InfoNCE | Few labels per class → triplet with hard mining |
| Ranking | Pairwise logistic (RankNet) | Optimize NDCG directly → LambdaRank |
| Distillation | KL on softened logits + CE on labels | — |
| Generative modeling | NLL / ELBO / diffusion MSE | Adversarial training → GAN losses |

**Rule of thumb:** start with the loss implied by a sensible probabilistic model of your target (the NLL table in section 4), then change it only when a specific problem shows up.

---

## 8. Practical Notes and Pitfalls

- **Pass logits, not probabilities.** Use `BCEWithLogitsLoss`, `CrossEntropyLoss`, or `from_logits=True`. Applying sigmoid or softmax first and then taking the log loses precision and can produce `NaN` or `inf`.
- **Check the reduction.** `mean` vs `sum` changes the effective learning rate by a factor of the batch size. With masked tokens or padding, average over real elements only.
- **Sanity-check the initial loss.** For $C$ balanced classes, an untrained model should give cross-entropy near $\log C$ (e.g. 2.30 for 10 classes). A very different value usually means a bug in the labels or output layer.
- **Verify that the model can overfit a small batch** as a debugging test. If the loss cannot go to near zero on a single small batch, the problem is in the model or the data pipeline, not the hyperparameters.
- **Scale targets for regression.** Standardizing $y$ keeps MSE gradients in a reasonable range and makes Huber's $\delta$ easier to choose.
- **Combine losses with care.** In multi-task setups ($L = \sum_k w_k L_k$), the terms can differ by orders of magnitude. Log each one separately and tune the weights, or use learned uncertainty weighting.
- **The training loss is not the evaluation metric.** Accuracy, F1, AUC, and NDCG are usually non-differentiable, so we optimize a surrogate. Always monitor the metric you actually care about on a validation set.
- **Calibration differs by loss.** Cross-entropy (and the Brier score) are strictly proper scoring rules: in expectation they reward reporting the true probabilities. That does not guarantee calibration in practice. Large networks often become overconfident, and calibration degrades under model misspecification or distribution shift. Hinge loss produces scores, not probabilities, and focal loss and label smoothing deliberately change the probability targets. If you need reliable probabilities, check calibration on validation data (reliability diagram, Brier score, ECE) and apply temperature scaling after training if needed.
