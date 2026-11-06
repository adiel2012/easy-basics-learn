# Regularization

**Regularization** is anything you change in a model or its training to make it do better on **new data**, even if that costs a little accuracy on the training data.

Why accept a worse fit on the training data? Because training data contains noise: random quirks of this particular sample that will not show up again. A flexible enough model can memorize those quirks, and memorized noise only hurts on new data. Regularization holds the model back from memorizing.

There are two families:

- **Explicit** regularization adds a penalty to the training objective, so the model is charged for being complex (sections 2 and 6).
- **Implicit** regularization changes the data, the architecture, or the training procedure instead (sections 3 to 5). In deep learning these are often the most effective.

The explicit form looks like this:

$$
J(\theta) = \underbrace{J_{\text{data}}(\theta)}_{\text{data loss}} + \lambda\thinspace \underbrace{R(\theta)}_{\text{complexity penalty}}
$$

- The **data loss** $J_{\text{data}}(\theta)$ measures how badly the model fits the training data.
- The **penalty** $R(\theta)$ is large when the parameters look "too complex", for example when the weights are very large.
- The **strength** $\lambda \ge 0$ sets the balance. $\lambda = 0$ means no regularization; a larger $\lambda$ pushes harder toward a simple model.

See also: [`loss_functions.md`](loss_functions.md) for the data-loss term $J_{\text{data}}(\theta)$, the average of a per-example loss $L$.

### Notation

Symbols follow [`loss_functions.md`](loss_functions.md#notation) and keep one meaning throughout this file:

| Symbol | Meaning |
|---|---|
| $\theta$, $\theta_j$ | Model parameters (the weights) and one parameter $j$ |
| $J(\theta)$ | Full training objective: data loss plus penalty |
| $J_{\text{data}}(\theta)$ | Data loss alone |
| $R(\theta)$ | Penalty (the regularizer) |
| $\lambda$ | Penalty strength |
| $\eta$ | Learning rate (step size) |
| $x$, $y$, $X$ | Input, target, and the matrix of all inputs (one row per example) |
| $i$, $i'$ | Example indices |
| $\mathcal{D}$ | The training data |
| $\tilde{x}$, $\tilde{h}$, $\tilde{\theta}$ | Modified or intermediate values: mixed inputs, dropped-out activations, a weight before soft-thresholding |
| $s^2$ | A variance ($\sigma$ is reserved for the sigmoid) |
| $\varepsilon$ | A small constant: the label-smoothing amount, or Adam's numerical stabilizer |
| $r$ | Dropout rate (`p` in `nn.Dropout`) |
| $B$ | An upper bound (a penalty budget, or a max-norm limit) |

Symbols used by a single technique are defined where they appear: $\rho$ (elastic-net mix), $\omega$ and $\alpha$ (mixup weight and its Beta parameter), $\hat{m}_t$ and $\hat{v}_t$ (Adam's moment estimates at step $t$), and $M$ (Monte Carlo dropout passes). In code, `lam` is $\lambda$, `lr` is $\eta$, `z` are logits, `target` holds class indices, and `param` loops over parameters.

## Contents

1. [Why regularize: bias, variance, and overfitting](#1-why-regularize-bias-variance-and-overfitting)
2. [Parameter-norm penalties](#2-parameter-norm-penalties)
3. [Data-based regularization](#3-data-based-regularization)
4. [Architecture-based regularization](#4-architecture-based-regularization)
5. [Training-procedure regularization](#5-training-procedure-regularization)
6. [Bayesian and probabilistic view](#6-bayesian-and-probabilistic-view)
7. [Regularization in classical models](#7-regularization-in-classical-models)
8. [Code reference](#8-code-reference)
9. [Choosing and tuning regularization](#9-choosing-and-tuning-regularization)

---

## 1. Why Regularize: Bias, Variance, and Overfitting

A model can fail on new data in two opposite ways:

- **Underfitting (high bias):** the model is too simple to capture the real pattern, like fitting a straight line to a curve. It does badly on the training data *and* on new data.
- **Overfitting (high variance):** the model is so flexible that it fits the noise too, like a curve that passes through every training point and wiggles wildly in between. It does very well on the training data but badly on new data.

Expected test error splits roughly into three parts:

$$
\text{error} = \text{bias}^2 + \text{variance} + \text{irreducible noise}
$$

The last part is noise in the targets themselves, and no model can remove it. Regularization trades the other two: it **lowers variance at the cost of a little bias**. The right amount is whatever gives the lowest error on a **validation set**, data held out from training.

**Diagnose before regularizing.** Compare training and validation error:

| Training error | Validation error | Diagnosis | Action |
|---|---|---|---|
| High | High | Underfitting | Bigger model, train longer, less regularization |
| Low | High | Overfitting | More data, augmentation, stronger regularization |
| Low | Low | Good fit | Done; check on a held-out test set |

Regularization only helps in the second row. In the first row it makes things worse.

> **Modern caveat:** very large networks can memorize completely random labels and still generalize well on real data ("benign overfitting", "double descent"). The bias–variance picture is still a useful diagnostic, but in deep learning the optimizer, the architecture, and the data often regularize more than explicit penalties do.

---

## 2. Parameter-Norm Penalties

These penalties make training pay for large weights. A **norm** is a way of measuring the size of the weight vector $\theta$. L2 and L1 measure size differently, and that difference changes how the weights end up behaving.

### L2 Regularization (Ridge, Weight Decay)

**The idea.** Large weights let a model fit noise: a tiny change in the input can swing the output a lot. L2 regularization charges a cost for large weights, so the model uses a big weight only when it buys a real drop in the data loss.

**Step 1: the penalty.** Add up the squares of all the weights $\theta_1, \dots, \theta_n$:

$$
R(\theta) = \tfrac{1}{2}\sum_j \theta_j^2 = \tfrac{1}{2}\lVert \theta \rVert_2^2
$$

$\lVert \theta \rVert_2$ is the ordinary length of the weight vector, so $R$ is half its squared length. Squaring makes big weights cost far more than small ones: a weight of 3 costs 4.5, while a weight of 0.3 costs only 0.045. The $\tfrac{1}{2}$ is there only to make the derivative in step 3 clean.

**Step 2: the objective.** Training minimizes the data loss $J_{\text{data}}(\theta)$ plus the penalty, scaled by the strength $\lambda \ge 0$:

$$
J(\theta) = J_{\text{data}}(\theta) + \frac{\lambda}{2}\sum_j \theta_j^2
$$

With $\lambda = 0$ there is no regularization. A larger $\lambda$ pushes harder toward small weights.

**Step 3: the gradient.** The derivative of the penalty for one weight is $\frac{\partial}{\partial \theta_j}\big(\tfrac{\lambda}{2}\theta_j^2\big) = \lambda\theta_j$. So the penalty simply adds $\lambda\theta$ to the usual gradient:

$$
\nabla J(\theta) = \nabla J_{\text{data}}(\theta) + \lambda\theta
$$

This extra term is proportional to the weight itself. Gradient descent moves against the gradient, so it pulls every weight back toward zero, and pulls large weights hardest.

**Step 4: the update.** A gradient-descent step with learning rate $\eta$ (the step size) is $\theta \leftarrow \theta - \eta\thinspace\nabla J(\theta)$:

$$
\theta \leftarrow \theta - \eta\big(\nabla J_{\text{data}}(\theta) + \lambda\theta\big)
$$

Collect the two terms that contain $\theta$:

$$
\theta \leftarrow \underbrace{(1 - \eta\lambda)\thinspace\theta}_{\text{shrink}} - \underbrace{\eta\nabla J_{\text{data}}(\theta)}_{\text{usual step}}
$$

So each step does two things: it multiplies every weight by $1 - \eta\lambda$, a number slightly below 1, and then takes the ordinary data-loss step. That shrinking is why L2 regularization is also called **weight decay**.

**A worked example.** With $\eta = 0.1$ and $\lambda = 0.01$, the shrink factor is $1 - 0.1 \times 0.01 = 0.999$, so each step removes 0.1% of every weight. A weight that the data loss never pushes on decays like this:

| Steps | 0 | 100 | 1,000 | 5,000 |
|---|---|---|---|---|
| Weight, starting at 1.0 | 1.0 | 0.905 | 0.368 | 0.0067 |

A weight the data does need is pushed back up by $\nabla J_{\text{data}}$. It settles where the two pulls cancel, at $\nabla J_{\text{data}}(\theta) = -\lambda\theta$: big enough to help the fit, but no bigger.

The same update written two ways in PyTorch; both give identical weights:

```python
def l2_penalty(params):
    return 0.5 * sum((param ** 2).sum() for param in params)

# 1. Add the penalty to the loss and let autograd produce the extra lam * w
loss = data_loss + lam * l2_penalty(model.parameters())

# 2. Or shrink the weights directly, using the gradient of the data loss only
with torch.no_grad():
    for param in model.parameters():
        param.mul_(1 - lr * lam)   # shrink:     (1 - eta * lambda) * theta
        param.sub_(lr * param.grad)  # usual step: - eta * grad J_data
```

**Effects:**

- Shrinks all weights smoothly toward zero; rarely makes any of them exactly zero.
- Prefers spreading weight across many correlated features rather than relying on a few.
- Makes ill-conditioned problems well-posed. Ridge regression has the closed form $\hat{\theta} = (X^\top X + \lambda I)^{-1} X^\top y$, which exists even when $X^\top X$ is singular.
- Shrinks coefficients more along directions where the data has low variance, which are exactly the directions most prone to noise.

#### L2 Penalty vs. Decoupled Weight Decay (AdamW)

For plain gradient descent, the two forms in the code above are exactly the same thing. For adaptive optimizers such as Adam, they are not.

**The problem.** Adam gives every weight its own step size: it divides each weight's gradient by a running estimate of how large that gradient usually is. If $\lambda\theta$ is part of the gradient, it gets divided too. A weight with large gradients then receives only a tiny decay, which is the opposite of what we want.

**The fix: AdamW.** Keep the decay outside Adam's division:

$$
\theta \leftarrow \theta - \eta\left(\frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \varepsilon} + \lambda\thinspace\theta\right)
$$

- The first term is Adam's usual step. $\hat{m}_t$ and $\hat{v}_t$ are running averages of the gradient and the squared gradient at step $t$, computed from $\nabla J_{\text{data}}$ only, and $\varepsilon$ prevents division by zero.
- The second term is the plain decay from the L2 section, so every weight shrinks at the same rate.

Use **AdamW**, not Adam with an L2 term, when training with adaptive optimizers. Typical decay values are $0.01$–$0.1$ for transformers and around $5\times10^{-4}$ for SGD on CNNs.

### L1 Regularization (Lasso)

**The idea.** Penalize the absolute values of the weights instead of their squares:

$$
R(\theta) = \lVert \theta \rVert_1 = \sum_j |\theta_j|
$$

**Why it behaves differently from L2.** Compare how hard each penalty pulls a weight toward zero:

| Penalty on one weight | Its gradient | Pull toward zero as $\theta_j$ gets small |
|---|---|---|
| L2: $\tfrac{\lambda}{2}\theta_j^2$ | $\lambda\theta_j$ | Gets weaker, and vanishes at zero |
| L1: $\lambda\lvert\theta_j\rvert$ | $\lambda\thinspace\mathrm{sign}(\theta_j)$ | Stays the same size |

Under L2 the pull fades as a weight shrinks, so the weight approaches zero but rarely reaches it. Under L1 the pull never fades, so weights the data loss does not need are pushed all the way to **exactly zero**. The result is a **sparse** model, one where many weights are exactly 0, which also works as automatic feature selection.

**Getting exact zeros in practice.** $\lvert\theta_j\rvert$ has a sharp corner at zero (no derivative there), so plain gradient descent just hops back and forth around 0. Solvers instead apply **soft-thresholding** after each ordinary gradient step:

$$
\theta_j \leftarrow \mathrm{sign}(\tilde{\theta}_j)\thinspace\max\big(|\tilde{\theta}_j| - \eta\lambda,\ 0\big)
$$

where $\tilde{\theta}_j$ is the weight after an ordinary step on the data loss. In words: move the weight $\eta\lambda$ closer to zero, and if that would carry it past zero, set it to exactly zero. With $\eta\lambda = 0.05$, a weight of $0.3$ becomes $0.25$, while a weight of $0.03$ becomes $0$.

- With strongly correlated features, Lasso tends to keep one of them (somewhat arbitrarily) and drop the rest.

### Why L1 Gives Sparsity and L2 Does Not

Picture a model with just two weights. Adding a penalty is equivalent to giving the weights a **budget**: minimize the data loss while keeping $R(\theta) \le B$.

- The L2 budget, $\theta_1^2 + \theta_2^2 \le$ constant, is a **circle**.
- The L1 budget, $\lvert\theta_1\rvert + \lvert\theta_2\rvert \le$ constant, is a **diamond** with its corners on the axes.

The best allowed point is where the data loss's contour lines first touch the shape. A diamond sticks out furthest at its corners, so the contours usually touch it there, and at a corner one of the weights is exactly zero. A circle has no corners, so the touching point usually has both weights nonzero.

### Elastic Net

**The idea.** Use both penalties at once:

$$
R(\theta) = \rho\lVert \theta \rVert_1 + \tfrac{1 - \rho}{2}\lVert \theta \rVert_2^2
$$

- $\rho \in [0, 1]$ mixes the two: $\rho = 1$ is pure L1 and $\rho = 0$ is pure L2 (scikit-learn calls it `l1_ratio`).
- You get L1's exact zeros, plus L2's habit of keeping groups of correlated features together instead of picking just one.
- A good default for linear models with many correlated features.

### Comparison

| Penalty | Effect on weights | Sparse | Handles correlated features | Typical use |
|---|---|---|---|---|
| L2 | Uniform shrinkage | No | Spreads weight across the group | Default for most models |
| L1 | Some exactly zero | Yes | Picks one, drops the rest | Feature selection, interpretability |
| Elastic net | Shrinkage plus zeros | Yes | Keeps groups together | Many correlated features |

### Other Constraints on Weights

These limit the weights directly instead of adding a penalty:

- **Max-norm:** after each update, if a unit's incoming weight vector $w$ is longer than a bound $B$ (often 3 to 4), scale it back down so that $\lVert w \rVert_2 = B$. Pairs well with dropout and high learning rates.
- **Spectral normalization:** divides each weight matrix by its largest singular value, the most it can stretch any input. This caps how much the layer can amplify a small change in its input. Standard in GAN discriminators.
- **Orthogonality penalties:** push a weight matrix $W$ toward $W^\top W \approx I$, meaning the layer rotates its input without stretching or squashing it. This keeps signals and gradients from blowing up or fading away in very deep or recurrent networks.

---

## 3. Data-Based Regularization

### Get More (or Better) Data

Overfitting means memorizing the quirks of a small sample. With more representative data the quirks average out, so more data is the most reliable cure. When you cannot get more, borrow it: **transfer learning** starts from a model already trained on a large related dataset, and **self-supervised pretraining** learns useful features from unlabeled data. On small datasets this is usually the biggest single win.

### Data Augmentation

**The idea.** Create new training examples by changing existing ones in ways that do not change the correct answer. A cat photo flipped left to right is still a cat, so training on flipped copies teaches the model that the flip does not matter.

| Domain | Common augmentations |
|---|---|
| Images | Random resized crop, horizontal flip, color jitter, rotation, RandAugment / AutoAugment, Random Erasing / Cutout |
| Text | Back-translation, synonym replacement, random token masking or deletion |
| Audio | Time shift, speed and pitch change, background noise, SpecAugment (time and frequency masking) |
| Tabular | Gaussian noise on numeric features, SMOTE for minority-class oversampling |

**Check that the label really stays the same.** Flipping a "6" upside down makes a "9", and mirroring text or some medical images changes their meaning.

### Mixup and CutMix

**The idea.** Blend two training examples, and blend their labels by the same amount:

$$
\tilde{x} = \omega x_i + (1 - \omega)x_{i'}, \qquad \tilde{y} = \omega y_i + (1 - \omega)y_{i'}, \qquad \omega \sim \mathrm{Beta}(\alpha, \alpha)
$$

- $x_i$ and $x_{i'}$ are two random training examples, and $y_i$, $y_{i'}$ their one-hot labels.
- $\omega \in [0, 1]$ is the mixing weight, drawn fresh for every batch. The mixup paper calls it $\lambda$; here $\lambda$ is reserved for the penalty strength.
- $\alpha$ controls how strong the mixes are. A small $\alpha$ such as 0.2 usually gives $\omega$ close to 0 or 1, so most mixes are mild.

**Example.** With $\omega = 0.7$, mixing a cat image (label $[1, 0]$) with a dog image (label $[0, 1]$) gives an image that is 70% cat and 30% dog, with target $[0.7, 0.3]$.

**CutMix** pastes a rectangle from one image onto another and mixes the labels in proportion to area: a patch covering 25% of the image gives a target that is 75% the first class and 25% the second.

Both teach the model to change smoothly between training examples instead of jumping sharply, which also tends to make its probabilities better calibrated. Typical values are $\alpha = 0.2$ for mixup and $\alpha = 1.0$ for CutMix.

### Label Smoothing

**The idea.** Do not ask the model to be 100% sure. Lower the target probability of the true class slightly and spread the difference over all classes:

$$
y_c = (1 - \varepsilon)\thinspace\mathbb{1}[c = k] + \frac{\varepsilon}{C}
$$

Here $k$ is the true class, $\mathbb{1}[c = k]$ is 1 for the true class and 0 for the others, and $C$ is the number of classes. With $C = 4$ and $\varepsilon = 0.1$, the target $[0, 1, 0, 0]$ becomes $[0.025, 0.925, 0.025, 0.025]$.

- A one-hot target can only be reached with infinitely large logits, so the model keeps growing more confident. The smoothed target can be reached, which reduces overconfidence. The gradient details are in [`loss_functions.md`](loss_functions.md#categorical-cross-entropy).
- $\varepsilon = 0.1$ is a common default.
- Can hurt knowledge distillation, because the teacher's softened outputs then carry less information about which classes are similar.

### Noise Injection

**The idea.** Add a little random noise during training, so the model cannot depend on exact values.

- **Input noise:** adding small Gaussian noise to the inputs is, to a first approximation, the same as penalizing how sharply the output changes when the input changes slightly (a *Tikhonov* penalty).
- **Weight noise:** if the loss stays low even when the weights are jiggled, the model sits in a "flat" region of the loss, and flat solutions tend to generalize better.
- **Noisy labels:** when some labels are wrong, methods such as *co-teaching* (two networks that each train the other only on the examples it finds easy) or robust losses (e.g. generalized cross-entropy) keep the model from memorizing the mistakes.

---

## 4. Architecture-Based Regularization

### Dropout

**The idea.** During training, randomly switch off some of the units on every step. Because the network can never rely on any single unit, it learns redundant, more robust features.

Each unit is zeroed with probability $r$ (the dropout rate) and kept with probability $1 - r$. With **inverted dropout** (what frameworks implement), the kept units are scaled up so the average activation stays the same:

$$
\tilde{h}_i = \frac{\xi_i\thinspace h_i}{1 - r}, \qquad \xi_i \sim \mathrm{Bernoulli}(1 - r)
$$

where $h_i$ is unit $i$'s activation and $\xi_i$ is 1 if the unit is kept and 0 if it is dropped.

**Example.** With $r = 0.5$, about half the units are dropped on each step and the survivors are doubled (divided by $0.5$), so the layer's average output does not change.

At inference dropout is turned off and nothing is rescaled. **Remember `model.eval()` in PyTorch**; otherwise dropout keeps running at prediction time.

- **Why it works:** units cannot *co-adapt*, that is, depend on specific partner units always being present. It is also roughly like averaging a huge number of smaller networks that share their weights.
- **Typical rates $r$:** 0.1 in transformers, 0.2–0.5 in fully connected layers, often lower or none in CNNs that use batch normalization.

**Variants:**

| Variant | Drops | Use |
|---|---|---|
| DropConnect | Individual weights | Fully connected layers |
| Spatial dropout | Whole feature maps | CNNs, where neighboring pixels are correlated |
| DropBlock | Contiguous regions of a feature map | CNNs |
| Variational / recurrent dropout | Same mask at every time step | RNNs |
| Attention dropout | Attention weights | Transformers |
| Stochastic depth (DropPath) | Entire residual blocks | Deep ResNets, vision transformers |

### Normalization Layers

These rescale a layer's activations to mean 0 and variance 1:

| Layer | Normalizes over | Notes |
|---|---|---|
| Batch norm | Batch dimension, per channel | Each batch's mean and variance differ a little, which adds noise that acts as a mild regularizer; behaves differently in train and eval mode; unreliable with small batches |
| Layer norm | Features within one example | Standard in transformers; no batch dependence |
| Group norm | Channel groups within one example | CNNs with small batch sizes |

Normalization's main job is making optimization stable. Its regularization effect is a side effect, and usually weaker than dropout or weight decay.

### Architectural Priors

A **prior** here is an assumption about the problem that is built into the architecture. Each assumption removes freedom the model would otherwise have to overfit:

- **Convolutions** apply the same small filter at every position, because a pattern such as an edge means the same thing anywhere in an image (translation equivariance).
- **Pooling** summarizes small neighborhoods, so shifting the input by a pixel or two barely changes the output (local translation invariance).
- **Recurrent layers** use the same weights at every time step.
- **Graph networks** use the same weights at every node and give the same answer however the nodes are numbered (permutation symmetry).
- **Bottlenecks and low-rank layers** limit capacity directly by forcing information through a narrow layer.

The right prior often beats any explicit penalty, which is why CNNs outperform fully connected networks of the same size on images.

---

## 5. Training-Procedure Regularization

### Early Stopping

**The idea.** As training goes on, validation error usually falls, reaches a minimum, and then rises again as the model starts memorizing. Stop at the minimum.

How: evaluate on validation data at regular intervals and save the best checkpoint so far. If validation has not improved for a set number of evaluations (the *patience*), stop and restore the best checkpoint.

- Cheap and nearly always worth using.
- For a quadratic loss and gradient descent, early stopping is approximately the same as L2: the weights do not have time to grow large, and the number of steps plays the role of $1/\lambda$ (fewer steps act like stronger regularization).
- The validation set is now used to decide when to stop, so report final results on a separate test set.

### Ensembling and Weight Averaging

- **Ensembles:** train several models independently and average their predictions. Their mistakes partly cancel out, which nearly always improves accuracy and calibration, at a multiple of the inference cost.
- **Stochastic Weight Averaging (SWA):** average the weights saved at several points late in training. You get a single model that sits in a flatter region of the loss.
- **Exponential Moving Average (EMA):** keep a running average of the weights during training, for example $\bar{\theta} \leftarrow 0.999\thinspace\bar{\theta} + 0.001\thinspace\theta$ after every step, and use $\bar{\theta}$ for evaluation. Standard in diffusion models and self-supervised learning.

### Optimizer Effects

- **Small batches:** a gradient computed from few examples is noisy, and that noise keeps training from settling into sharp, narrow minima. Flatter minima often generalize better.
- **Learning-rate schedules** such as warmup followed by cosine decay interact with regularization: a higher learning rate adds more noise, and therefore more implicit regularization.
- **Sharpness-Aware Minimization (SAM):** on each step, first find the nearby weights where the loss is worst, then update to reduce the loss *there*. This favors flat regions, at roughly twice the compute per step.

### Gradient Clipping

If the gradient's norm exceeds a threshold, scale the gradient down to that threshold. Its job is stability (preventing exploding gradients in RNNs and transformers) rather than generalization, but it is usually configured together with the regularizers above.

---

## 6. Bayesian and Probabilistic View

**The idea.** A penalty can be read as a **prior**: a belief about the weights held *before* seeing any data. The L2 penalty, for example, says "I expect the weights to be small, scattered around zero like a bell curve."

Formally, **maximum a posteriori (MAP)** estimation picks the most probable weights given the data $\mathcal{D}$. By Bayes' rule, it minimizes

$$
-\log p(\theta \mid \mathcal{D}) = \underbrace{-\log p(\mathcal{D} \mid \theta)}_{\text{data loss (NLL)}}\ \underbrace{-\log p(\theta)}_{\text{regularizer}} + \text{const}
$$

- The first term asks how well the weights explain the data; it is the negative log-likelihood, the data loss.
- The second term asks how plausible the weights were to begin with; it is the regularizer.

Each choice of prior gives a familiar penalty:

| Prior on $\theta$ | Equivalent penalty |
|---|---|
| Gaussian $\mathcal{N}(0, s^2)$ | L2 with $\lambda = 1/s^2$ (scaled by the noise variance) |
| Laplace | L1 |
| Mixture of Laplace and Gaussian | Elastic net |
| Spike-and-slab | Exact sparsity / variable selection |

**Why a Gaussian prior gives L2:** the negative log of a Gaussian density is $\theta_j^2 / (2s^2)$ plus a constant, which is exactly the L2 penalty with $\lambda = 1/s^2$. A narrower prior (smaller $s$) means a stronger belief that the weights are small, and therefore stronger regularization.

Going beyond a single MAP estimate:

- **Bayesian neural networks** keep a whole distribution over the weights and average predictions over it. They give principled uncertainty but are expensive.
- **Monte Carlo dropout** leaves dropout on at test time and averages $M$ forward passes. How much the $M$ predictions disagree approximates the model's uncertainty.
- **Deep ensembles** are a strong practical baseline for both accuracy and uncertainty.

---

## 7. Regularization in Classical Models

| Model | Main regularization knobs |
|---|---|
| Linear / logistic regression | L2 or L1 strength. In scikit-learn, `C` is the **inverse** of $\lambda$: smaller `C` means stronger regularization |
| SVM | `C` (inverse strength), kernel width `gamma` |
| Decision trees | `max_depth`, `min_samples_leaf`, `min_samples_split`, cost-complexity pruning (`ccp_alpha`) |
| Random forests | Number and depth of trees, `max_features` per split, bootstrap sampling |
| Gradient boosting (XGBoost, LightGBM) | Learning rate (shrinkage), number of trees with early stopping, `max_depth` / `num_leaves`, `subsample`, `colsample_bytree`, L1/L2 on leaf weights (`reg_alpha`, `reg_lambda`), `min_child_weight` |
| k-nearest neighbors | Larger `k` smooths the decision boundary |
| Polynomial / spline models | Degree, number of knots, smoothness penalty |

---

## 8. Code Reference

### PyTorch

```python
import torch
import torch.nn as nn

# Decoupled weight decay; exclude biases and normalization parameters
decay, no_decay = [], []
for name, param in model.named_parameters():
    if not param.requires_grad:
        continue
    (no_decay if param.ndim == 1 or name.endswith(".bias") else decay).append(param)

optimizer = torch.optim.AdamW(
    [{"params": decay, "weight_decay": 0.05},           # weight_decay is lambda
     {"params": no_decay, "weight_decay": 0.0}],
    lr=3e-4,
)

# Dropout inside a model
mlp = nn.Sequential(
    nn.Linear(512, 256), nn.ReLU(), nn.Dropout(p=0.3),  # p is the dropout rate r
    nn.Linear(256, 10),
)

# Explicit L1 penalty (e.g. for sparsity)
lam = 1e-5
l1 = sum(param.abs().sum() for param in decay)
loss = criterion(z, target) + lam * l1          # J = J_data + lambda * R

# Label smoothing
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

# Mixup
def mixup(x, target, alpha=0.2):
    omega = torch.distributions.Beta(alpha, alpha).sample().item()
    idx = torch.randperm(x.size(0), device=x.device)          # partner i' for each example i
    return omega * x + (1 - omega) * x[idx], target, target[idx], omega

x_mix, target_i, target_i2, omega = mixup(x, target)
z = model(x_mix)
loss = omega * criterion(z, target_i) + (1 - omega) * criterion(z, target_i2)

# Gradient clipping
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

# Switch dropout and batch norm to inference behavior
model.eval()
```

### Early Stopping

```python
best, patience, bad_epochs = float("inf"), 10, 0
for epoch in range(max_epochs):
    train_one_epoch(model)
    val_loss = evaluate(model)
    if val_loss < best - 1e-4:
        best, bad_epochs = val_loss, 0
        torch.save(model.state_dict(), "best.pt")
    else:
        bad_epochs += 1
        if bad_epochs >= patience:
            break
model.load_state_dict(torch.load("best.pt"))
```

### scikit-learn

```python
from sklearn.linear_model import Ridge, Lasso, ElasticNet, LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# scikit-learn's alpha plays the role of lambda (its exact scaling differs per model), l1_ratio is rho
ridge = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
lasso = make_pipeline(StandardScaler(), Lasso(alpha=0.01))
enet  = make_pipeline(StandardScaler(), ElasticNet(alpha=0.01, l1_ratio=0.5))
logit = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, penalty="l2"))
# RidgeCV / LassoCV / ElasticNetCV / LogisticRegressionCV select the strength by cross-validation
```

### Keras

```python
from tensorflow.keras import layers, regularizers, callbacks

layer = layers.Dense(256, activation="relu",
                     kernel_regularizer=regularizers.l2(1e-4))   # adds 1e-4 * sum(w**2): lambda = 2e-4 here
drop = layers.Dropout(0.3)
early = callbacks.EarlyStopping(monitor="val_loss", patience=10,
                                restore_best_weights=True)
```

---

## 9. Choosing and Tuning Regularization

### Starting Points

| Situation | Start with | Add if still overfitting |
|---|---|---|
| Linear model, many features | Ridge (L2) | L1 or elastic net for sparsity |
| Gradient-boosted trees | Small learning rate + early stopping | Lower depth, `subsample` / `colsample` < 1, leaf L2 |
| CNN, image classification | Augmentation (crop, flip), weight decay | Mixup / CutMix, label smoothing, stochastic depth |
| Transformer / language model | AdamW weight decay, dropout ≈ 0.1 | Label smoothing, more data, early stopping |
| Fine-tuning a pretrained model | Low learning rate, few epochs, early stopping | Freeze lower layers, LoRA / adapters, layer-wise learning-rate decay |
| Small dataset of any kind | Transfer learning, strong augmentation | Dropout, ensembling |
| Overconfident probabilities | Label smoothing | Temperature scaling after training, ensembles |

### Tuning Workflow

1. **Establish the model can learn.** With no regularization, confirm it can overfit a small subset of the data. If it cannot, the issue is the model or the pipeline, not regularization.
2. **Train on the full data and read the gap** between training and validation curves.
3. **Add one regularizer at a time**, starting with the cheapest and most reliable: early stopping, then weight decay and augmentation, then dropout.
4. **Search strengths on a log scale**, e.g. weight decay in $\lbrace10^{-5}, 10^{-4}, 10^{-3}, 10^{-2}, 10^{-1}\rbrace$, then refine.
5. **Stop when validation error stops improving**, and confirm on a held-out test set that was never used for tuning.

### Pitfalls

- **Unscaled features with L1/L2.** The penalty treats every coefficient the same, so a feature measured in large units gets regularized more. Standardize features first for linear models.
- **Decaying biases and normalization parameters.** Usually unnecessary and sometimes harmful. Exclude them, as in the PyTorch example.
- **Adam + L2 instead of AdamW.** Weight decay is weakened for exactly the parameters that most need it.
- **Forgetting eval mode.** Leaving dropout or batch norm in training mode at inference gives noisy, worse predictions.
- **Tuning on the test set.** Any regularization choice made by looking at test performance leaks information and inflates the reported score.
- **Over-regularizing.** If training error is high, more regularization makes things worse. Reduce it, enlarge the model, or train longer.
- **Stacking everything.** Strong augmentation, high dropout, heavy weight decay, and label smoothing together can underfit. Dropout before batch norm can also cause a train–inference mismatch in batch statistics.
