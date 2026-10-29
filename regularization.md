# Regularization

Regularization is any change to a model or its training that is meant to reduce **generalization error** (error on new data) rather than training error. It trades a little fit on the training set for better behavior on unseen data.

Most explicit regularizers add a penalty to the training loss:

$$
J_{\text{reg}}(\theta) = \underbrace{J(\theta)}_{\text{data loss}} + \lambda\, \underbrace{R(\theta)}_{\text{complexity penalty}}
$$

where $\lambda \ge 0$ controls the strength. Many of the most effective techniques in deep learning, however, are *implicit*: they change the data, the architecture, or the training procedure rather than the loss.

See also: [`loss_functions.md`](loss_functions.md) for the data-loss term $J(\theta)$.

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

Expected test error decomposes roughly as:

$$
\text{error} = \text{bias}^2 + \text{variance} + \text{irreducible noise}
$$

- **High bias (underfitting):** training and validation error are both high. The model is too constrained. *Reduce* regularization or increase capacity.
- **High variance (overfitting):** training error is low but validation error is much higher. The model has memorized noise. *Add* regularization or more data.

Regularization lowers variance at the cost of some bias. The goal is the setting that minimizes **validation** error.

**Diagnose before regularizing.** Plot training and validation curves:

| Training error | Validation error | Diagnosis | Action |
|---|---|---|---|
| High | High | Underfitting | Bigger model, train longer, less regularization |
| Low | High | Overfitting | More data, augmentation, stronger regularization |
| Low | Low | Good fit | Done; check on a held-out test set |

> **Modern caveat:** very large networks can fit random labels perfectly and still generalize well on real data ("benign overfitting", "double descent"). The classical bias–variance picture remains a useful diagnostic, but in deep learning *implicit* regularization from the optimizer, architecture, and data often matters more than explicit penalties.

---

## 2. Parameter-Norm Penalties

### L2 Regularization (Ridge, Weight Decay)

$$
R(\theta) = \tfrac{1}{2}\lVert \theta \rVert_2^2 = \tfrac{1}{2}\sum_j \theta_j^2
$$

The gradient step becomes

$$
\theta \leftarrow \theta - \eta\big(\nabla J(\theta) + \lambda\theta\big) = \underbrace{(1 - \eta\lambda)\,\theta}_{\text{shrink}} - \eta\nabla J(\theta)
$$

so every step multiplies the weights by a factor slightly below 1. Hence the name **weight decay**.

- Shrinks all weights smoothly toward zero; rarely makes any of them exactly zero.
- Prefers spreading weight across many correlated features rather than relying on a few.
- Makes ill-conditioned problems well-posed. Ridge regression has the closed form $\hat{\theta} = (X^\top X + \lambda I)^{-1} X^\top y$, which exists even when $X^\top X$ is singular.
- Shrinks coefficients more along directions where the data has low variance, which are exactly the directions most prone to noise.

#### L2 Penalty vs. Decoupled Weight Decay (AdamW)

For plain SGD, adding $\lambda\theta$ to the gradient and decaying the weights are the same thing. For **adaptive optimizers** they are not: Adam divides the gradient, including the $\lambda\theta$ term, by a running estimate of its scale, so weights with large gradients end up barely regularized.

**AdamW** fixes this by applying decay separately from the adaptive update:

$$
\theta \leftarrow \theta - \eta\left(\frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon} + \lambda\,\theta\right)
$$

Use **AdamW**, not Adam with an L2 term, when training with adaptive optimizers. Typical decay values are $0.01$–$0.1$ for transformers and around $5\times10^{-4}$ for SGD on CNNs.

### L1 Regularization (Lasso)

$$
R(\theta) = \lVert \theta \rVert_1 = \sum_j |\theta_j|
$$

- The gradient $\lambda\operatorname{sign}(\theta_j)$ has constant magnitude, so it can push small weights all the way to zero. The result is a **sparse** model, which doubles as feature selection.
- Not differentiable at zero. Solvers use coordinate descent or **proximal gradient** updates (soft-thresholding):

$$
\theta_j \leftarrow \operatorname{sign}(\tilde{\theta}_j)\,\max\big(|\tilde{\theta}_j| - \eta\lambda,\; 0\big)
$$

  where $\tilde{\theta}$ is the parameter after an ordinary gradient step. Plain SGD with an L1 term hovers around zero rather than landing exactly on it.

- With strongly correlated features, Lasso tends to keep one arbitrarily and drop the others.

### Why L1 Gives Sparsity and L2 Does Not

Regularization is equivalent to minimizing the loss subject to $R(\theta) \le t$. The L2 constraint region is a sphere; the L1 region is a diamond with corners on the axes. The loss contours typically first touch the diamond at a corner, where some coordinates are exactly zero. A sphere has no corners, so the touching point usually has all coordinates nonzero.

### Elastic Net

$$
R(\theta) = \alpha\lVert \theta \rVert_1 + \tfrac{1 - \alpha}{2}\lVert \theta \rVert_2^2
$$

- Sparsity from L1, plus L2's tendency to keep groups of correlated features together.
- A good default for linear models with many correlated features.

### Comparison

| Penalty | Effect on weights | Sparse | Handles correlated features | Typical use |
|---|---|---|---|---|
| L2 | Uniform shrinkage | No | Spreads weight across the group | Default for most models |
| L1 | Some exactly zero | Yes | Picks one, drops the rest | Feature selection, interpretability |
| Elastic net | Shrinkage plus zeros | Yes | Keeps groups together | Many correlated features |

### Other Constraints on Weights

- **Max-norm:** after each update, rescale each unit's incoming weight vector so that $\lVert w \rVert_2 \le c$. Pairs well with dropout and high learning rates.
- **Spectral normalization:** divides each weight matrix by its largest singular value, bounding the layer's Lipschitz constant. Standard in GAN discriminators.
- **Orthogonality penalties:** encourage $W^\top W \approx I$, which keeps signal and gradient norms stable in deep or recurrent networks.

---

## 3. Data-Based Regularization

### Get More (or Better) Data

More representative data is the most reliable regularizer. When it is unavailable, **transfer learning** and **self-supervised pretraining** borrow it from related data, which is usually the biggest single win on small datasets.

### Data Augmentation

Apply label-preserving transformations so the model learns the invariances you already know hold.

| Domain | Common augmentations |
|---|---|
| Images | Random resized crop, horizontal flip, color jitter, rotation, RandAugment / AutoAugment, Random Erasing / Cutout |
| Text | Back-translation, synonym replacement, random token masking or deletion |
| Audio | Time shift, speed and pitch change, background noise, SpecAugment (time and frequency masking) |
| Tabular | Gaussian noise on numeric features, SMOTE for minority-class oversampling |

Only use transformations that truly preserve the label. Flipping a "6" vertically makes a "9", and horizontal flips break text and some medical images.

### Mixup and CutMix

**Mixup** trains on convex combinations of pairs of examples and their labels:

$$
\tilde{x} = \lambda x_i + (1 - \lambda)x_j, \qquad \tilde{y} = \lambda y_i + (1 - \lambda)y_j, \qquad \lambda \sim \operatorname{Beta}(\alpha, \alpha)
$$

**CutMix** pastes a rectangular patch from one image into another and mixes labels in proportion to the patch area. Both encourage linear behavior between training examples and improve calibration. Typical values are $\alpha = 0.2$ for mixup and $\alpha = 1.0$ for CutMix.

### Label Smoothing

Replace one-hot targets with

$$
y_c = (1 - \varepsilon)\,\mathbb{1}[c = k] + \frac{\varepsilon}{C}
$$

- Keeps the model from driving logits to infinity to reach probability 1, which reduces overconfidence.
- $\varepsilon = 0.1$ is a common default.
- Can hurt knowledge distillation, because the teacher's softened outputs carry less information about class similarity.

### Noise Injection

- **Input noise:** adding small Gaussian noise to inputs is, to first order, equivalent to penalizing the model's sensitivity to its inputs (a Tikhonov-style penalty).
- **Weight noise:** perturbing weights during training favors flat minima.
- **Label noise robustness:** for datasets with noisy labels, techniques like co-teaching or robust losses (e.g. generalized cross-entropy) act as a form of regularization.

---

## 4. Architecture-Based Regularization

### Dropout

During training each unit is kept with probability $1 - p$ and zeroed otherwise. With **inverted dropout** (what frameworks implement), the kept units are scaled up so the expected activation is unchanged:

$$
\tilde{h}_i = \frac{m_i\, h_i}{1 - p}, \qquad m_i \sim \operatorname{Bernoulli}(1 - p)
$$

At inference dropout is turned off and no rescaling is needed. **Remember `model.eval()` in PyTorch.**

- Prevents co-adaptation, since no unit can rely on any specific other unit being present.
- Approximates averaging an exponential number of weight-sharing sub-networks.
- Typical rates: 0.1 in transformers, 0.2–0.5 in fully connected layers, often lower or none in CNNs that use batch normalization.

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

| Layer | Normalizes over | Notes |
|---|---|---|
| Batch norm | Batch dimension, per channel | Batch statistics add noise that acts as a mild regularizer; behaves differently in train and eval mode; unreliable with small batches |
| Layer norm | Features within one example | Standard in transformers; no batch dependence |
| Group norm | Channel groups within one example | CNNs with small batch sizes |

Normalization's main job is stabilizing optimization. Any regularization effect is secondary and usually weaker than dropout or weight decay.

### Architectural Priors

Building assumptions into the architecture shrinks the space of functions the model can represent:

- **Convolutions** share weights across positions (translation equivariance).
- **Pooling** adds local translation invariance.
- **Recurrent layers** share weights across time steps.
- **Graph networks** share weights across nodes and respect permutation symmetry.
- **Bottlenecks and low-rank layers** limit capacity directly.

The right prior often beats any explicit penalty, which is why CNNs outperform equally sized fully connected networks on images.

---

## 5. Training-Procedure Regularization

### Early Stopping

Track validation performance during training. Stop after it fails to improve for a set number of evaluations (the *patience*), and restore the best checkpoint.

- Cheap and nearly always worth using.
- For a quadratic loss and gradient descent, early stopping is approximately equivalent to L2 regularization, with the number of steps playing the role of $1/\lambda$.
- Needs a validation set that the stopping rule is allowed to see, so report final results on a separate test set.

### Ensembling and Weight Averaging

- **Ensembles** of independently trained models reduce variance and nearly always improve accuracy and calibration, at a multiple of the inference cost.
- **Stochastic Weight Averaging (SWA)** averages weights from late in training, giving a single model that sits in a flatter region of the loss.
- **Exponential Moving Average (EMA)** of weights does something similar continuously; it is standard in diffusion models and self-supervised learning.

### Optimizer Effects

- **Small batches and SGD noise** tend to steer toward flatter minima, which often generalize better.
- **Learning-rate schedules** such as warmup followed by cosine decay interact with regularization: higher learning rates add implicit regularization.
- **Sharpness-Aware Minimization (SAM)** explicitly minimizes the worst-case loss in a small neighborhood of the weights, seeking flat minima at roughly twice the compute per step.

### Gradient Clipping

Rescales gradients when their norm exceeds a threshold. Its role is stability (preventing exploding gradients in RNNs and transformers) rather than generalization, but it is usually configured together with the regularizers above.

---

## 6. Bayesian and Probabilistic View

Maximum a posteriori (MAP) estimation minimizes

$$
-\log p(\theta \mid \mathcal{D}) = \underbrace{-\log p(\mathcal{D} \mid \theta)}_{\text{data loss (NLL)}}\; \underbrace{-\log p(\theta)}_{\text{regularizer}} + \text{const}
$$

so a regularizer is a **prior belief about the weights**:

| Prior on $\theta$ | Equivalent penalty |
|---|---|
| Gaussian $\mathcal{N}(0, \sigma^2)$ | L2 with $\lambda = 1/\sigma^2$ (scaled by the noise variance) |
| Laplace | L1 |
| Mixture of Laplace and Gaussian | Elastic net |
| Spike-and-slab | Exact sparsity / variable selection |

Going beyond a single MAP estimate:

- **Bayesian neural networks** keep a distribution over weights and average predictions over it. They give principled uncertainty but are expensive.
- **Monte Carlo dropout** leaves dropout on at test time and averages $T$ forward passes. The spread of predictions approximates model uncertainty.
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
for name, p in model.named_parameters():
    if not p.requires_grad:
        continue
    (no_decay if p.ndim == 1 or name.endswith(".bias") else decay).append(p)

optimizer = torch.optim.AdamW(
    [{"params": decay, "weight_decay": 0.05},
     {"params": no_decay, "weight_decay": 0.0}],
    lr=3e-4,
)

# Dropout inside a model
mlp = nn.Sequential(
    nn.Linear(512, 256), nn.ReLU(), nn.Dropout(p=0.3),
    nn.Linear(256, 10),
)

# Explicit L1 penalty (e.g. for sparsity)
l1 = sum(p.abs().sum() for p in decay)
loss = criterion(outputs, targets) + 1e-5 * l1

# Label smoothing
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

# Mixup
def mixup(x, y, alpha=0.2):
    lam = torch.distributions.Beta(alpha, alpha).sample().item()
    idx = torch.randperm(x.size(0), device=x.device)
    return lam * x + (1 - lam) * x[idx], y, y[idx], lam

x_mix, y_a, y_b, lam = mixup(x, y)
out = model(x_mix)
loss = lam * criterion(out, y_a) + (1 - lam) * criterion(out, y_b)

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
                     kernel_regularizer=regularizers.l2(1e-4))
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
4. **Search strengths on a log scale**, e.g. weight decay in $\{10^{-5}, 10^{-4}, 10^{-3}, 10^{-2}, 10^{-1}\}$, then refine.
5. **Stop when validation error stops improving**, and confirm on a held-out test set that was never used for tuning.

### Pitfalls

- **Unscaled features with L1/L2.** The penalty treats every coefficient the same, so a feature measured in large units gets regularized more. Standardize features first for linear models.
- **Decaying biases and normalization parameters.** Usually unnecessary and sometimes harmful. Exclude them, as in the PyTorch example.
- **Adam + L2 instead of AdamW.** Weight decay is weakened for exactly the parameters that most need it.
- **Forgetting eval mode.** Leaving dropout or batch norm in training mode at inference gives noisy, worse predictions.
- **Tuning on the test set.** Any regularization choice made by looking at test performance leaks information and inflates the reported score.
- **Over-regularizing.** If training error is high, more regularization makes things worse. Reduce it, enlarge the model, or train longer.
- **Stacking everything.** Strong augmentation, high dropout, heavy weight decay, and label smoothing together can underfit. Dropout before batch norm can also cause a train–inference mismatch in batch statistics.
