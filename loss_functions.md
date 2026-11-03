# Loss Functions

A loss function measures how far a model's prediction is from the target. Training adjusts the parameters $\theta$ to minimize a **training objective**, which is usually the average loss plus an optional regularization penalty:

$$
J(\theta) = \underbrace{\frac{1}{N}\sum_{i=1}^{N} L\big(y_i, f(x_i; \theta)\big)}_{\text{prediction error}} + \underbrace{\lambda\thinspace R(\theta)}_{\text{regularization}}
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

Each loss below comes with a **from-scratch PyTorch implementation** built only from basic tensor operations (no `torch.nn` or `torch.nn.functional` losses), so you can see exactly how the formula turns into code; autograd supplies the gradients. The snippets assume `import math, torch` and return the mean over the batch. Two numerically stable helpers, `log_sigmoid` and `log_softmax`, are defined once in section 2 and reused after that.

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

```python
def mse_loss(y_hat, y):
    return ((y - y_hat) ** 2).mean()
```

### Mean Absolute Error (MAE, L1 loss)

$$
L = |y - \hat{y}| \qquad \frac{\partial L}{\partial \hat{y}} = -\mathrm{sign}(y - \hat{y})
$$

- **Optimal prediction:** minimizing expected absolute error yields a conditional median (it need not be unique when the distribution has flat regions).
- Gradient magnitude is constant, so outliers cannot dominate. The downside is that the gradient does not shrink near the optimum, so learning-rate decay helps convergence.
- Not differentiable at $e = 0$; frameworks use a subgradient.
- Equivalent to maximum likelihood under Laplace noise.

```python
def mae_loss(y_hat, y):
    return (y - y_hat).abs().mean()      # autograd uses sign(0) = 0 as the subgradient at e = 0
```

### Huber Loss

$$
L_\delta(e) =
\begin{cases}
\tfrac{1}{2}e^2 & |e| \le \delta \cr
\delta\big(|e| - \tfrac{1}{2}\delta\big) & |e| > \delta
\end{cases}
\qquad
\frac{\partial L}{\partial \hat{y}} = -\mathrm{clip}(e, -\delta, \delta)
$$

- Quadratic near zero and linear in the tails. Combines MSE's smooth behavior near zero with MAE's robustness to large errors.
- Continuously differentiable ($C^1$), but not twice differentiable at the transition points $e = \pm\delta$, where the second derivative jumps from 1 to 0.
- The gradient with respect to the prediction is the negative **clipped** residual. Its magnitude is capped at $\delta$, so large errors cannot produce arbitrarily large gradients.
- $\delta$ should be on the scale of a "normal" residual. A common choice is $\delta = 1$ on standardized targets.
- **Smooth L1** (common in object detection) is Huber divided by $\delta$ (PyTorch calls the threshold `beta`). The two have different scaling conventions, so `HuberLoss(delta=d)` and `SmoothL1Loss(beta=d)` differ by a factor of $d$.

```python
def huber_loss(y_hat, y, delta=1.0):
    e = (y - y_hat).abs()
    quadratic = 0.5 * e ** 2                     # |e| <= delta
    linear = delta * (e - 0.5 * delta)           # |e| >  delta
    return torch.where(e <= delta, quadratic, linear).mean()

def smooth_l1_loss(y_hat, y, beta=1.0):
    return huber_loss(y_hat, y, delta=beta) / beta    # Smooth L1 = Huber / delta
```

### Log-Cosh Loss

$$
L = \log\cosh(\hat{y} - y) \qquad \frac{\partial L}{\partial \hat{y}} = \tanh(\hat{y} - y)
$$

- Approximately $\tfrac{1}{2}e^2$ for small errors and $|e| - \log 2$ for large ones.
- Like Huber but infinitely differentiable and with no $\delta$ to tune. Useful for second-order methods such as gradient-boosted trees.
- Its transition scale is fixed at $|e| \approx 1$, so it behaves sensibly only when targets are on a sensible scale (standardize them).
- **Optimal prediction:** the value $\hat{y}$ satisfying $\mathbb{E}[\tanh(\hat{y} - Y) \mid X = x] = 0$. This is a robust location estimate; it is generally neither the conditional mean nor the median, nor a fixed blend of the two.

```python
def log_cosh_loss(y_hat, y):
    a = (y_hat - y).abs()
    # log(cosh(a)) = a + log(1 + e^{-2a}) - log 2; torch.cosh(a) itself overflows for a > ~89
    return (a + torch.log1p(torch.exp(-2 * a)) - math.log(2.0)).mean()
```

### Quantile (Pinball) Loss

$$
L_\tau(e) = \max\big(\tau e,\ (\tau - 1)e\big), \qquad \tau \in (0, 1)
$$

- **Optimal prediction:** the conditional $\tau$-quantile. With $\tau = 0.5$ it is half the MAE.
- Under-prediction costs $\tau$ per unit and over-prediction costs $1 - \tau$, so a high $\tau$ pushes predictions upward.
- Train at $\tau = 0.05$ and $\tau = 0.95$ to *estimate* a central 90% prediction interval. Check its coverage on validation data: estimated quantiles are not automatically calibrated, so the interval may cover more or less than 90% of unseen targets.
- **Asymmetric costs.** If under-predicting costs $c_u$ per unit and over-predicting costs $c_o$ (e.g. running short on electricity is worse than over-supplying), the cost-minimizing prediction is the $\tau$-quantile with $\tau = c_u / (c_u + c_o)$. Quantile loss *is* the general linear asymmetric loss, so no separate loss is needed.

```python
def quantile_loss(y_hat, y, tau=0.5):
    e = y - y_hat
    return torch.maximum(tau * e, (tau - 1) * e).mean()
```

### Gaussian Negative Log-Likelihood (heteroscedastic regression)

Plain MSE gives one prediction per input and assumes the noise is the same everywhere. Here the network predicts **two numbers for every input $x$**: a mean $\mu(x)$ (the prediction) and a variance $s^2(x)$ (how uncertain that prediction is):

$$
L = \frac{1}{2}\left(\log s^2(x) + \frac{\big(y - \mu(x)\big)^2}{s^2(x)}\right)
$$

**What gets trained.** $\mu$ and $s^2$ are not parameters. They are outputs of the network, recomputed for every example, typically from two output heads on a shared body. Training updates the network's weights, and the loss reaches them by backpropagating through both outputs. There is no target for $s^2$: the network learns uncertainty only from how far $\mu$ lands from $y$.

**What each output learns** (writing $v = \log s^2$, which is what the network actually outputs):

$$
\frac{\partial L}{\partial \mu} = -\frac{y - \mu}{s^2} \qquad \frac{\partial L}{\partial v} = \frac{1}{2}\left(1 - \frac{(y - \mu)^2}{s^2}\right)
$$

- **$\mu$** is pulled toward $y$, like with MSE, but the pull is divided by $s^2$. Examples the model marks as uncertain move $\mu$ less.
- **$s^2$** settles where the second gradient is zero, at $s^2 = (y - \mu)^2$. In expectation that is the variance of $y$ around $\mu$ for inputs like $x$. So $s^2$ learns to predict the squared error the model makes on that kind of input.
- **The trade-off.** On a hard example the model can shrink the squared-error term by predicting a larger variance, but pays $\log s^2$ for it. Neither term can be cheated alone.
- **Pitfall.** Raising $s^2$ also weakens the pull on $\mu$, so the model can "give up" on hard examples instead of fitting them. β-NLL (Seitzer et al., 2022) counters this by weighting each example's loss by a detached $(s^2)^\beta$, with $\beta \approx 0.5$.
- **Predict $\log s^2$, not $s^2$.** Any real output is then a valid variance, and there is no division by a value near zero. Clamping it (e.g. to $[-10, 10]$) adds a further guard early in training.
- With $s^2$ fixed to the same constant for every example, the loss is MSE scaled by $1/(2s^2)$ plus a constant, so this is a strict generalization of MSE.

```python
def gaussian_nll_loss(mu, log_var, y):
    # the model outputs log s^2, so the variance exp(log_var) is always positive
    return 0.5 * (log_var + (y - mu) ** 2 / log_var.exp()).mean()

class MeanVarianceNet(torch.nn.Module):
    # one shared body, two heads: mu(x) and log s^2(x)
    def __init__(self, d_in, d_hidden=64):
        super().__init__()
        self.body = torch.nn.Sequential(torch.nn.Linear(d_in, d_hidden), torch.nn.ReLU())
        self.mu_head = torch.nn.Linear(d_hidden, 1)
        self.log_var_head = torch.nn.Linear(d_hidden, 1)

    def forward(self, x):
        h = self.body(x)
        return self.mu_head(h).squeeze(-1), self.log_var_head(h).squeeze(-1).clamp(-10, 10)

model = MeanVarianceNet(d_in=8)
mu, log_var = model(x)                      # a different mu and s^2 for every example
loss = gaussian_nll_loss(mu, log_var, y)
loss.backward()                             # gradients reach the body and both heads
```

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

```python
def poisson_nll_loss(log_rate, y):
    # predict the log-rate so the rate exp(log_rate) is always positive; log(y!) is dropped
    return (log_rate.exp() - y * log_rate).mean()

def msle_loss(y_hat, y):
    return ((torch.log1p(y) - torch.log1p(y_hat)) ** 2).mean()

def mape(y_hat, y, eps=1e-8):
    # fraction, not percent; eps only avoids division by zero, MAPE is still meaningless at y = 0
    return ((y - y_hat).abs() / y.abs().clamp_min(eps)).mean()

def tweedie_loss(log_mu, y, p=1.5):
    # Tweedie NLL for power 1 < p < 2, up to terms that depend only on y
    return (-y * torch.exp((1 - p) * log_mu) / (1 - p)
            + torch.exp((2 - p) * log_mu) / (2 - p)).mean()
```

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

With label $y \in \lbrace0, 1\rbrace$, the model outputs a **logit** $z$: a raw score that can be any real number. The **sigmoid** function $\sigma$ turns it into a probability $p = \sigma(z)$:

$$
\sigma(z) = \frac{1}{1 + e^{-z}}
$$

| $z$ | $-4$ | $-2$ | $0$ | $2$ | $4$ |
|---|---|---|---|---|---|
| $p = \sigma(z)$ | 0.018 | 0.12 | 0.5 | 0.88 | 0.982 |

The sigmoid squashes any $z$ into $(0, 1)$: $z = 0$ means a 50/50 guess, large positive $z$ pushes $p$ toward 1, and large negative $z$ toward 0. It is the inverse of the log-odds, $z = \log\frac{p}{1 - p}$, so the logit is the model's evidence for class 1 on a log-odds scale. The loss is then:

$$
L = -\big[y \log p + (1 - y)\log(1 - p)\big] \qquad \frac{\partial L}{\partial z} = p - y
$$

- The gradient with respect to the logit is just *prediction minus target*. That simplicity is a large part of why the sigmoid–cross-entropy pairing trains so well.
- Penalizes confident mistakes heavily: predicting $p = 0.01$ for a positive costs $\log 100 \approx 4.6$.
- **Multi-label classification** applies binary cross-entropy independently to each label.

**Stable form from logits** (what frameworks compute internally):

$$
L = \max(z, 0) - z\thinspace y + \log\big(1 + e^{-|z|}\big)
$$

```python
def log_sigmoid(z):
    # log σ(z) = min(z, 0) - log(1 + e^{-|z|}); never exponentiates a large positive number
    return z.clamp_max(0) - torch.log1p(torch.exp(-z.abs()))

def bce_with_logits(z, y):
    # the stable form above: max(z, 0) - z*y + log(1 + e^{-|z|})
    return (z.clamp_min(0) - z * y + torch.log1p(torch.exp(-z.abs()))).mean()
```

### From Loss to Gradient: A Worked Example

This is where the loss meets [`backpropagation.md`](backpropagation.md). Take one example with target $y = 1$ where the model currently predicts $p = \sigma(z) = 0.2$:

| Quantity | Value |
|---|---|
| Loss | $-\log 0.2 \approx 1.61$ |
| Output gradient $\partial L / \partial z = p - y$ | $0.2 - 1 = -0.8$ |
| Gradient-descent step $z \leftarrow z - \eta\thinspace(-0.8)$ | $z$ increases by $0.8\eta$ |

The negative gradient means *increasing* the logit lowers the loss, so the update pushes $z$ up and $p$ toward 1. Backpropagation then carries this $-0.8$ backward through the network with the chain rule to get the gradient for every weight. The size of the push is proportional to how wrong the prediction was: at $p = 0.9$ the output gradient would be only $-0.1$.

```python
z = torch.log(torch.tensor(0.2 / 0.8)).requires_grad_()   # logit with σ(z) = 0.2
loss = bce_with_logits(z, torch.tensor(1.0))
loss.backward()
print(round(loss.item(), 2), round(z.grad.item(), 2))     # 1.61 -0.8  (loss -log 0.2, gradient p - y)
```

### Weighted Binary Cross-Entropy

$$
L = -\big[w_+\thinspace y \log p + w_-(1 - y)\log(1 - p)\big]
$$

- Up-weight the rare class. A common starting point is $w_+ = N_{\text{neg}} / N_{\text{pos}}$, then tune it.
- In PyTorch this is the `pos_weight` argument of `BCEWithLogitsLoss`.

```python
def weighted_bce_with_logits(z, y, w_pos=1.0, w_neg=1.0):
    # log(1 - σ(z)) = log σ(-z); w_pos plays the role of PyTorch's pos_weight
    return -(w_pos * y * log_sigmoid(z) + w_neg * (1 - y) * log_sigmoid(-z)).mean()
```

### Categorical Cross-Entropy

With logits $z \in \mathbb{R}^C$, softmax probabilities $p_c = e^{z_c} / \sum_k e^{z_k}$, and one-hot target $y$:

$$
L = -\sum_{c=1}^{C} y_c \log p_c = -z_k + \log\sum_{c} e^{z_c} \qquad \frac{\partial L}{\partial z_c} = p_c - y_c
$$

where $k$ is the true class.

- The second form (**log-sum-exp**) is how it is computed stably.
- **Sparse** categorical cross-entropy is the same loss with integer class labels instead of one-hot vectors.

**Label smoothing.** A one-hot target asks for probability 1 on the true class $k$. Softmax only reaches 1 in the limit where $z_k$ is infinitely larger than every other logit, so cross-entropy keeps pushing the logits apart even on examples the model already classifies correctly. Label smoothing takes a small amount $\varepsilon$ of the target's probability and spreads it evenly over all $C$ classes:

$$
y_c = (1 - \varepsilon)\thinspace\mathbb{1}[c = k] + \frac{\varepsilon}{C}
$$

Here $\mathbb{1}[c = k]$ is 1 for the true class and 0 for the others. With $C = 4$ classes, $\varepsilon = 0.1$, and true class $k = 2$:

| Class | 1 | 2 (true) | 3 | 4 |
|---|---|---|---|---|
| One-hot target | 0 | 1 | 0 | 0 |
| Smoothed target | 0.025 | 0.925 | 0.025 | 0.025 |

- **The target becomes reachable.** The loss is now minimized at a finite gap between the true logit and the others, $z_k - z_c = \log\frac{1 - \varepsilon + \varepsilon/C}{\varepsilon/C}$, which is $\log 37 \approx 3.6$ in the example.
- **The gradient pushes back on overconfidence.** It is still $p_c - y_c$, but with the smoothed $y_c$. Once $p_k$ exceeds $0.925$, the gradient on $z_k$ turns positive and gradient descent lowers it. This keeps the logits bounded and acts as a regularizer, which often improves generalization.
- **An equivalent view:** the loss equals $(1 - \varepsilon)$ times the usual cross-entropy plus $\varepsilon$ times the cross-entropy against a uniform distribution, i.e. the normal loss plus a penalty for predictions far from uniform.
- **In practice:** $\varepsilon = 0.1$ is a common default. PyTorch exposes it as `nn.CrossEntropyLoss(label_smoothing=0.1)`; the `cross_entropy` function below builds the smoothed target explicitly.
- **Trade-offs:** the model is trained toward the smoothed targets, so its probabilities no longer estimate the true class probabilities. Calibration can get better (for an overconfident network) or worse. It can also hurt knowledge distillation. See [`regularization.md`](regularization.md#label-smoothing).

```python
def log_softmax(z, dim=-1):
    z = z - z.max(dim=dim, keepdim=True).values           # shift so the largest logit is 0
    return z - z.exp().sum(dim=dim, keepdim=True).log()    # z_c - log Σ_k e^{z_k}

def cross_entropy(z, target, label_smoothing=0.0):
    # z: (N, C) logits, target: (N,) int64 class indices
    C = z.size(1)
    y = torch.full_like(z, label_smoothing / C)                       # smoothed one-hot target
    y.scatter_(1, target.unsqueeze(1), 1 - label_smoothing + label_smoothing / C)
    return -(y * log_softmax(z)).sum(dim=1).mean()
```

### Focal Loss

$$
L = -\alpha_t (1 - p_t)^\gamma \log p_t, \qquad p_t = \begin{cases} p & y = 1 \cr 1 - p & y = 0 \end{cases}, \qquad \alpha_t = \begin{cases} \alpha & y = 1 \cr 1 - \alpha & y = 0 \end{cases}
$$

- $p_t$ is the probability assigned to the true class, and $\alpha_t$ balances the positive and negative classes.
- The factor $(1 - p_t)^\gamma$ shrinks the loss on examples the model already gets right. With $\gamma = 2$, an example at $p_t = 0.9$ is down-weighted by 100×.
- Designed for extreme imbalance, such as dense object detection where most anchors are background. Common defaults are $\gamma = 2$ and $\alpha = 0.25$. Note that $\alpha = 0.25$ *down*-weights the rare positives: the focusing term already suppresses the many easy negatives, so $\alpha$ ends up compensating in the other direction. Tune the two together.
- With $\gamma = 0$ it reduces to $\alpha$-weighted cross-entropy; with $\gamma = 0$ and $\alpha = 0.5$ it is plain cross-entropy scaled by $\tfrac{1}{2}$.

```python
def focal_loss(z, y, alpha=0.25, gamma=2.0):
    # binary focal loss from logits z, labels y in {0, 1}
    log_p_t = y * log_sigmoid(z) + (1 - y) * log_sigmoid(-z)    # log of p_t
    alpha_t = y * alpha + (1 - y) * (1 - alpha)
    return (-alpha_t * (1 - log_p_t.exp()) ** gamma * log_p_t).mean()
```

### Hinge Loss

With label $y \in \lbrace-1, +1\rbrace$ and raw score $f(x)$:

**What $f(x)$ is.** The model's output before any squashing: a single real number that can be anything from $-\infty$ to $+\infty$. For a linear SVM it is $f(x) = w^\top x + b$. Like the logit $z$ in binary cross-entropy, but it is never turned into a probability.

- **Sign = predicted class.** $f(x) > 0$ predicts $+1$, $f(x) < 0$ predicts $-1$, and $f(x) = 0$ is the decision boundary.
- **Size = confidence.** A larger $\lvert f(x) \rvert$ means the example is farther from the boundary.
- **$y f(x)$ is the margin.** It is positive when the prediction is right and negative when it is wrong, so one number captures both correctness and confidence.

The hinge loss is:

$$
L = \max\big(0,\ 1 - y f(x)\big)
$$

For a positive example ($y = +1$):

| $f(x)$ | $y f(x)$ | Loss | Situation |
|---|---|---|---|
| 2.5 | 2.5 | 0 | Correct and confident: no gradient |
| 1.0 | 1.0 | 0 | Correct, exactly on the margin |
| 0.4 | 0.4 | 0.6 | Correct, but too close to the boundary |
| 0 | 0 | 1 | On the decision boundary |
| −1.5 | −1.5 | 2.5 | Wrong class: loss grows linearly |

The loss asks for more than just the right sign. Every example must be correct by a margin of at least 1, and the penalty grows linearly with how far short it falls.

- The loss of linear SVMs. It is zero once an example is on the correct side with margin at least 1, so only margin violators (the "support vectors") contribute gradient.
- Produces scores, not calibrated probabilities.
- **Squared hinge** $\max(0, 1 - yf(x))^2$ is differentiable and penalizes violations more strongly.
- **Multi-class hinge** (Crammer–Singer): $\max\big(0,\ 1 + \max_{c \ne k} f_c - f_k\big)$.

```python
def hinge_loss(score, y):
    # y in {-1, +1}
    return (1 - y * score).clamp_min(0).mean()

def squared_hinge_loss(score, y):
    return (1 - y * score).clamp_min(0).pow(2).mean()

def multiclass_hinge_loss(scores, target):
    # Crammer–Singer; scores: (N, C), target: (N,) int64
    f_k = scores.gather(1, target.unsqueeze(1)).squeeze(1)                # true-class score
    rivals = scores.scatter(1, target.unsqueeze(1), float("-inf"))        # hide the true class
    return (1 + rivals.max(dim=1).values - f_k).clamp_min(0).mean()
```

### KL Divergence

$$
D_{\text{KL}}(P \thinspace\Vert\thinspace Q) = \sum_x P(x)\log\frac{P(x)}{Q(x)} = H(P, Q) - H(P)
$$

- Cross-entropy minus the target's own entropy. When $P$ is a fixed one-hot label, $H(P) = 0$ and minimizing KL is identical to minimizing cross-entropy.
- KL matters when the target is itself a distribution, e.g. **knowledge distillation**, where a student matches the softened outputs of a teacher at temperature $T$:

$$
L = T^2 \cdot D_{\text{KL}}\Big(\mathrm{softmax}(z^{\text{teacher}}/T) \thinspace\Big\Vert\thinspace \mathrm{softmax}(z^{\text{student}}/T)\Big)
$$

- Asymmetric: $D_{\text{KL}}(P\Vert Q) \neq D_{\text{KL}}(Q\Vert P)$. When fitting an approximation $Q$ to a target $P$, forward KL $D_{\text{KL}}(P\Vert Q)$ tends to favor covering all the regions where $P$ has mass, while reverse KL $D_{\text{KL}}(Q\Vert P)$ often favors concentrating on individual modes. These are tendencies in common approximation settings, not universal guarantees.

```python
def kl_divergence(log_p, log_q):
    # D_KL(P || Q) from log-probabilities: sum over classes, mean over the batch
    return (log_p.exp() * (log_p - log_q)).sum(dim=-1).mean()

def distillation_loss(student_logits, teacher_logits, T=4.0):
    log_p = log_softmax(teacher_logits / T)     # target P: softened teacher
    log_q = log_softmax(student_logits / T)     # approximation Q: softened student
    return T ** 2 * kl_divergence(log_p, log_q)
```

### Brier Score

MSE applied to predicted probabilities:

$$
L = (p - y)^2 \qquad \text{(multi-class: } L = \textstyle\sum_{c=1}^{C} (p_c - y_c)^2\text{)}
$$

- A **strictly proper scoring rule**: in expectation it is minimized by reporting the true probability, just like log loss.
- Bounded between 0 and 1 (binary case), so it does not penalize confident mistakes as aggressively as log loss. Predicting $p = 0.01$ for a positive costs $0.98$ under Brier but $4.6$ under log loss.
- Mostly used as an **evaluation metric for calibration and probabilistic forecasts**, where its boundedness makes it robust to a few confidently wrong predictions. As a training loss it suffers from the vanishing-gradient problem described next.

```python
def brier_score(p, y):
    # binary: p, y of shape (N,); multi-class: p of shape (N, C) and one-hot y
    sq = (p - y) ** 2
    return (sq.sum(dim=1) if sq.dim() == 2 else sq).mean()
```

### Why Cross-Entropy Instead of MSE for Classification?

With a sigmoid output, the MSE (Brier) gradient with respect to the logit is $2(p - y)\thinspace p(1 - p)$. When the model is confidently wrong, $p(1 - p) \approx 0$ and learning stalls. Cross-entropy's gradient $p - y$ has no such factor, so confidently wrong predictions get the largest updates.

```python
z = torch.tensor(-6.0, requires_grad=True)          # confidently wrong: p ≈ 0.0025 but y = 1
y = torch.tensor(1.0)
brier = (log_sigmoid(z).exp() - y) ** 2
ce = bce_with_logits(z, y)
print(torch.autograd.grad(brier, z)[0].item())      # ≈ -0.0049: learning stalls
print(torch.autograd.grad(ce, z)[0].item())         # ≈ -0.9975: large corrective step
```

---

## 3. Ranking, Metric, and Contrastive Losses

These losses shape an **embedding space** or a **ranking** rather than predict a label.

### Contrastive (Pairwise) Loss

For an embedding distance $d$ between two items and $y = 1$ if they are similar:

$$
L = y\thinspace d^2 + (1 - y)\max(0,\ m - d)^2
$$

Pulls similar pairs together and pushes dissimilar pairs apart until they are at least margin $m$ away.

```python
def euclidean(a, b, eps=1e-12):
    return ((a - b) ** 2).sum(dim=-1).add(eps).sqrt()    # eps keeps the gradient finite at d = 0

def contrastive_loss(x1, x2, y, margin=1.0):
    # x1, x2: (N, D) embeddings; y = 1 for similar pairs, 0 for dissimilar
    d = euclidean(x1, x2)
    return (y * d ** 2 + (1 - y) * (margin - d).clamp_min(0) ** 2).mean()
```

### Triplet Loss

$$
L = \max\big(0,\ d(a, p) - d(a, n) + \alpha\big)
$$

- Anchor $a$ must be closer to positive $p$ than to negative $n$ by margin $\alpha$.
- **Mining matters more than the formula.** Random triplets are mostly already satisfied and give zero gradient. Use semi-hard or batch-hard negative mining.

```python
def triplet_loss(anchor, positive, negative, margin=1.0):
    d_ap = euclidean(anchor, positive)
    d_an = euclidean(anchor, negative)
    return (d_ap - d_an + margin).clamp_min(0).mean()

def batch_hard_triplet_loss(emb, labels, margin=0.3):
    # every item is an anchor; pick its farthest positive and closest negative in the batch
    d = euclidean(emb[:, None, :], emb[None, :, :])                        # (N, N)
    same = labels[:, None] == labels[None, :]
    hardest_pos = d.masked_fill(~same, 0.0).max(dim=1).values
    hardest_neg = d.masked_fill(same, float("inf")).min(dim=1).values
    return (hardest_pos - hardest_neg + margin).clamp_min(0).mean()
```

### InfoNCE / NT-Xent (Contrastive Self-Supervised Learning)

For an anchor $q$, one positive $k^+$, and a batch of negatives, with similarity $s(\cdot,\cdot)$ (usually cosine) and temperature $\tau$:

$$
L = -\log \frac{\exp\big(s(q, k^+)/\tau\big)}{\sum_{j} \exp\big(s(q, k_j)/\tau\big)}
$$

- Cross-entropy where the "correct class" is the positive pair. Used in SimCLR, MoCo, and CLIP.
- The number and quality of negatives, along with the temperature $\tau$ (often 0.05–0.2), strongly affect the learned representations. Large batch sizes help because every other item in the batch serves as a negative.

```python
def info_nce(q, k, tau=0.1):
    # q, k: (N, D); row i of k is the positive for row i of q, all other rows are negatives
    q = q / q.norm(dim=1, keepdim=True)
    k = k / k.norm(dim=1, keepdim=True)
    logits = q @ k.T / tau                         # (N, N) cosine similarities / temperature
    return -log_softmax(logits).diagonal().mean()  # cross-entropy with the diagonal as the class

# CLIP-style symmetric version: 0.5 * (info_nce(img, txt) + info_nce(txt, img))
```

### Pairwise Ranking Losses

| Loss | Formula | Used in |
|---|---|---|
| RankNet / BPR | $-\log \sigma(s_i - s_j)$ for $i$ preferred over $j$ | Recommenders, learning to rank |
| Margin ranking | $\max(0,\ m - (s_i - s_j))$ | Retrieval |
| Bradley–Terry reward loss | $-\log\sigma\big(r(y_w) - r(y_l)\big)$ | Reward models for RLHF |
| LambdaRank / ListNet | Weights pairs by their effect on NDCG / uses whole list | Search ranking |

```python
def ranknet_loss(s_i, s_j):
    # item i preferred over j; same formula as BPR and the Bradley–Terry reward loss
    return -log_sigmoid(s_i - s_j).mean()

def margin_ranking_loss(s_i, s_j, margin=1.0):
    return (margin - (s_i - s_j)).clamp_min(0).mean()

def listnet_loss(scores, relevance):
    # scores, relevance: (num_queries, list_len); cross-entropy between top-1 distributions
    target = log_softmax(relevance).exp()
    return -(target * log_softmax(scores)).sum(dim=1).mean()
```

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

```python
def nll_loss(log_probs, target):
    # log_probs: (N, C) log-probabilities, target: (N,) int64; cross_entropy = nll_loss(log_softmax(z))
    return -log_probs.gather(1, target.unsqueeze(1)).mean()

def perplexity(token_logits, tokens):
    # token_logits: (N, T, V) next-token logits, tokens: (N, T) int64
    nll = -log_softmax(token_logits).gather(2, tokens.unsqueeze(2)).mean()
    return nll.exp()
```

### VAE: Evidence Lower Bound (ELBO)

$$
\mathcal{L}_{\text{VAE}} = \underbrace{-\thinspace\mathbb{E}_{q(z\mid x)}\big[\log p(x \mid z)\big]}_{\text{reconstruction}} + \beta \cdot \underbrace{D_{\text{KL}}\big(q(z \mid x)\thinspace\Vert\thinspace p(z)\big)}_{\text{latent regularizer}}
$$

- Minimize this (the negative ELBO). $\beta = 1$ is the standard VAE; $\beta > 1$ ($\beta$-VAE) encourages disentangled latents.
- For a Gaussian encoder and standard normal prior, the KL term has a closed form: $\tfrac{1}{2}\sum_j\big(\mu_j^2 + \sigma_j^2 - \log\sigma_j^2 - 1\big)$.

```python
def reparameterize(mu, log_var):
    # z = mu + sigma * eps keeps sampling differentiable with respect to mu and log_var
    return mu + (0.5 * log_var).exp() * torch.randn_like(mu)

def vae_loss(x_logits, x, mu, log_var, beta=1.0):
    # Bernoulli decoder for pixels in [0, 1]; encoder q(z|x) = N(mu, exp(log_var))
    recon = (x_logits.clamp_min(0) - x_logits * x
             + torch.log1p(torch.exp(-x_logits.abs()))).flatten(1).sum(1)    # -log p(x|z)
    kl = 0.5 * (mu ** 2 + log_var.exp() - log_var - 1).sum(1)              # closed-form KL
    return (recon + beta * kl).mean()
```

### GAN Losses

**Original (non-saturating) GAN:**

$$
L_D = -\mathbb{E}_{x}\big[\log D(x)\big] - \mathbb{E}_{z}\big[\log(1 - D(G(z)))\big], \qquad L_G = -\mathbb{E}_{z}\big[\log D(G(z))\big]
$$

The generator uses $-\log D(G(z))$ rather than $\log(1 - D(G(z)))$ because the latter has vanishing gradients early in training.

**Wasserstein GAN (WGAN-GP):** the critic $D$ outputs an unbounded score and is kept approximately 1-Lipschitz with a gradient penalty:

$$
L_D = \mathbb{E}_{z}\big[D(G(z))\big] - \mathbb{E}_{x}\big[D(x)\big] + \lambda\thinspace\mathbb{E}_{\hat{x}}\Big[\big(\lVert\nabla_{\hat{x}} D(\hat{x})\rVert_2 - 1\big)^2\Big], \qquad L_G = -\mathbb{E}_{z}\big[D(G(z))\big]
$$

The critic objective approximates the negative Wasserstein-1 (Earth Mover's) distance, while the gradient penalty *encourages* the critic to satisfy the 1-Lipschitz constraint rather than enforcing it exactly. This gives more informative gradients and more stable training than the original objective.

```python
# Original GAN, from discriminator logits (D = σ(logit))
def gan_d_loss(real_logits, fake_logits):
    # -log D(x) - log(1 - D(G(z))); compute fake_logits on G(z).detach()
    return -(log_sigmoid(real_logits) + log_sigmoid(-fake_logits)).mean()

def gan_g_loss(fake_logits):
    return -log_sigmoid(fake_logits).mean()      # non-saturating: -log D(G(z))

# WGAN-GP
def wgan_gp_d_loss(critic, real, fake, lam=10.0):
    fake = fake.detach()
    eps = torch.rand(real.size(0), *[1] * (real.dim() - 1), device=real.device)
    x_hat = (eps * real + (1 - eps) * fake).requires_grad_(True)     # random interpolates
    grad = torch.autograd.grad(critic(x_hat).sum(), x_hat, create_graph=True)[0]
    penalty = ((grad.flatten(1).norm(dim=1) - 1) ** 2).mean()
    return critic(fake).mean() - critic(real).mean() + lam * penalty

def wgan_g_loss(critic, fake):
    return -critic(fake).mean()
```

### Diffusion Models

The common simplified objective trains a network $\epsilon_\theta$ to predict the noise added at a random timestep $t$:

$$
L = \mathbb{E}_{x_0, \epsilon, t}\Big[\big\lVert \epsilon - \epsilon_\theta(x_t, t)\big\rVert^2\Big]
$$

This is plain MSE, derived from a weighted variational bound.

```python
betas = torch.linspace(1e-4, 0.02, 1000)            # linear noise schedule (DDPM)
alpha_bar = torch.cumprod(1 - betas, dim=0)

def diffusion_loss(eps_model, x0, alpha_bar):
    t = torch.randint(0, len(alpha_bar), (x0.size(0),), device=x0.device)
    a = alpha_bar[t].view(-1, *[1] * (x0.dim() - 1))
    noise = torch.randn_like(x0)
    x_t = a.sqrt() * x0 + (1 - a).sqrt() * noise      # sample q(x_t | x_0) in one step
    return ((noise - eps_model(x_t, t)) ** 2).mean()
```

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

```python
def dice_loss(logits, target, eps=1.0):
    # per-image soft Dice for binary segmentation; logits, target: (N, ...)
    p = log_sigmoid(logits).exp().flatten(1)
    g = target.flatten(1)
    inter = (p * g).sum(1)
    return 1 - ((2 * inter + eps) / (p.sum(1) + g.sum(1) + eps)).mean()
```

### IoU (Jaccard) and Box-Regression Losses

- **Soft IoU loss:** $1 - \frac{\sum p_i g_i}{\sum p_i + \sum g_i - \sum p_i g_i}$.
- **GIoU, DIoU, CIoU:** variants for bounding boxes that still give gradients when boxes do not overlap and also account for center distance and aspect ratio.

```python
def soft_iou_loss(logits, target, eps=1e-6):
    p = log_sigmoid(logits).exp().flatten(1)
    g = target.flatten(1)
    inter = (p * g).sum(1)
    union = p.sum(1) + g.sum(1) - inter
    return 1 - ((inter + eps) / (union + eps)).mean()

def giou_loss(pred, target, eps=1e-7):
    # boxes: (N, 4) as (x1, y1, x2, y2)
    def area(b):
        return (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    inter = (torch.minimum(pred[:, 2:], target[:, 2:])
             - torch.maximum(pred[:, :2], target[:, :2])).clamp_min(0).prod(dim=1)
    union = area(pred) + area(target) - inter
    iou = inter / (union + eps)
    enclose = (torch.maximum(pred[:, 2:], target[:, 2:])
               - torch.minimum(pred[:, :2], target[:, :2])).prod(dim=1)   # smallest enclosing box
    return (1 - iou + (enclose - union) / (enclose + eps)).mean()
```

### Tversky Loss

Generalizes Dice by weighting false positives and false negatives separately:

$$
L_{\text{Tversky}} = 1 - \frac{TP + \epsilon}{TP + \alpha\thinspace FP + \beta\thinspace FN + \epsilon}
$$

with soft counts $TP = \sum_i p_i g_i$, $FP = \sum_i p_i (1 - g_i)$, and $FN = \sum_i (1 - p_i) g_i$.

- $\alpha = \beta = 0.5$ recovers Dice; $\alpha = \beta = 1$ gives soft IoU.
- $\beta > \alpha$ penalizes missed foreground more than false alarms, trading precision for recall. This helps when objects are small and easy to miss (e.g. lesions); $\alpha = 0.3, \beta = 0.7$ is a common starting point.

```python
def tversky_loss(logits, target, alpha=0.3, beta=0.7, eps=1.0):
    p = log_sigmoid(logits).exp().flatten(1)
    g = target.flatten(1)
    tp = (p * g).sum(1)
    fp = (p * (1 - g)).sum(1)
    fn = ((1 - p) * g).sum(1)
    return 1 - ((tp + eps) / (tp + alpha * fp + beta * fn + eps)).mean()
```

### CTC Loss

Connectionist Temporal Classification sums the probability of every alignment between an input sequence and a shorter label sequence. Used for speech recognition and handwriting/OCR when frame-level alignments are unknown.

```python
def ctc_loss(log_probs, labels, blank=0):
    # one sequence: log_probs (T, C) from log_softmax, labels (L,) int64 without blanks
    ext = torch.full((2 * len(labels) + 1,), blank, dtype=torch.long)
    ext[1::2] = labels                                   # blank, l1, blank, l2, ..., blank
    S = len(ext)
    # a path may jump two states (skip a blank) unless the two labels are equal
    can_skip = torch.zeros(S, dtype=torch.bool)
    can_skip[2:] = (ext[2:] != blank) & (ext[2:] != ext[:-2])
    NEG = -1e30                                          # finite "log 0", so gradients stay finite
    alpha = torch.full((S,), NEG)
    alpha[:2] = log_probs[0, ext[:2]]                    # start with a blank or the first label
    for t in range(1, log_probs.size(0)):
        stay = alpha
        step = torch.cat([torch.full((1,), NEG), alpha[:-1]])
        skip = torch.where(can_skip, torch.cat([torch.full((2,), NEG), alpha[:-2]]), NEG)
        alpha = torch.logsumexp(torch.stack([stay, step, skip]), dim=0) + log_probs[t, ext]
    return -torch.logsumexp(alpha[-2:], dim=0)           # end on the last label or the final blank

# for a batch, average ctc_loss over sequences; nn.CTCLoss does the same thing vectorized
```

### Perceptual Loss

Computes the distance between feature maps of a pretrained network (often VGG) rather than between raw pixels. Produces sharper results than pixel MSE in super-resolution and style transfer.

```python
import torchvision

vgg = torchvision.models.vgg16(weights="IMAGENET1K_V1").features.eval().requires_grad_(False)

def vgg_features(x, layers=(3, 8, 15)):          # relu1_2, relu2_2, relu3_3
    feats = []
    for i, layer in enumerate(vgg):
        x = layer(x)
        if i in layers:
            feats.append(x)
        if i == max(layers):
            break
    return feats

def perceptual_loss(x_hat, x):
    # x_hat, x: (N, 3, H, W) images normalized with the ImageNet mean and std
    with torch.no_grad():
        target_feats = vgg_features(x)
    return sum(((f_hat - f) ** 2).mean() for f_hat, f in zip(vgg_features(x_hat), target_feats))
```

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

- PyTorch `MultiMarginLoss` is a multi-class hinge that *sums* over wrong classes, unlike the Crammer–Singer *max* form; Keras `Hinge` is the binary hinge with targets in $\lbrace-1, +1\rbrace$ (or $\lbrace0, 1\rbrace$, converted internally).
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
