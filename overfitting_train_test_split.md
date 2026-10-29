# Overfitting and the Train/Test Split

*Why a trained model can look good and still fail.*

---

## The core problem

A model is trained to do well on the data it sees. What we actually want is for it to do well on data it **hasn't** seen: next month's customers, a new patient, tomorrow's prices.

These are not the same goal. A model can score very well on its training data while having learned nothing that carries over. That gap is the whole story of overfitting.

---

## What overfitting is

**Overfitting** happens when a model learns the *noise* and quirks of its training data instead of the underlying pattern.

An analogy: a student memorizes the answers to last year's exam. They'd score 100% on that exact exam, but they haven't learned the subject, so a new exam exposes them.

Signs of an overfit model:

- Very high accuracy (or very low error) on training data
- Noticeably worse performance on new data
- Overly complex behavior, such as a decision boundary that twists around individual points

### The opposite: underfitting

**Underfitting** is when a model is too simple to capture the real pattern (for example, fitting a straight line to clearly curved data). It does poorly on *both* training and new data.

| | Training performance | New-data performance |
|---|---|---|
| Underfit | Poor | Poor |
| Good fit | Good | Good (slightly lower) |
| Overfit | Excellent | Poor |

---

## Why a model can look good and still fail

If you measure a model's performance on the same data it was trained on, you're grading the student with the exam they memorized. The score tells you how well it memorized, not how well it generalizes.

Common ways this fools people:

1. **Evaluating on training data.** The most basic mistake. Training accuracy is almost always optimistic.
2. **Data leakage.** Information from the test set (or from the future) sneaks into training, such as scaling features using statistics computed on the full dataset, or a feature that's only known after the outcome.
3. **Tuning against the test set.** Trying many models or settings and keeping whichever scores best on the test set slowly turns the test set into training data.
4. **Distribution shift.** The model generalizes fine to data like its training set, but the real world changes (new users, new season, new equipment).
5. **Duplicates or near-duplicates** shared between training and test sets, inflating test scores.

---

## The train/test split

The basic defense: **hold back some data the model never sees during training**, then use it to estimate real-world performance.

```
Full dataset
├── Training set (e.g. 80%)  → the model learns from this
└── Test set     (e.g. 20%)  → used once, at the end, to estimate performance
```

The test set stands in for "the future." Its score is your honest estimate of how the model will perform on new data.

### Adding a validation set

When you're choosing between models or tuning settings (hyperparameters), you need a separate set for those decisions, so the test set stays untouched.

```
Full dataset
├── Training set   (~60–70%)  → fit model parameters
├── Validation set (~15–20%)  → compare models, tune hyperparameters
└── Test set       (~15–20%)  → final, one-time evaluation
```

Rule of thumb: **look at the test set as little as possible, ideally once.**

### Cross-validation

With limited data, a single split can be noisy. **k-fold cross-validation** splits the data into *k* parts, trains *k* times (each time holding out a different part), and averages the results. This gives a more stable estimate and uses all data for both training and validation. Keep a separate final test set if you can.

---

## Doing the split correctly

- **Shuffle randomly** before splitting (unless order matters, see below).
- **Stratify** for classification so each split has similar class proportions, especially with rare classes.
- **Split by time** for time-series data: train on the past, test on the future. Random shuffling leaks future information.
- **Split by group** when rows aren't independent (e.g. multiple records per patient or user). Keep all of one group's rows in the same split.
- **Fit preprocessing on training data only.** Scalers, imputers, encoders, and feature selection should learn from the training set and then be *applied* to validation/test.

---

## Diagnosing overfitting

Compare training and validation performance:

- **Big gap** (train much better than validation) → likely overfitting
- **Both poor** → likely underfitting
- **Both good and close** → healthy

**Learning curves** (performance vs. training set size or training time) help. For iterative models, if validation error starts rising while training error keeps falling, overfitting has begun.

---

## How to reduce overfitting

- **Get more data.** The most reliable fix.
- **Simplify the model.** Fewer parameters, shallower trees, fewer features.
- **Regularization.** Penalize complexity (L1/L2, weight decay, dropout in neural networks).
- **Early stopping.** Stop training when validation performance stops improving.
- **Data augmentation.** Create realistic variations of training examples (common for images, audio, text).
- **Ensembling.** Average multiple models (e.g. random forests, bagging) to smooth out individual quirks.
- **Cross-validation** for model selection, so choices aren't driven by one lucky split.

---

## The bias–variance view (briefly)

- **Bias**: error from wrong or overly simple assumptions → underfitting.
- **Variance**: error from sensitivity to the specific training sample → overfitting.

Making a model more flexible lowers bias but raises variance. The goal is the balance point where total error on new data is lowest.

---

## Key takeaways

1. Training performance is not real performance.
2. Always evaluate on data the model hasn't seen, and keep that data truly separate.
3. Use a validation set (or cross-validation) for decisions; save the test set for the end.
4. Split in a way that matches how the model will be used (by time, by group).
5. Watch for leakage: if results look too good, they often are.
6. Even a correct test score assumes the future resembles the past; monitor models after deployment.
