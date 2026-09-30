# ML Evaluation Pitfalls: A Case Study in Model Evaluation Rigor

## Overview

Most machine learning projects report a single accuracy number and move on. This project asks a different question: **how much can you trust that number?** Using California housing price prediction as a concrete testbed, this project investigates several common but under-scrutinized pitfalls in how machine learning models are evaluated — and demonstrates, with real evidence, how much they matter.

## Tech Stack

- **Python 3**
- **pandas / numpy** — data handling and numerical computation
- **scikit-learn** — `LinearRegression`, `KFold`, `cross_val_score`, `Pipeline`, `StandardScaler`, `train_test_split`, evaluation metrics
- **XGBoost** — `XGBRegressor`
- **SciPy** (`scipy.stats`) — paired t-test for significance testing
- **Matplotlib** — visualization (bias-variance tradeoff curve)

## Research Questions

1. How sensitive are performance estimates to random train-test splits?
2. How does cross-validation improve reliability over a single split?
3. Does model complexity increase generalization uncertainty (overfitting risk)?
4. Are performance differences between models statistically significant, or could they be noise?

## Dataset

[California Housing dataset](https://scikit-learn.org/stable/datasets/real_world.html#california-housing-dataset) (via scikit-learn), 20,640 samples, 8 numerical features, target is median house value.

**Known limitation:** the target variable is censored at 5.0 (≈$500k) — 992 samples (4.8%) hit this cap due to how the original census data was collected. This is a property of the dataset, not an artifact of this analysis, and is discussed further below.

## Methods

Four increasingly rigorous approaches to estimating model performance were compared:

- **Method A — Single random split (×10 repeats):** the naive baseline. A model is evaluated once on a random 80/20 split; repeated 10 times with different seeds to observe how much the reported score varies by chance alone.
- **Method B — 5-fold cross-validation:** the standard, more robust alternative.
- **Method C — Bootstrap (1,000 resamples, Out-of-Bag evaluation):** samples are drawn with replacement; the ~37% of samples not selected in each resample serve as a natural held-out test set, avoiding any additional train/test split.
- **Model comparison:** Linear Regression (simple) vs. XGBoost (complex), evaluated on identical 5-fold splits for a fair, controlled comparison.

All pipelines use `sklearn.pipeline.Pipeline` to ensure `StandardScaler` is fit only on training data within each fold/resample — avoiding a common and easy-to-miss source of data leakage.

## Results

### Method A vs. Method B: does validation strategy matter?

| Method | Mean R² | Std |
|---|---|---|
| A — Single split (×10) | 0.6066 | 0.015 |
| B — 5-fold CV | 0.6014 | 0.017 |

The two methods agree closely here. This is itself a finding worth noting: on a dataset of this size (20,640 samples), single-split variance turns out to be modest. The risk single-split evaluation poses is likely to be much larger on smaller datasets — a natural extension of this analysis.

### Method C: what a simple standard deviation misses

Bootstrap (1,000 resamples) gives a mean R² ≈ 0.60, consistent with Methods A and B. But its 95% confidence interval is notably wide (≈[0.04, 0.62]), driven by a long tail: roughly 3–4% of resamples produced R² below 0.3, including extreme negative values (worse than predicting the mean). This tail was verified to be reproducible (not a coding artifact) using a fixed random seed across repeated runs.

**Takeaway:** a simple standard deviation, as computed in Methods A and B, does not capture this kind of tail risk. Bootstrap's value lies not in changing the central estimate, but in revealing instability that a coarser method hides.

### Model complexity: Linear Regression vs. XGBoost

| Model | Mean R² (5-fold CV) | Std |
|---|---|---|
| Linear Regression | 0.6014 | 0.017 |
| XGBoost (max_depth=6) | 0.8321 | 0.0085 |

XGBoost substantially outperforms Linear Regression — and does so *more* consistently (lower std), not less.

**Overfitting check** (train vs. test R², max_depth=6):

| Model | Train R² | Test R² | Gap |
|---|---|---|---|
| Linear Regression | 0.6099 | 0.5911 | 0.0188 |
| XGBoost | 0.8986 | 0.8311 | 0.0675 |

XGBoost's train/test gap is over 3× larger — clear evidence of higher overfitting risk. However, its absolute test performance still far exceeds Linear Regression's, so this risk does not erase its advantage.

**Bias-variance tradeoff** (`max_depth` swept from 1 to 20, using a proper train/validation/test split so model selection does not contaminate the final evaluation):

![Bias-Variance Tradeoff Curve](Bias-Variance%20Tradeoff%20Curve.png)

Validation R² peaks around `max_depth=6–8` and declines beyond that as the model begins to fit noise rather than signal — the textbook underfit → optimal → overfit pattern. The selected optimal depth (8, validation R² = 0.8327) was confirmed on a held-out test set never used during selection: test R² = 0.8330, nearly identical, indicating the selection process itself was not overfit to the validation data.

### Is the XGBoost advantage statistically significant?

A paired t-test on the 5 matched CV folds (same splits for both models):

- t-statistic = 39.51, p-value ≈ 0.000002
- Per-fold difference (XGBoost − Linear Regression): consistently between +0.217 and +0.250 across all 5 folds

The advantage is not only large but remarkably consistent — no fold favored Linear Regression. This is strong evidence the difference reflects a real effect, not sampling noise.

**Limitation:** a standard paired t-test assumes independent observations. CV folds share overlapping training data and are therefore not fully independent, which can understate the true variance. A Nadeau-Bengio corrected variance estimate would be more rigorous; it was not implemented here due to time constraints and is noted as a direction for future work.

## Synthesis

Putting the pieces together: XGBoost is a riskier model in the sense that it overfits more (a larger train/test gap), but its raw performance advantage is large enough — and statistically robust enough — that it remains clearly better even after accounting for that risk. This is a more complete answer than either "complex models are always better" or "complex models are always overfitting" in isolation.

## Limitations

- All experiments use a single dataset (California Housing); conclusions about the relative stability of single-split vs. CV evaluation, in particular, may not generalize to smaller datasets.
- The census-imposed target ceiling at $500k likely degrades performance on high-value homes specifically, for all models compared here.
- Significance testing uses a standard paired t-test rather than a CV-aware corrected variant (see above).
- Bootstrap resampling was implemented with a simplified in-bag/out-of-bag split rather than a nested train/val split within each resample.

## Repository Structure

```
eda.py                  # Data loading and exploratory analysis
baseline.py              # Method A: single random split (x10)
crossval.py              # Method B: 5-fold cross-validation
bootstrap.py              # Method C: bootstrap with OOB evaluation
XGBoost.py                # XGBoost vs. Linear Regression comparison
overfitting_check.py      # Train/test gap analysis
bias_variance.py          # Bias-variance tradeoff curve (max_depth sweep)
significance.py           # Paired t-test
notes.md                  # Full working notes, including debugging process
```
