# ML Evaluation Pitfalls

A hands-on study of common pitfalls in evaluating machine learning models, using the California Housing regression dataset as a testbed. The project shows how naive evaluation (a single train/test split) can produce misleading, unstable performance estimates, and walks through progressively more rigorous techniques — cross-validation, bootstrap resampling, overfitting diagnostics, and statistical significance testing — to compare a simple Linear Regression baseline against XGBoost.

## Project Overview

Model evaluation is often treated as an afterthought, but the choice of evaluation method can change the reported conclusion. This project answers four questions empirically:

1. **How sensitive are performance estimates to random train/test splits?**
   (`baseline.py` — 10 repeated splits, reports mean and std of R²)
2. **How does cross-validation improve reliability over a single split?**
   (`crossval.py` — 5-fold CV; `bootstrap.py` — 1000-resample bootstrap with out-of-bag evaluation and a 95% confidence interval)
3. **Does model complexity increase generalization uncertainty?**
   (`overfitting_check.py` — train/test R² gap; `bias_variance.py` — bias-variance tradeoff curve across XGBoost `max_depth`)
4. **Are performance differences between models statistically significant?**
   (`significance.py` — paired t-test between XGBoost and Linear Regression across matched CV folds)

Key takeaway (see `notes.md` for full write-up): XGBoost has a larger train/test gap than Linear Regression (higher overfitting risk), but its performance advantage (R² ≈ 0.83 vs. 0.60) is large, consistent across every fold, and highly statistically significant (p ≈ 2e-6) — so the added complexity is justified on this dataset.

## Tech Stack

- **Python 3**
- **pandas / numpy** — data handling and numerical computation
- **scikit-learn** — `LinearRegression`, `KFold`, `cross_val_score`, `Pipeline`, `StandardScaler`, `train_test_split`, evaluation metrics
- **XGBoost** — `XGBRegressor`
- **SciPy** (`scipy.stats`) — paired t-test for significance testing
- **Matplotlib** — visualization (e.g. bias-variance tradeoff curve)

## Quick Start / How to Run

**Requirements:** Python 3.9+

```bash
pip install numpy pandas scikit-learn xgboost scipy matplotlib
```

Each script is self-contained and uses scikit-learn's built-in California Housing dataset (fetched automatically, no manual download needed). `baseline.py` defines the shared `X`/`y` data that several other scripts import, so running any of them will also execute the baseline experiment.

```bash
python eda.py                 # exploratory data analysis
python baseline.py            # repeated random splits
python crossval.py            # 5-fold cross-validation
python bootstrap.py           # bootstrap resampling + confidence interval
python overfitting_check.py   # train/test gap: XGBoost vs Linear Regression
python bias_variance.py       # bias-variance tradeoff curve (saves a PNG)
python significance.py        # paired t-test on CV fold scores
```

## Project Structure

```
eda.py                              # exploratory data analysis
baseline.py                         # repeated train/test split baseline
crossval.py                         # k-fold cross-validation
bootstrap.py                        # bootstrap resampling + OOB evaluation
overfitting_check.py                # train vs. test overfitting diagnostic
bias_variance.py                    # bias-variance tradeoff curve
significance.py                     # paired t-test for model comparison
notes.md                            # detailed experiment log and findings
Bias-Variance Tradeoff Curve.png    # output plot from bias_variance.py
```
