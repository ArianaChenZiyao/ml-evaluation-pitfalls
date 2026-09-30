## EDA
### EDA
- Dataset: California Housing, 20640 samples, 8 features, 1 target (MedHouseVal)
- Missing values: none
- Features most correlated with target: MedInc, AveRooms
- High-correlation feature pairs (possible multicollinearity):
  [AveRooms, AveBedrms], [Latitude, Longitude]
- Known issue: target values are censored at 5.0 (992 samples affected), a real limitation of this dataset from the original data collection process

## Research Questions
1. How sensitive are performance estimates to random train-test splits?
2. How does cross-validation improve reliability?
3. Does model complexity increase generalization uncertainty?
4. Are performance differences between models statistically significant?

## Method A: Single Random Split (×10 repeats) — Linear Regression
- 10 R² values: [0.5839, 0.6182, 0.6159, 0.6014, 0.6067, 0.6081, 0.5806, 0.6059, 0.6107, 0.6347]
- Mean = 0.6066, Std = 0.015
- Debugging notes: initially got much lower/unstable results due to two bugs
  — StandardScaler output not assigned back to a variable, and r2_score called with arguments in the wrong order (r2_score(y_true, y_pred) is required; scores are NOT symmetric to argument order, unlike MSE)

## Method B: 5-Fold Cross-Validation — Linear Regression
- 5 R² values: [0.57578771, 0.61374822, 0.60856043, 0.62126494, 0.5875292]
- Mean = 0.6014, Std = 0.017
- Debugging note: KFold defaults to shuffle=False; without shuffle=True, results were initially unstable because this dataset is sorted by geographic location, causing unrepresentative folds
- Consistent with Method A — single-split and 5-fold CV agree closely on this dataset, likely because the sample size (20640) is large enough that single-split variance is limited. This may not hold on smaller datasets.

## Method C: Bootstrap (1000 resamples, Out-of-Bag evaluation) — Linear Regression
- Mean R² ≈ 0.60, median ≈ 0.60
- 95% CI (percentile method): wide, approx [0.04, 0.62]
- ~3-4% of resamples (31-38 out of 1000) produced R² < 0.3, including extreme negative values (min ≈ -8.3)
- Verified with a fixed random seed across two runs — the extreme-value phenomenon is real and reproducible, not a one-off coding artifact
- Takeaway: even a more rigorous method like Bootstrap has its own failure modes (a long tail of unstable resamples), which a simple standard-deviation-based estimate (Methods A/B) would not reveal

## Model Comparison: Linear Regression vs XGBoost (5-Fold CV, same KFold split)
| Model | Mean R² | Std |
|---|---|---|
| Linear Regression | 0.6014 | 0.017 |
| XGBoost (n_estimators=100, max_depth=6, learning_rate=0.1) | 0.8321 | 0.0085 |

## Overfitting Check (train vs test R², fixed max_depth=6)
| Model | Train R² | Test R² | Gap |
|---|---|---|---|
| XGBoost | 0.8986 | 0.8311 | 0.0675 |
| Linear Regression | 0.6099 | 0.5911 | 0.0188 |
XGBoost shows a train/test gap over 3x larger than Linear Regression — consistent with higher overfitting risk for more complex models.

## Bias-Variance Tradeoff Curve (max_depth = 1 to 20, three-way train/val/test split)
- Train R²: rises steadily from 0.651 to 0.9998 as depth increases
- Validation R²: rises to a peak around depth=6-8 (~0.832), then declines to ~0.792 by depth=20 — the classic underfit → optimal → overfit shape
- Best depth = 8 (validation R² = 0.8327)
- Final held-out test R² (never touched during model selection) = 0.8330 
    — nearly identical to the validation estimate, confirming the model selection process was not overfit to the validation set

## Significance Testing (Paired t-test, XGBoost vs Linear Regression, same 5 folds)
- t-statistic = 39.5092
- p-value = 0.000002
- Per-fold difference (XGB - LinReg): [0.250, 0.223, 0.217, 0.225, 0.238]
- Mean difference = 0.2307, Std of difference = 0.0117
- Conclusion: XGBoost's advantage is highly statistically significant and consistent across every fold — not attributable to chance.
- Limitation: a standard paired t-test assumes independence between folds; CV folds share overlapping training data, so a Nadeau-Bengio corrected variance estimate would be more rigorous (not implemented here due to time constraints — noted as a direction for future work)

## Overall Synthesis
XGBoost carries a higher overfitting risk than Linear Regression (larger train/test gap), but its absolute performance advantage is large enough that it remains significantly and consistently better even accounting for that risk. This is a more nuanced conclusion than either"complex models are always better" or "complex models are always riskier" — both risk and reward increase with model complexity, and in this case the reward outweighs the risk.