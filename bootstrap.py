import numpy as np
from baseline import X,y
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

n_bootstrap = 1000 # number of repetitions = 1000
bootstrap_r2_scores = []
rng = np.random.default_rng(42)
for i in range(n_bootstrap):
    # Draw a bootstrap sample (with replacement) from X and y, same size as the original data
    row_num = X.shape[0]
    boot_indices = rng.choice(row_num, size = row_num, replace = True)
    X_boot = X.iloc[boot_indices]
    y_boot = y.iloc[boot_indices]
    # Use the row indices that were not sampled as the test set (out-of-bag)
    oob_indices = np.setdiff1d(np.arange(row_num), boot_indices)
    X_test_set = X.iloc[oob_indices]
    y_test_set = y.iloc[oob_indices]
    # Standardize + train LinearRegression
    pipe = Pipeline([('scaler', StandardScaler()),('LinReg', LinearRegression())])
    X_boot_scaled, y_boot_scaled = pipe.fit(X_boot, y_boot) # pipe.fit calls fit_transform directly, then calls LinReg's fit. Note the scaler only stores its parameters when fit is called; afterwards it keeps transforming new data until pipe.fit is called again
    y_pred_test_set = pipe.predict(X_test_set) # pipe.predict only transforms, then calls LinReg's predict
    # Compute R^2
    r2 = round(r2_score(y_test_set, y_pred_test_set),4)
    bootstrap_r2_scores.append(r2)
bootstrap_r2_scores = np.array(bootstrap_r2_scores)
# Check for anomalies
worst_indices = np.argsort(bootstrap_r2_scores)[:10]
print("10 lowest R² scores:", bootstrap_r2_scores[worst_indices])
print("Minimum R²:", bootstrap_r2_scores.min())
print("Median R²:", np.median(bootstrap_r2_scores))
print("Number of R² scores below 0.3:", (bootstrap_r2_scores < 0.3).sum())
# Compute the 95% confidence interval
r2_lower, r2_mean, r2_upper = np.percentile(bootstrap_r2_scores, [2.5, 50, 97.5])
print(f"95% confidence interval: [{r2_lower:.4f}, {r2_upper:.4f}], interval width: {r2_upper - r2_lower:.4f}")
print(f"Mean: {r2_mean:.4f}")
