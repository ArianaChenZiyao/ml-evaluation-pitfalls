import numpy as np
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, cross_val_score
from baseline import X,y

# Build a Pipeline with standardization + XGBoost
pipe = Pipeline([('scaler', StandardScaler()), ('xgb', XGBRegressor(n_estimators = 100, max_depth = 6, learning_rate = 0.1, random_state = 42))])
# Run 5-fold cross validation
kf = KFold(n_splits = 5, shuffle = True, random_state = 42)
score = cross_val_score(pipe, X, y, cv = kf, scoring = 'r2')
print(score)
score_mean = np.mean(score)
score_std = np.std(score)
print(f'mean = {score_mean:.4f}，standard variation = {score_std:.4f}')


