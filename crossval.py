import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import Pipeline
from baseline import X,y

housing = fetch_california_housing(as_frame = True) # Load data
df = housing.frame
# 5-fold cross validation using KFold
pipe = Pipeline([('scaler', StandardScaler()), ('LinReg', LinearRegression())]) # StandardScaler must not be fit on all the data, only on each fold, so a pipeline is required
kf = KFold(n_splits = 5, shuffle = True, random_state = 42)
score = cross_val_score(pipe, X, y, cv = kf, scoring = 'r2')
print(score)
# Mean & std of the 5 scores
score_mean = round(np.mean(score),4)
score_std = round(np.std(score),4)
print(score_mean, score_std)
