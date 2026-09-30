"""
Random split + linear regression baseline
"""
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error

# Load data, split into X (features) and y (target)
housing = fetch_california_housing(as_frame = True)
df = housing.frame
X = df.drop(columns = ['MedHouseVal'])
y = df['MedHouseVal']

# Loop over 10 different random seeds
r2_list = []
rmse_list = []
for seed in range(10):
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size = 0.20, random_state = seed) # train_test_split: train = 60%, validation = 20%, test = 20%
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size = 0.25, random_state= seed)
    """ StandardScaler standardization: initialize, fit + transform on the train set,
        then transform the validation and test sets """
    scalar = StandardScaler()
    scalar.fit(X_train)
    X_train_scaled = scalar.transform(X_train)
    X_val_scaled = scalar.transform(X_val)
    X_test_scaled = scalar.transform(X_test)
    # Train linear regression
    reg = LinearRegression()
    reg.fit(X_train_scaled, y_train)
    y_pred = reg.predict(X_val_scaled)
    # calculate r2 & RMSE (mean_squared_error)
    r2 = round(r2_score(y_val, y_pred),4)
    rmse = round(mean_squared_error(y_val, y_pred),4)
    r2_list.append(r2) # put the value of r2 and rmse into arrays
    rmse_list.append(rmse)
print(r2_list)
print(rmse_list)

# Calculate the mean and std of r2
r2_mean = round(np.mean(r2_list),4)
r2_std = round(np.std(r2_list),4)
print(r2_mean, r2_std)