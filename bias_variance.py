import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from sklearn.metrics import r2_score
from baseline import X,y
# Split train, validation and test sets
X_temp, X_test, y_temp, y_test = train_test_split(X,y,test_size = 0.20, random_state = 42 )
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size = 0.25, random_state = 42)
# Define the list of max_depth values
depths = [1,2,3,4,5,6,8,10,15,20]
train_r2_list = []
val_r2_list = []
# Loop over each depth
for i in depths:
    pipe = Pipeline([('std',StandardScaler()),('xgb',XGBRegressor(n_estimators = 100, max_depth = i, learning_rate = 0.1, random_state = 42))])
    pipe.fit(X_train, y_train)
    y_train_pred = pipe.predict(X_train)
    y_val_pred = pipe.predict(X_val)
    train_r2 = r2_score(y_train, y_train_pred)
    val_r2 = r2_score(y_val, y_val_pred)
    train_r2_list.append(train_r2)
    val_r2_list.append(val_r2)
print(train_r2_list, val_r2_list)
# Sketch graph
x = depths
line1 = train_r2_list
line2 = val_r2_list

fig, ax = plt.subplots(figsize = (8,5))
ax.plot(x, line1, marker = 'o', label = 'train_r2')
ax.plot(x, line2, marker = 'o', label = 'val_r2')
ax.set_xlabel('max depth')
ax.set_ylabel('r2 score')
ax.set_title('Bias-Variance Tradeoff Curve')
ax.legend()
ax.grid(alpha = 0.3)

plt.tight_layout
plt.savefig('Bias-Variance Tradeoff Curve.png', dpi = 120)
plt.show()
# Find the best depth
index = np.argmax(val_r2_list)
best_depth = depths[index]
print(f'best depth = {best_depth}, with r2 = {val_r2_list[index]:.4f}')
# Use XGBoost model with depth = 8 on the test set
model = Pipeline([('std', StandardScaler()),('xgb', XGBRegressor(n_estimators = 100, max_depth = 8, learning_rate = 0.1, random_state = 42))])
model.fit(X_train, y_train)
y_test_pred = model.predict(X_test)
test_r2 = r2_score(y_test, y_test_pred)
print(f'test_r2 = {test_r2:.4f}')