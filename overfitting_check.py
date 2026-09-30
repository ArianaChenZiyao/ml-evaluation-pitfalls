from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from baseline import X,y

# train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.25, random_state= 42)
# Use Pipeline to train XGBoost (same parameters as in XGBoost)
pipe = Pipeline([('scaler', StandardScaler()),('xgb', XGBRegressor(n_estimators = 100, max_depth = 6, learning_rate = 0.1, random_state = 42))])
pipe.fit(X_train, y_train)
# Use pipe on train and test sets, calculate r2
y_train_pred = pipe.predict(X_train) # Predict on train set (data the model has "seen")
y_test_pred = pipe.predict(X_test) # Predict on test set (data the model has not "seen")

train_r2 = r2_score(y_train, y_train_pred)
test_r2 = r2_score(y_test, y_test_pred)

# Print r2, calculate train_r2 - test_r2
# Rule of thumb: a gap within 0.05 is usually fine; a gap over 0.1-0.15 indicates noticeable overfitting and should be reported honestly in the discussion
print(f"Train set $R^2$ = {train_r2:.4f}")
print(f"Test set $R^2$ = {test_r2:4f}")
print(f"r2_difference = {train_r2 - test_r2:.4f}")

# As a comparison, test whether linear regression is overfitting
pipe2 = Pipeline([('scaler', StandardScaler()),('LinReg', LinearRegression())])
pipe2.fit(X_train, y_train)
y_train_linreg_pred = pipe2.predict(X_train)
y_test_linreg_pred = pipe2.predict(X_test)

train_linreg_r2 = r2_score(y_train, y_train_linreg_pred)
test_linreg_r2 = r2_score(y_test, y_test_linreg_pred)

print(f"LinReg train set $R^2$ = {train_linreg_r2:.4f}")
print(f"LinReg test set $R^2$ = {test_linreg_r2:4f}")
print(f"LinReg_r2_difference = {train_linreg_r2 - test_linreg_r2:.4f}")
