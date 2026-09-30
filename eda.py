"""
Environment setup + data loading + exploratory data analysis (EDA)
=================================================
Goal: get familiar with the California Housing dataset to prepare for modeling
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing

# ---------------------------------------------------------
# Load data (built into sklearn, no need to fetch extra data online)
# ---------------------------------------------------------
housing = fetch_california_housing(as_frame=True)
df = housing.frame # This is a pandas DataFrame containing both features and target
# fetch is a function, by using as_frame = True, we turned the function into a package of data (e.g. including data, target and frame). By using .frame, we use the dataframe in housing.

print("=" * 60)
print("Dataset Basic Info")
print("=" * 60)
print(f"Number of samples: {df.shape[0]}, Number of features: {df.shape[1] - 1}")
# df.shape[0] = number of rows, df.shape[1] = number of columns
print(f"\nFeature list: {list(df.columns)}")
print(f"\nDataset description:\n{housing.DESCR[:800]}")  # Print part of the official description; DESCR = description, [:800] is Python string slicing syntax

# ---------------------------------------------------------
# Check missing values (California Housing is usually clean, but it's good practice)
# ---------------------------------------------------------
print("\n" + "=" * 60)
print("Missing Value Check")
print("=" * 60)
missing = df.isnull().sum()
# df.isnull() checks every cell in df, marking 1 if missing and 0 if not. df.isnull().sum() sums each column, returning a pandas Series with one count per column, e.g. MedInc:0, HouseAge:0 ...
if missing.sum() > 0:
    print(f"missing number = {missing.sum()}")
else:
    print(f"no missing value")

# ---------------------------------------------------------
# Check basic statistics (value ranges, obvious outliers)
# ---------------------------------------------------------
print("\n" + "=" * 60)
print("Descriptive Statistics")
print("=" * 60)
print(df.describe().round(2))

# ---------------------------------------------------------
# Check the distribution of the target (median house value, MedHouseVal)
# ---------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
# subplots(1,2) means 1 row, 2 columns
axes[0].hist(df['MedHouseVal'], bins=50, edgecolor='black')
axes[0].set_title('Target Distribution: Median House Value ($100k)')
axes[0].set_xlabel('MedHouseVal')
axes[0].set_ylabel('Frequency')

# Note: the target in this dataset is clearly capped at 5.0 (the original data collection had a ceiling)
# This is a data limitation worth mentioning in the README
print(f"\nNote: {(df['MedHouseVal'] >= 4.999).sum()} samples are capped near the target value of 5.0, "
      f"which is a ceiling effect from data collection and a known limitation of this dataset.")

# ---------------------------------------------------------
# Check correlations between features (check for obvious multicollinearity)
# ---------------------------------------------------------
corr = df.corr()
im = axes[1].imshow(corr, cmap='coolwarm', vmin=-1, vmax=1)
axes[1].set_xticks(range(len(corr.columns)))
axes[1].set_yticks(range(len(corr.columns)))
axes[1].set_xticklabels(corr.columns, rotation=90, fontsize=8)
axes[1].set_yticklabels(corr.columns, fontsize=8)
axes[1].set_title('Feature Correlation Matrix')
plt.colorbar(im, ax=axes[1])
plt.show()

# Find the features most correlated with the target
print("\n" + "=" * 60)
print("Correlation with Target (sorted)")
print("=" * 60)
print(corr['MedHouseVal'].sort_values(ascending=False))

# Find feature pairs with high correlation (multicollinearity warning, >0.8 is considered high)
print("\n" + "=" * 60)
print("High-Correlation Feature Pairs (|r| > 0.8, possible multicollinearity)")
print("=" * 60)
high_corr_pairs = []
for i in range(len(corr.columns)):
    for j in range(i+1, len(corr.columns)):
        if abs(corr.iloc[i, j]) > 0.8 and corr.columns[i] != 'MedHouseVal' and corr.columns[j] != 'MedHouseVal':
            high_corr_pairs.append((corr.columns[i], corr.columns[j], round(corr.iloc[i, j], 3)))
print(high_corr_pairs if high_corr_pairs else "No excessively high correlations found between features")
