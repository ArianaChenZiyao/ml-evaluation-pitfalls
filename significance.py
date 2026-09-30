import numpy as np
from scipy import stats

# r2 scores from 5-fold cross validation
lin_reg_scores = [0.57578771, 0.61374822, 0.60856043, 0.62126494, 0.5875292 ]
xgb_scores = [0.8258649, 0.83720328, 0.82594315, 0.84652032, 0.82510333]

t_stat, p_value = stats.ttest_rel(xgb_scores, lin_reg_scores) # Paired t-test
print(f't-stat = {t_stat:.4f}')
print(f'p_value = {p_value:.6f}')

diff = np.array(xgb_scores)- np.array(lin_reg_scores)
print(f'Difference of each fold between XGB and LinReg is {diff}')
print(f'mean difference = {diff.mean():.4f}, Std of difference = {diff.std():.4f}')