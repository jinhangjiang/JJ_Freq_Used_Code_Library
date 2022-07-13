from sklearn.model_selection import cross_validate 
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, make_scorer
import numpy as np

def smape(A, F):
    return 100/len(A) * np.sum(2 * np.abs(F - A) / (np.abs(A) + np.abs(F)))

scoring = {'RMSE': make_scorer(mean_squared_error, squared=False),
           'R2': make_scorer(r2_score),
           'smape': make_scorer(smape),
           'MAE': make_scorer(mean_absolute_error)
           }
cross_validation_op = cross_validate(estimator, X, y, cv=5, scoring=scoring)

avgDict = {}
for k,v in cross_validation_op.items():
    # v is the list of grades for student k
    avgDict[k] = sum(v)/ float(len(v))

print(avgDict)
