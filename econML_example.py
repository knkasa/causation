import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestRegressor
from econml.dml import CausalForestDML
import matplotlib.pyplot as plt

# https://medium.com/data-science-collective/what-happened-when-i-put-a-causal-ml-model-to-the-test-514210f3da77
# https://medium.com/analytics-vidhya/a-world-of-causal-inference-with-econml-by-microsoft-research-7dd43e97ce09
# https://qiita.com/yellow_detteiu/items/e0915ef1042a6af49382
# https://www.salesanalytics.co.jp/datascience/datascience186/
# sample example https://www.salesanalytics.co.jp/datascience/datascience187/
#paper https://github.com/grf-labs/grf/?tab=readme-ov-file
# parameters https://econml.azurewebsites.net/_autosummary/econml.dml.CausalForestDML.html
# code example https://saltcooky.hatenablog.com/entry/2024/08/18/172203

np.random.seed(42)

# sample data with user info.
n_samples = 10000  # number of sample needs to be large.
age = np.random.normal(35, 10, n_samples).astype(int)           # Age
education_years = np.random.normal(12, 2, n_samples)                # Years of education
prior_income = np.random.normal(30000, 10000, n_samples)           # Prior income
X = np.column_stack([age, education_years, prior_income])

# Create treatment effect to the above users (binary data.  1=Treated  0=not treated)
# People with lower income and less education more likely to attend.
propensity = 1 / (1 + np.exp(0.01 * prior_income - 0.5 * education_years))
T = np.random.binomial(1, propensity)
print("number of treatments [T=0, T=1]:", np.bincount(T))  # Should show both T=0 and T=1

# Create Output data Y with treatment intentionaly included.
treatment_effect = 2000 + 100 * education_years - 0.05 * prior_income
Y = 30000 + 1500 * education_years + 0.6 * prior_income + T * treatment_effect + np.random.normal(0, 5000, n_samples)

X_train, X_test, T_train, T_test, Y_train, Y_test = train_test_split(X, T, Y, test_size=0.2, random_state=42)

# cross validation data needs to have T=1 and T=0 enough and equally to train. 
cv = StratifiedKFold(n_splits=2, shuffle=True, random_state=42) 

cf = CausalForestDML(
    model_y=RandomForestRegressor(n_estimators=100, min_samples_leaf=10),
    model_t=RandomForestRegressor(n_estimators=100, min_samples_leaf=10),
    discrete_treatment=True,
    n_estimators=100,
    min_samples_leaf=5,
    verbose=0,
    random_state=42,
    cv=cv
    )

cf.tune(Y, T, X=X)
cf.fit(Y_train, T_train, X=X_train)

cf.feature_importance_

# effects show the difference between output(with treatment) - output(not treated)
effects = cf.effect(X_test)

for i in range(5):
    print(f"Individual {i+1}: Estimated treatment effect = ${effects[i]:.2f}")

plt.hist(effects, bins=30, edgecolor='k')
plt.title('Estimated Treatment Effects (Job Training on Income)')
plt.xlabel('Estimated Effect ($)')
plt.ylabel('Number of Individuals')
plt.show()
