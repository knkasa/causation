# https://arxiv.org/abs/2104.03220?utm_source=chatgpt.com
# Use of Double machine learning to estimate causation.

from doubleml import DoubleMLData, DoubleMLPLR
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

# prepare data
dml_data = DoubleMLData(df, y_col="outcome", d_cols=["treatment"], x_cols=["x1","x2","x3"])
# setup
ml_g       = RandomForestRegressor(n_estimators=100, random_state=42)
ml_m       = RandomForestClassifier(n_estimators=100, random_state=42)
dml_plr    = DoubleMLPLR(dml_data, ml_g=ml_g, ml_m=ml_m)
# fit
dml_plr.fit()
print("Estimate of causal effect:", dml_plr.coef)
