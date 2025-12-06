# DoWhy (Microsoft) to find causal effect.
# If you have Target, Treatment, Cofounders, use either EconML or DoWhy.
# EconML: If you perturb Treatment, how much Target would change?
# DoWhy: If you want to know the strength between Treatment and Target, use this.

import dowhy
from dowhy import CausalModel

model = CausalModel(
    data=df,
    treatment='X',
    outcome='Y',
    common_causes=['Z1','Z2']
)

identified_estimand = model.identify_effect()
estimate = model.estimate_effect(identified_estimand,
                                 method_name="backdoor.linear_regression")
