# Use of bnlearn to make DAG & inference(probability)
# https://erdogant.medium.com/six-causal-libraries-compared-which-bayesian-approach-finds-hidden-causes-in-your-data-9fa66fd02825

import datazets as dz
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = dz.import_example(data='census_income')

# Data cleaning
drop_cols = ['age', 'fnlwgt', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week', 'race', 'sex']
df.drop(labels=drop_cols, axis=1, inplace=True)

df.head()
#          workclass  education  ... native-country salary
#0         State-gov  Bachelors  ...  United-States  <=50K
#1  Self-emp-not-inc  Bachelors  ...  United-States  <=50K
#2           Private    HS-grad  ...  United-States  <=50K
#3           Private       11th  ...  United-States  <=50K
#4           Private  Bachelors  ...           Cuba  <=50K
#[5 rows x 7 columns]

import bnlearn as bn

# Structure learning. other method type=ex(PC algorithm), tan, chow-liu, exhaustivesearch.
model = bn.structure_learning.fit(df, methodtype='hillclimbsearch', scoretype='bic')

# Test edges significance and remove.
model = bn.independence_test(model, df, test="chi_square", alpha=0.05, prune=True)

# Learn the parameters (optional)
model = bn.parameter_learning.fit(model, df)

# Make plot
G = bn.plot(model, interactive=False)

# Ceate dotgraph plot
dotgraph = bn.plot_graphviz(model)
dotgraph.view(filename=r'c:/temp/bnlearn_plot')

# Make plot interactive(not needed.)
G = bn.plot(model, interactive=True)
# Show edges
print(model['model_edges'])
"""
[('education', 'occupation'),
 ('marital-status', 'relationship'),
 ('occupation', 'workclass'),
 ('relationship', 'salary'),
 ('relationship', 'education'),
 ('salary', 'education')]
"""

# Be warned that sometimes that arrow points in opposite direction compared to your intuition.
# In this case, you should use other DAG approach like PC Algorithm.
# Or, manually create DAG as below.
# model = bn.make_DAG(model['model_edges'])

# Learn the CPD by providing the model and dataframe
model = bn.parameter_learning.fit(model, df)

# Print the CPD
CPD = bn.print_CPD(model)
CPD of salary:
#+----------------+---------------+---------------+
#| education      | salary(<=50K) | salary(>50K)  |
#+----------------+---------------+---------------+
#| HS-grad        | 0.85          | 0.15          |
#| Bachelors      | 0.70          | 0.30          |
#| Doctorate      | 0.29          | 0.71          |
#| Masters        | 0.45          | 0.55          |
#| Others         | 0.90          | 0.10          |

# Question 1: What is the probability of having a salary > 50k given the education is Doctorate:
# Start making inferences
query = bn.inference.fit(model, variables=['salary'], evidence={'education':'Doctorate'})
print(query)

"""
+---------------+---------------+
| salary        |   phi(salary) |
+===============+===============+
| salary(<=50K) |        0.2907 |
+---------------+---------------+
| salary(>50K)  |        0.7093 |
+---------------+---------------+

Summary for variables: ['salary']
Given evidence: education=Doctorate

salary outcomes:
- salary: <=50K (29.1%)
- salary: >50K (70.9%)
"""

# Question 3: What is the probability of being in a certain workclass given that education is Doctorate and the marital status is never-married. 
# Start making inferences
query = bn.inference.fit(model, variables=['workclass'], evidence={'education':'Doctorate', 'marital-status':'Never-married'})
print(query)

"""
+----+------------------+------------+
|    | workclass        |          p |
+====+==================+============+
|  0 | ?                | 0.0420424  |
+----+------------------+------------+
|  1 | Federal-gov      | 0.0420328  |
+----+------------------+------------+
|  2 | Local-gov        | 0.132582   |
+----+------------------+------------+
|  3 | Never-worked     | 0.0034366  |
+----+------------------+------------+
|  4 | Private          | 0.563884   | <--- HIGHEST PROBABILITY
+----+------------------+------------+
|  5 | Self-emp-inc     | 0.0448046  |
+----+------------------+------------+
|  6 | Self-emp-not-inc | 0.0867973  |
+----+------------------+------------+
|  7 | State-gov        | 0.0810306  |
+----+------------------+------------+
|  8 | Without-pay      | 0.00338961 |
+----+------------------+------------+

Summary for variables: ['workclass']
Given evidence: education=Doctorate, marital-status=Never-married

workclass outcomes:
- workclass: ? (4.2%)
- workclass: Federal-gov (4.2%)
- workclass: Local-gov (13.3%)
- workclass: Never-worked (0.3%)
- workclass: Private (56.4%)
- workclass: Self-emp-inc (4.5%)
- workclass: Self-emp-not-inc (8.7%)
- workclass: State-gov (8.1%)
- workclass: Without-pay (0.3%)

"""



