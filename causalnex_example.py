# Example of causalnex
# https://erdogant.medium.com/six-causal-libraries-compared-which-bayesian-approach-finds-hidden-causes-in-your-data-9fa66fd02825

# Load libraries
from causalnex.structure.notears import from_pandas
from causalnex.network import BayesianNetwork
import networkx as nx
import datazets as dz
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
le = LabelEncoder()

# Import data set and drop continous and sensitive features
df = dz.get(data='census_income')

# Clean
drop_cols = ['age', 'fnlwgt', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week', 'race', 'sex']
df.drop(labels=drop_cols, axis=1, inplace=True)

# Next, we want to make our data numeric, since this is what the NOTEARS expect.
df_num = df.copy()
for col in df_num.columns:
    df_num[col] = le.fit_transform(df_num[col])

# Structure learning
sm = from_pandas(df_num)

# Thresholding
sm.remove_edges_below_threshold(0.8)

# Make plot
plt.figure(figsize=(15,10));
edge_width = [ d['weight']*0.3 for (u,v,d) in sm.edges(data=True)]
nx.draw_networkx(sm, node_size=400, arrowsize=20, alpha=0.6, edge_color='b', width=edge_width)

# If required, remove spurious edges and relearn structure.
sm = from_pandas(df_num, tabu_edges=[("relationship", "native-country")], w_threshold=0.8)

# Step 1: Create a new instance of BayesianNetwork
bn = BayesianNetwork(sm)

# Step 2: Reduce the cardinality of categorical features
# Use domain knowledge or other manners to remove redundant features.

# Step 3: Create Labels for Numeric Features
# Create a dictionary that describes the relation between numeric value and label.

# Step 4: Specify all of the states that each node can take
bn = bn.fit_node_states(df)

# Step 5: Fit Conditional Probability Distributions
bn = bn.fit_cpds(df, method="BayesianEstimator", bayes_prior="K2")

# Return CPD for education
result = bn.cpds["education"]

# Extract any information and probabilities related to education.
print(result)

# marital-status  Divorced              ...   Widowed            
# salary             <=50K              ...      >50K            
# workclass              ? Federal-gov  ... State-gov Without-pay
# education                             ...                      
# 10th            0.077320    0.019231  ...  0.058824      0.0625
# 11th            0.061856    0.012821  ...  0.117647      0.0625
# 12th            0.020619    0.006410  ...  0.058824      0.0625
# 1st-4th         0.015464    0.006410  ...  0.058824      0.0625
# 5th-6th         0.010309    0.006410  ...  0.058824      0.0625
# 7th-8th         0.056701    0.006410  ...  0.058824      0.0625
# 9th             0.067010    0.006410  ...  0.058824      0.0625
# Assoc-acdm      0.025773    0.057692  ...  0.058824      0.0625
# Assoc-voc       0.046392    0.051282  ...  0.058824      0.0625
# Bachelors       0.097938    0.128205  ...  0.058824      0.0625
# Doctorate       0.005155    0.006410  ...  0.058824      0.0625
# HS-grad         0.278351    0.333333  ...  0.058824      0.0625
# Masters         0.015464    0.032051  ...  0.058824      0.0625
# Preschool       0.005155    0.006410  ...  0.058824      0.0625
# Prof-school     0.015464    0.006410  ...  0.058824      0.0625
# Some-college    0.201031    0.314103  ...  0.058824      0.0625
# [16 rows x 126 columns]