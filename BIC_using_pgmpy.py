import pandas as pd
from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler
from pgmpy.estimators import PC, HillClimbSearch, BicScore, K2Score, TreeSearch, ExhaustiveSearch, MmhcEstimator
import networkx as nx
import matplotlib.pyplot as plt

# Causation using Bayesian network (BIC=Bayesian inference causation) using pgmpy
# Note: pgmpy=0.1.26.  Do not use pgmpy>1.0.0

wine = load_wine()
df = pd.DataFrame(wine.data, columns=wine.feature_names)
df['Wine Class'] = wine.target  # target variable

scaler = StandardScaler()
df_scaled = scaler.fit_transform(df)
df = pd.DataFrame(df_scaled, columns=df.columns)

# Binning
df_binned = df.copy()
for col in wine.feature_names:
    df_binned[col] = pd.cut(df_binned[col], 5, labels=False)
#df_binned['Wine Class'] = pd.cut(df_binned['Wine Class'], 3, labels=False)

# Estimate DAG using PC algorithm
#est = PC(data=df_binned)
#model = est.estimate()

# Estimate using Hill climb search(BIC)
#est = HillClimbSearch(data=df_binned)
#model = est.estimate(scoring_method=BicScore(df_binned))

# Estimate using K2Score
#est = HillClimbSearch(df_binned)
#model = est.estimate(scoring_method=K2Score(df_binned))

# Estimate using Tree search
#est = TreeSearch(df_binned)
#model = est.estimate()

# Estimate using Exhaustive search (number of features need to be less than 6)
#est = ExhaustiveSearch(df_binned)
#model = est.estimate()

# Extimate using Mmhc (takes long time)
est = MmhcEstimator(df_binned)
model = est.estimate()

g = nx.DiGraph()
g.add_edges_from(model.edges())

# Save the graph to GML file
#nx.write_gml(g, 'wine_dag.gml')
pos = nx.spring_layout(g)
nx.draw(g, pos, with_labels=True, node_color='skyblue', edge_color='gray')
plt.title("Estimated DAG from Wine Dataset")
plt.show()


#======= Calculate strength of connection(edge) ================
'''
from sklearn.metrics import mutual_info_score

# Calculate mutual information between each connected variable
def compute_mutual_info(df, edges):
    mi_dict = {}
    for u, v in edges:
        mi = mutual_info_score(df[u], df[v])
        mi_dict[(u, v)] = mi
    return mi_dict

# Calculate MI and build graph with weights
edges = model.edges()
mi_scores = compute_mutual_info(df_binned, edges)

g = nx.DiGraph()
for (src, dst), mi in mi_scores.items():
    g.add_edge(src, dst, weight=mi)

# Save with weights
nx.write_gml(g, 'wine_dag_weighted.gml')

pos = nx.spring_layout(g)
edge_labels = nx.get_edge_attributes(g, 'weight')
nx.draw(g, pos, with_labels=True, node_color='lightblue', edge_color='gray')
nx.draw_networkx_edge_labels(g, pos, edge_labels={k: f"{v:.2f}" for k, v in edge_labels.items()})
plt.title("DAG with Mutual Information Weights")
plt.show()

'''

#=========== get Bayesian probability ======================
'''
from pgmpy.models import BayesianNetwork
from pgmpy.estimators import BayesianEstimator

# Build BN with learned structure
model = BayesianNetwork(best_model.edges())

# Learn the CPDs from data
model.fit(df_binned, estimator=BayesianEstimator)

# Now you can query probabilities
print(model.get_cpds('some_variable'))
'''
