import pandas as pd
import numpy as np
from sklearn.feature_selection import mutual_info_classif
from itertools import combinations

# Bayesian Inference Causation (BIC) from scratch.
# Use of Bayesian network (Hill climb approach) to perform BIC.

def is_independent(df, X, Y, conditioned_on=None, threshold=0.01):
    """Check if X and Y are conditionally independent given Z using mutual information."""
    data = df.copy()

    if conditioned_on is None or len(conditioned_on) == 0:
        mi = mutual_info_classif(data[[X]], data[Y], discrete_features=True)[0]
    else:
        from sklearn.linear_model import LogisticRegression
        # Predict Y using X + Z vs Z only
        clf_full = LogisticRegression(max_iter=1000).fit(data[[X] + conditioned_on], data[Y])
        clf_reduced = LogisticRegression(max_iter=1000).fit(data[conditioned_on], data[Y])
        score_full = clf_full.score(data[[X] + conditioned_on], data[Y])
        score_reduced = clf_reduced.score(data[conditioned_on], data[Y])
        delta = score_full - score_reduced
        mi = delta

    return mi < threshold

def learn_skeleton(df):
    variables = list(df.columns)
    edges = set()

    for (X, Y) in combinations(variables, 2):
        if not is_independent(df, X, Y):
            edges.add(frozenset((X, Y)))  # undirected

    return edges

def orient_colliders(df, edges):
    graph = {var: set() for var in df.columns}
    for edge in edges:
        a, b = tuple(edge)
        graph[a].add(b)
        graph[b].add(a)

    directed = set()

    # Try to detect triplets: A - C - B (unconnected A-B)
    for (A, B, C) in combinations(df.columns, 3):
        if frozenset((A, C)) in edges and frozenset((B, C)) in edges and frozenset((A, B)) not in edges:
            # check if C is a collider: A → C ← B
            if is_independent(df, A, B, conditioned_on=[]):  # not a collider
                continue
            if not is_independent(df, A, B, conditioned_on=[C]):
                directed.add((A, C))
                directed.add((B, C))

    return directed

def estimate_dag(df):
    skeleton = learn_skeleton(df)
    directed = orient_colliders(df, skeleton)

    # Convert to edge list with direction
    dag = []
    for edge in skeleton:
        a, b = tuple(edge)
        if (a, b) in directed:
            dag.append((a, b))
        elif (b, a) in directed:
            dag.append((b, a))
        else:
            dag.append((a, b))  # unknown direction (can add heuristics)

    return dag

df = pd.DataFrame({
    'Rain': [0, 0, 1, 1, 0, 1, 0, 1],
    'Sprinkler': [0, 1, 0, 1, 1, 0, 0, 1],
    'WetGrass': [0, 1, 1, 1, 1, 1, 0, 1]
  })

dag = estimate_dag(df)

print("Estimated DAG edges (with direction where known):")
for edge in dag:
    print(f"{edge[0]} --> {edge[1]}")
