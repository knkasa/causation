import numpy as np
from sklearn.decomposition import FastICA
from sklearn.preprocessing import StandardScaler
import itertools
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Generate data: X1 → X2 → X3
np.random.seed(42)
n_samples = 5000

e1 = np.random.laplace(size=n_samples)
e2 = np.random.laplace(size=n_samples)
e3 = np.random.laplace(size=n_samples)

X1 = e1
X2 = 0.8 * X1 + e2
X3 = 0.5 * X2 + e3

X = np.c_[X2, X1, X3]

# 2. Standardize
X_std = StandardScaler().fit_transform(X)

# 3. ICA
ica = FastICA(n_components=3, random_state=42)
S_ = ica.fit_transform(X_std)
A_ = ica.mixing_
W = np.linalg.inv(A_)

# 4. Try all permutations to find best B.
best_perm = None
best_score = np.inf
B_best = None

for perm in itertools.permutations(range(3)):
    P = np.eye(3)[list(perm)]
    W_perm = np.matmul(P, W)
    
    #B = W_perm / np.diag(W_perm)[:, None]
    B = W_perm / np.diag(W_perm)[:]
    
    # Zero diagonal
    B_offdiag = B - np.eye(3)
    #score = np.sum(np.abs(np.triu(B_offdiag, k=0)))  # penalize upper triangular values (this is not the best way)

    # Find permutation such that sum of off-diagonals are smallest.  
    # Usually, this is the best results as the weights are expected to be smaller than 1 if input data are standardized.
    score = np.sum( np.abs(B_offdiag) ) 
    
    if score < best_score:
        best_score = score
        best_perm = perm
        B_best = B

# 5. Threshold small values
B_final = B_best.copy()
B_final[np.abs(B_final) < 0.1] = 0

# 6. Output
print(f"Best permutation: {best_perm}")
print("Estimated Causal Matrix B (after permutation):")
print(np.round(B_final, 2))

#================== Network Graph ==============================
import networkx as nx

# 7. Create causal graph from B matrix
G = nx.DiGraph()
var_names = ["X1", "X2", "X3"]

# Add nodes
#G.add_nodes_from(var_names)

# Add edges (from j to i if B[i, j] != 0 and i != j)
for i in range(B_final.shape[0]):
    for j in range(B_final.shape[1]):
        if i != j and np.abs(B_final[i, j]) > 0:
            G.add_edge(var_names[j], var_names[i], weight=np.round(B_final[i, j], 2))

# 8. Draw the graph
pos = nx.spring_layout(G, seed=42)  # You can try shell_layout or kamada_kawai_layout
edge_labels = nx.get_edge_attributes(G, 'weight')

plt.figure(figsize=(6, 4))
nx.draw(G, pos, with_labels=True, node_size=1500, node_color="skyblue", font_size=12, font_weight="bold", arrows=True)
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=10)
plt.title("Estimated Causal Graph (LiNGAM)")
plt.show()
