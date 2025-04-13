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

# 4. Try all permutations to find a strictly lower-triangular B
best_perm = None
best_score = np.inf
B_best = None

for perm in itertools.permutations(range(3)):
    P = np.eye(3)[list(perm)]
    W_perm = P @ W
    B = W_perm / np.diag(W_perm)[:, None]
    # Zero diagonal
    B_offdiag = B - np.eye(3)
    score = np.sum(np.abs(np.triu(B_offdiag, k=0)))  # penalize upper triangular values
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

plt.figure(figsize=(6, 4))
sns.heatmap(B_final, annot=True, fmt=".2f", cmap="coolwarm", xticklabels=["X1", "X2", "X3"], yticklabels=["X1", "X2", "X3"])
plt.title("Estimated Causal Influence Matrix B")
plt.show()
