import numpy as np
from sklearn.decomposition import FastICA
from sklearn.preprocessing import StandardScaler
import itertools
import matplotlib.pyplot as plt
import seaborn as sns

# Try bunch of different random seeds, and find the smallest off-diagonal sums -> This will be the likely solution.

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

best_score = np.inf
B_best = None

for n in range(10):
  ica = FastICA( n_components=X.shape[1], random_state=n)
  _S = ica.fit_transform(X_std)
  _A = ica.mixing_
  W = np.linalg.inv(_A)

  B = W/np.diag(W)[:]  # make diagonal one.
  B_offdiag = B - np.eye(X.shape[1])

  score = np.sum( np.abs(B_offdiag) )

  if score<best_score:
    best_score = score
    B_best = B

print("Best matrix B:")
print(np.round(B_best,2))

