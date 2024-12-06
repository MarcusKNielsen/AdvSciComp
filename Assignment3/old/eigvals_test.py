import numpy as np
from scipy.linalg import block_diag

np.random.seed(2)

N = 7

# Generate random matrices for the diagonal blocks
A = np.random.rand(N, N)

# Create a block diagonal matrix
K = 10
blocks = [A] * K
B = block_diag(*blocks)

for i in range(K):
    B[i*N-1,i*N] = 1
    B[i*N,i*N-1] = 1

eigs_A = np.linalg.eigvals(A)
eigs_B = np.linalg.eigvals(B)

print(np.max(np.abs(eigs_A)))
print(np.max(np.abs(eigs_B)))

