import numpy as np

def compute_svd(A):
    U, S, VT = np.linalg.svd(A, full_matrices=False)
    return U, S, VT

def low_rank_approximation(U, S, VT, k):
    S_k = np.diag(S[:k])
    U_k = U[:, :k]
    VT_k = VT[:k, :]
    A_k = U_k @ S_k @ VT_k
    return A_k
