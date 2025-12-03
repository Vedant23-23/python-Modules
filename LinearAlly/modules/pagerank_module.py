import numpy as np

def pagerank(M, alpha=0.85, max_iter=100, tol=1e-6):
    n = M.shape[0]
    
    # Ensure column-stochastic matrix
    for i in range(n):
        if M[:, i].sum() == 0:
            M[:, i] = 1.0 / n
        else:
            M[:, i] /= M[:, i].sum()
    
    # Google Matrix
    G = alpha * M + (1 - alpha) * np.ones((n, n)) / n

    # Power Method
    x = np.ones(n) / n
    for _ in range(max_iter):
        x_new = G @ x
        if np.linalg.norm(x_new - x, 1) < tol:
            break
        x = x_new
    return x / x.sum()
