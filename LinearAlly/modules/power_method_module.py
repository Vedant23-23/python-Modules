import numpy as np

def power_method(A, max_iter=1000, tol=1e-6):
    n = A.shape[0]
    x = np.random.rand(n)
    x = x / np.linalg.norm(x)
    
    for i in range(max_iter):
        x_new = A @ x
        x_new = x_new / np.linalg.norm(x_new)
        
        if np.linalg.norm(x_new - x) < tol:
            break
        x = x_new

    eigenvalue = (x.T @ A @ x) / (x.T @ x)
    return eigenvalue, x
