# linear_systems/system_solver.py

import numpy as np
from scipy.linalg import lu, qr

def solve_linear_system(A, b):
    try:
        x = np.linalg.solve(A, b)
        residual = np.linalg.norm(A @ x - b)
        return x, residual
    except np.linalg.LinAlgError as e:
        return None, str(e)

def get_lu_decomposition(A):
    P, L, U = lu(A)
    return P, L, U

def get_qr_decomposition(A):
    Q, R = qr(A)
    return Q, R
