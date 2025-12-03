# linear_systems/sensitivity.py

import numpy as np

def compute_matrix_norms(A):
    norms = {
        "1-Norm (max column sum)": np.linalg.norm(A, 1),
        "Infinity-Norm (max row sum)": np.linalg.norm(A, np.inf),
        "2-Norm (spectral norm)": np.linalg.norm(A, 2)
    }
    return norms

def compute_condition_number(A):
    try:
        cond_2 = np.linalg.cond(A, p=2)
        return cond_2
    except np.linalg.LinAlgError:
        return np.inf

def simulate_sensitivity(A, b, noise_level=1e-3):
    try:
        x = np.linalg.solve(A, b)
        b_perturbed = b + noise_level * np.random.randn(*b.shape)
        x_perturbed = np.linalg.solve(A, b_perturbed)
        diff = np.linalg.norm(x - x_perturbed)
        return x, x_perturbed, diff, b_perturbed
    except Exception as e:
        return None, None, None, None
