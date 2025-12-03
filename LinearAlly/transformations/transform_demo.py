# transformations/transform_demo.py

import numpy as np

def translation_matrix(tx, ty):
    return np.array([
        [1, 0, tx],
        [0, 1, ty],
        [0, 0, 1]
    ])

def rotation_matrix(theta_deg):
    theta = np.radians(theta_deg)
    return np.array([
        [np.cos(theta), -np.sin(theta), 0],
        [np.sin(theta),  np.cos(theta), 0],
        [0, 0, 1]
    ])

def scaling_matrix(sx, sy):
    return np.array([
        [sx, 0, 0],
        [0, sy, 0],
        [0, 0, 1]
    ])

def shear_matrix(shx, shy):
    return np.array([
        [1, shx, 0],
        [shy, 1, 0],
        [0, 0, 1]
    ])

def apply_transformation(points, matrix):
    """Apply 3x3 transformation to Nx2 points."""
    ones = np.ones((points.shape[0], 1))
    homo_points = np.hstack([points, ones])
    transformed = homo_points @ matrix.T
    return transformed[:, :2]
