"""
Adjacency matrix construction for Penn Action's 13-joint skeleton.

Joint mapping:
    0=head, 1=l_shoulder, 2=r_shoulder, 3=l_elbow, 4=r_elbow,
    5=l_wrist, 6=r_wrist, 7=l_hip, 8=r_hip, 9=l_knee,
    10=r_knee, 11=l_ankle, 12=r_ankle
"""

import numpy as np


def get_penn_action_adjacency() -> np.ndarray:
    """
    Build the 13×13 adjacency matrix with self-loops.

    Returns:
        (13, 13) float32 array — binary adjacency with self-loops.
    """
    edges = [
        (0, 1), (0, 2),
        (1, 7), (2, 8), (7, 8),
        (1, 3), (3, 5),
        (2, 4), (4, 6),
        (7, 9), (9, 11),
        (8, 10), (10, 12),
    ]
    A = np.zeros((13, 13), dtype=np.float32)
    for i, j in edges:
        A[i, j] = A[j, i] = 1.0
    A += np.eye(13, dtype=np.float32)
    return A


def normalize_adjacency(A: np.ndarray) -> np.ndarray:
    """Symmetric normalisation: D^{-1/2} A D^{-1/2}."""
    D_inv_sqrt = np.diag(1.0 / np.sqrt(A.sum(axis=1)))
    return D_inv_sqrt @ A @ D_inv_sqrt
