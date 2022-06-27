import numpy as np


def euclidean_distance(x1, x2):
    """Computes the euclidean distance between two points

    Args:
        x1 (np.array): Fist point in the euclidean plane.
        x2 (np.array): Second point in the euclidean plane.

    Returns:
        float: Euclidean distance between the two points.
    """
    return np.sqrt(np.sum((x1 - x2) ** 2))
