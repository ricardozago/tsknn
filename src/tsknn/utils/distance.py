import numpy as np

def sum_euclidean(M: np.ndarray, v: np.ndarray) -> np.ndarray:
    """
    Calculates the sum of squared Euclidean distances between each row of M and vector v.
    Args:
        M (np.ndarray): Sample matrix (n_samples, n_features).
        v (np.ndarray): Comparison vector (n_features,).
    Returns:
        np.ndarray: Array of distances for each row of M.
    """
    # https://stackoverflow.com/a/49633639
    tmp = M - v
    return np.einsum('ij,ij->i', tmp, tmp)


def get_distance(distance: str = "euclidean"):
    """
    Returns the appropriate distance function.
    Args:
        distance (str): Name of the distance metric.
    Returns:
        Callable: Distance function.
    """
    if distance == "euclidean":
        return sum_euclidean
    return sum_euclidean
