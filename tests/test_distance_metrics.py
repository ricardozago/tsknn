import numpy as np

from tsknn.tsknn import get_distance


def test_distance_functions():
    arr = np.array([[0, 0], [1, 1]])
    v = np.array([1, 0])
    euclidean = get_distance("euclidean")(arr, v)
    manhattan = get_distance("manhattan")(arr, v)
    chebyshev = get_distance("chebyshev")(arr, v)
    cosine = get_distance("cosine")(arr, v)
    assert np.allclose(euclidean, np.array([1, 1]))
    assert np.allclose(manhattan, np.array([1, 1]))
    assert np.allclose(chebyshev, np.array([1, 1]))
    assert np.allclose(np.round(cosine, 8), np.array([1.0, 0.29289322]))
