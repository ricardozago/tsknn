import numpy as np
import pandas as pd
from numpy.testing import assert_raises

from tsknn import tsknn

df = pd.read_csv("data/AirPassengers.csv")
df.set_index("Month", inplace=True)
df = df[-24:]
X = df.values.T[0]


def test_distances():
    model_euclidean = tsknn(
        cf="weighted", h=1, transform="multiplicative", lags=3, distance="euclidean"
    )
    model_euclidean.fit(X)
    x_pred_euclidean = model_euclidean.predict()

    model_manhattan = tsknn(
        cf="weighted", h=1, transform="multiplicative", lags=3, distance="manhattan"
    )
    model_manhattan.fit(X)
    x_pred_manhattan = model_manhattan.predict()

    assert_raises(
        AssertionError,
        np.testing.assert_array_equal,
        x_pred_euclidean,
        x_pred_manhattan,
    )


def test_invalid_distance():
    with assert_raises(ValueError):
        tsknn(distance="invalid_distance")
