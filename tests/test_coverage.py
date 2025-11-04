import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_almost_equal, assert_raises

from tsknn import tsknn

df = pd.read_csv("data/AirPassengers.csv")
df.set_index("Month", inplace=True)
df = df[-24:]
X = df.values.T[0]


def test_init_invalid_params():
    with assert_raises(ValueError):
        tsknn(k="invalid")
    with assert_raises(ValueError):
        tsknn(k=[1, "invalid"])
    with assert_raises(ValueError):
        tsknn(cf="invalid")
    with assert_raises(ValueError):
        tsknn(transform="invalid")
    with assert_raises(ValueError):
        tsknn(lags="invalid")
    with assert_raises(ValueError):
        tsknn(lags=[1, "invalid"])
    with assert_raises(ValueError):
        tsknn(h=0)
    with assert_raises(ValueError):
        tsknn(msas="invalid")
    with assert_raises(ValueError):
        tsknn(random_state="invalid")


def test_fit_invalid_input():
    model = tsknn()
    with assert_raises(TypeError):
        model.fit("invalid")
    with assert_raises(ValueError):
        model.fit(np.array([[1, 2], [3, 4]]))
    with assert_raises(ValueError):
        model = tsknn(msas="mimo", k=10)
        model.fit(X)


def test_predict_invalid_input():
    model = tsknn()
    model.fit(X)
    with assert_raises(TypeError):
        model.predict("invalid")
    with assert_raises(ValueError):
        model.predict(np.array([[1, 2], [3, 4]]))
    with assert_raises(ValueError):
        model.predict(np.array([1, 2]))


def test_integer_lags():
    model = tsknn(lags=5)
    model.fit(X)
    model.predict()


def test_median_cf():
    model = tsknn(cf="median")
    model.fit(X)
    model.predict()


def test_multiple_k():
    model = tsknn(k=[3, 5, 7])
    model.fit(X)
    model.predict()


def test_weighted_cf_zero_distance():
    model = tsknn(cf="weighted", k=3)
    # Create a situation where the distance is zero
    X_test = np.copy(X)
    X_test[-3:] = X_test[:3]
    model.fit(X_test)
    model.predict()


def test_additive_transform():
    model = tsknn(transform="additive")
    model.fit(X)
    model.predict()


def test_lag_indices():
    model = tsknn(lags=[1, 3, 5])
    model.fit(X)
    model.predict()


def test_force_stable():
    # This test creates a scenario with multiple equidistant (zero-distance)
    # neighbors to verify the correctness of stable sorting.
    X_fit = np.array([5, 5, 10, 5, 5, 20, 5, 5, 30, 1, 2, 3, 4])
    X_pred_vec = np.array([5, 5])

    # With stable sort, the model should find the first zero-distance neighbor (y=10)
    # because it appears earliest in the input data.
    model_stable = tsknn(force_stable=True, cf="weighted", k=3, lags=2, h=1)
    model_stable.fit(X_fit)
    x_pred_stable = model_stable.predict(X_pred_vec)
    assert_almost_equal(x_pred_stable, np.array([10]))
