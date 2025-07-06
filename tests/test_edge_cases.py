import numpy as np
import pytest
from tsknn import tsknn


def test_small_series_mimo_error():
    X = np.arange(5, dtype=float)
    model = tsknn(lags=3, h=4, msas="mimo")
    with pytest.raises(ValueError):
        model.fit(X)


def test_nan_values_propagate():
    X = np.array([1.0, 2.0, np.nan, 4.0, 5.0, 6.0])
    model = tsknn(lags=2, h=1)
    model.fit(X)
    pred = model.predict(X[-2:])
    assert np.isnan(pred).any()
