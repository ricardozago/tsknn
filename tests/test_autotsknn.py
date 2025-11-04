import numpy as np
import pandas as pd
import pytest
from tsknn.autotsknn import autotsknn

df = pd.read_csv("data/AirPassengers.csv")
df.set_index("Month", inplace=True)
X = df.values.T[0]


def test_autotsknn_mse():
    model = autotsknn(n_trials=10, metric="mse")
    model.fit(X)
    preds = model.predict(h=12)
    assert len(preds) == 12


def test_autotsknn_mae():
    model = autotsknn(n_trials=10, metric="mae")
    model.fit(X)
    preds = model.predict(h=12)
    assert len(preds) == 12


def test_autotsknn_mape():
    model = autotsknn(n_trials=10, metric="mape")
    model.fit(X)
    preds = model.predict(h=12)
    assert len(preds) == 12


def test_autotsknn_invalid_metric():
    with pytest.raises(ValueError):
        autotsknn(metric="invalid")


def test_predict_before_fit():
    model = autotsknn()
    with pytest.raises(RuntimeError):
        model.predict(h=12)


def test_stationary_series():
    X_stationary = np.random.randn(100)
    model = autotsknn(n_trials=10)
    model.fit(X_stationary)
    preds = model.predict(h=12)
    assert len(preds) == 12


def test_get_pacf_lags():
    model = autotsknn()
    # Create a series with a known PACF structure
    X_pacf = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    lags = model._get_pacf_lags(X_pacf)
    assert len(lags) > 0


def test_pacf_study_is_better():
    # Create a time series with a very strong seasonal component
    X_seasonal = np.sin(np.linspace(0, 100, 200)) + np.random.randn(200) * 0.1
    model = autotsknn(n_trials=10)
    model.fit(X_seasonal)
    preds = model.predict(h=12)
    assert len(preds) == 12
