import pandas as pd
import pytest

from tsknn import optimize_params, mae


df = pd.read_csv("data/AirPassengers.csv")
df["Month"] = pd.to_datetime(df["Month"])
df.set_index("Month", inplace=True)
df = df.asfreq("MS")
df.columns = ["passengers"]
X = df["passengers"].values

param_grid = {"k": [1, 2], "lags": [3], "h": [2]}


def test_metric_string_equals_callable():
    best_s, score_s = optimize_params(X, param_grid, test_size=2, metric="mae")
    best_f, score_f = optimize_params(X, param_grid, test_size=2, metric=mae)
    assert best_s == best_f
    assert score_s == score_f


def test_unknown_metric_raises():
    with pytest.raises(ValueError):
        optimize_params(X, param_grid, metric="invalid")


def test_random_search_runs():
    params, score = optimize_params(X, param_grid, test_size=2, method="random", n_iter=1)
    assert params is not None


def test_bayes_search_runs():
    params, score = optimize_params(X, param_grid, test_size=2, method="bayes", n_iter=2, random_state=0)
    assert params is not None
