import pandas as pd
from tsknn import cross_validate_params, optimize_params


df = pd.read_csv("data/AirPassengers.csv")
df["Month"] = pd.to_datetime(df["Month"])
df.set_index("Month", inplace=True)
df = df.asfreq("MS")
df.columns = ["passengers"]
X = df["passengers"].values

param_grid = {"k": [1, 2], "lags": [3], "h": [2]}


def test_cv_runs():
    params, score = cross_validate_params(X, param_grid, test_size=2, n_splits=2)
    assert params["k"] in [1, 2]
    assert params["lags"] == 3


def test_cv_equals_optimize():
    p1, s1 = cross_validate_params(X, param_grid, test_size=2, n_splits=1)
    p2, s2 = optimize_params(X, param_grid, test_size=2)
    assert p1 == p2
    assert s1 == s2
