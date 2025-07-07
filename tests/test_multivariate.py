import pandas as pd
import numpy as np
from numpy.testing import assert_almost_equal
from tsknn import tsknn, mtsknn


df = pd.read_csv("data/AirPassengers.csv")
df["Month"] = pd.to_datetime(df["Month"])
df.set_index("Month", inplace=True)
df = df.asfreq('MS')
df.columns = ["passengers"]

df = df[-24:]
df["passengers2"] = df["passengers"] * 2
X = df[["passengers", "passengers2"]].values


def test_mtsknn_shapes():
    model = mtsknn(lags=3, h=2)
    model.fit(X)
    preds = model.predict(X[-3:])
    assert preds.shape == (2, 2)


def test_mtsknn_equivalence():
    m1 = tsknn(lags=3, h=2)
    m2 = tsknn(lags=3, h=2)
    m1.fit(X[:, 0])
    m2.fit(X[:, 1])
    p1 = m1.predict(X[-3:, 0])
    p2 = m2.predict(X[-3:, 1])

    mm = mtsknn(lags=3, h=2)
    mm.fit(X)
    pm = mm.predict(X[-3:])

    assert_almost_equal(pm[:, 0], p1)
    assert_almost_equal(pm[:, 1], p2)
