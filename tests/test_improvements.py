import numpy as np
import pandas as pd
from numpy.testing import assert_almost_equal

from tsknn import select_lags_pacf, tsknn

df = pd.read_csv("data/AirPassengers.csv")
df["Month"] = pd.to_datetime(df["Month"])
df.set_index("Month", inplace=True)
df = df.asfreq("MS")
df.columns = ["passengers"]

df = df[-24:]
X = df.values.T[0]
X_pred = df.values.T[0][-3:]


def test_combine_k():
    m1 = tsknn(k=1, cf="mean", h=3, transform="multiplicative", lags=3)
    m2 = tsknn(k=2, cf="mean", h=3, transform="multiplicative", lags=3)
    mc = tsknn(k=[1, 2], cf="mean", h=3, transform="multiplicative", lags=3)
    for m in (m1, m2, mc):
        m.fit(X)
    p1 = m1.predict(X_pred)
    p2 = m2.predict(X_pred)
    pc = mc.predict(X_pred)
    assert_almost_equal(pc, (p1 + p2) / 2)


def test_direct_equals_recursive_h1():
    mr = tsknn(h=1, lags=3, transform="multiplicative", msas="recursive")
    md = tsknn(h=1, lags=3, transform="multiplicative", msas="direct")
    for m in (mr, md):
        m.fit(X)
    pr = mr.predict(X_pred)
    pd = md.predict(X_pred)
    assert_almost_equal(pr, pd)


def test_select_lags_pacf():
    lags = select_lags_pacf(X, 3)
    assert isinstance(lags, list)
    assert all(isinstance(l, int) for l in lags)
