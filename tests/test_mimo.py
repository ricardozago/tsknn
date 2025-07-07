import numpy as np
import pandas as pd
from numpy.testing import assert_almost_equal

from tsknn.tsknn import tsknn

np.set_printoptions(suppress=True)

df = pd.read_csv("data/AirPassengers.csv")
df["Month"] = pd.to_datetime(df["Month"])
df.set_index("Month", inplace=True)
df = df.asfreq("MS")
df.columns = ["passengers"]
df.head()

df = df[-24:]
X = df.values.T[0]
X_pred = df.values.T[0][-3:]


def test_answer():
    model = tsknn(cf="mean", h=12, transform="multiplicative", lags=3, msas="mimo")
    model.fit(X)
    x_pred = model.predict(X_pred)
    resp = np.array(
        [
            469.94095183,
            496.33109352,
            530.45167869,
            541.48866369,
            530.14307769,
            500.28989803,
            525.41630756,
            506.41834013,
            478.21661893,
            475.52638617,
            454.32574339,
            484.28273386,
        ]
    )
    assert_almost_equal(x_pred, resp, decimal=5)


def test_weighted_mimo():
    model = tsknn(
        cf="weighted", h=12, transform="multiplicative", lags=3, msas="mimo", k=3
    )
    model.fit(X)
    preds = model.predict(X_pred)

    idx, dist = model._get_k_closest_positions(X_pred)
    k_close = model._get_k_closest(idx)
    manual = np.average(k_close, axis=0, weights=1 / np.sqrt(dist + 1e-8))

    assert_almost_equal(preds, manual)


def test_weighted_distance_mimo():
    model = tsknn(
        cf="weighted",
        h=12,
        transform="multiplicative",
        lags=3,
        msas="mimo",
        k=3,
        weight_by="distance",
    )
    model.fit(X)
    preds = model.predict(X_pred)

    idx, dist = model._get_k_closest_positions(X_pred)
    k_close = model._get_k_closest(idx)
    manual = np.average(k_close, axis=0, weights=1 / np.sqrt(dist + 1e-8))

    assert_almost_equal(preds, manual)
