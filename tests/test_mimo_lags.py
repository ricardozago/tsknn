import numpy as np
import pandas as pd
from numpy.testing import assert_almost_equal
from tsknn import tsknn

df = pd.read_csv("data/serie_temporal.csv")
df.set_index("Date", inplace=True)
X = df.values.T[0]
lags = [1, 5, 6]


def test_answer():
    lags = [1, 5, 6]
    model = tsknn(cf="mean", h=12, transform=None, lags=lags, k=3, msas="mimo")
    model.fit(X)
    x_pred = model.predict()
    resp = np.array(
        [
            778.77112586,
            777.71784266,
            779.88881445,
            780.20742693,
            780.67856748,
            782.25235874,
            778.74935692,
            778.98903091,
            780.24489713,
            779.51670631,
            780.06152174,
            782.13080863,
        ]
    )
    assert_almost_equal(x_pred, resp, decimal=5)

    lags = [1, 2, 6]

    model = tsknn(
        cf="mean", h=12, transform="multiplicative", lags=lags, k=3, msas="mimo"
    )
    model.fit(X)
    x_pred = model.predict()
    resp = np.array(
        [
            780.81790847,
            780.40550243,
            780.51228385,
            779.56718005,
            782.10200198,
            777.22971824,
            778.05454031,
            778.71651937,
            777.82629493,
            775.41861458,
            777.20763805,
            774.73717218,
        ]
    )
    assert_almost_equal(x_pred, resp, decimal=5)
