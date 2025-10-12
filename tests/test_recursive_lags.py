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
    model = tsknn(cf="mean", h=12, transform=None, lags=lags, k=3, msas="recursive")
    model.fit(X)
    x_pred = model.predict()
    resp = np.array(
        [
            778.77112586,
            778.66876205,
            778.93768879,
            780.12744963,
            779.1832364,
            780.33531401,
            779.81882229,
            779.84256139,
            777.0321362,
            778.40770075,
            776.60578508,
            778.69765369,
        ]
    )
    assert_almost_equal(x_pred, resp, decimal=5)

    lags = [1, 2, 6]

    model = tsknn(
        cf="mean", h=12, transform="multiplicative", lags=lags, k=3, msas="recursive"
    )
    model.fit(X)
    x_pred = model.predict()
    resp = np.array(
        [
            780.81790847,
            779.40178815,
            780.20120556,
            783.01163126,
            779.1198421,
            781.30977658,
            776.55323128,
            779.57674757,
            773.89554726,
            776.37681803,
            774.5325975,
            773.9800934,
        ]
    )
    assert_almost_equal(x_pred, resp, decimal=5)
