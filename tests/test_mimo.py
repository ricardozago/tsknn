import numpy as np
import pandas as pd
from numpy.testing import assert_almost_equal

from tsknn import tsknn

df = pd.read_csv("data/AirPassengers.csv")
df.set_index("Month", inplace=True)
df = df[-24:]
X = df.values.T[0]


def test_answer():
    model = tsknn(cf="mean", h=12, transform="multiplicative", lags=3, msas="mimo")
    model.fit(X)
    x_pred = model.predict()
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
