from src.tsknn.tsknn import tsknn
from numpy.testing import assert_almost_equal
import pandas as pd
import numpy as np
np.set_printoptions(suppress=True)

df = pd.read_csv("data/AirPassengers.csv")
df["Month"] = pd.to_datetime(df["Month"])
df.set_index("Month", inplace=True)
df = df.asfreq('MS')
df.columns = ["passengers"]
df.head()

df = df[-24:]
X = df.values.T[0]
X_pred = df.values.T[0][-3:]


def test_answer():
    model = tsknn(cf="mean", h=12, transform="multiplicative", lags=3, msas="mimo")
    model.fit(X)
    x_pred = model.predict(X_pred)
    resp = np.array([469.94095183, 496.33109352, 530.45167869, 541.48866369, 530.14307769,
                     500.28989803, 525.41630756, 506.41834013, 478.21661893, 475.52638617,
                     454.32574339, 484.28273386])
    assert_almost_equal(x_pred, resp, decimal=5)
