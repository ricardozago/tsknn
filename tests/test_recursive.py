import numpy as np
import pandas as pd
from numpy.testing import assert_almost_equal
from tsknn import tsknn

df = pd.read_csv("data/AirPassengers.csv")
df.set_index("Month", inplace=True)
df = df[-24:]
X = df.values.T[0]


def test_answer():
    model = tsknn(cf="mean", h=36, transform="multiplicative", lags=3)
    model.fit(X)
    x_pred = model.predict()
    resp = np.array([477.77255465, 513.41943769, 534.67589608, 558.11551684,
                     595.47514856, 670.11628215, 741.96465098, 792.99898348,
                     806.55037482, 791.50334854, 882.60907403, 965.27425906,
                     991.58542677, 964.63389125, 1078.41066551, 1180.83369168,
                     1211.17178891, 1178.98146252, 1318.15809422, 1442.9762442,
                     1480.25351313, 1440.89856716, 1610.93030212, 1763.52161537,
                     1809.06852284, 1760.96302359, 1968.77400651, 2155.25722875,
                     2210.92065982, 2152.13119576, 2406.10285532, 2634.01007378,
                     2702.03851129, 2630.18976515, 2940.57685707, 3219.10976935])
    assert_almost_equal(x_pred, resp, decimal=5)

    model = tsknn(
        cf="mean",
        h=5,
        transform="multiplicative",
        lags=5,
        k=6,
        msas="mimo")
    model.fit(X)
    x_pred = model.predict()
    resp = np.array([454.68436422, 484.95522414, 514.55798944,
                     534.52379374, 559.34988627])
    assert_almost_equal(x_pred, resp, decimal=5)
