import pandas as pd
from tsknn import autotsknn


df = pd.read_csv("data/AirPassengers.csv")
df["Month"] = pd.to_datetime(df["Month"])
df.set_index("Month", inplace=True)
df = df.asfreq("MS")
df.columns = ["passengers"]
X = df["passengers"].values


def test_autotsknn_returns_model():
    model, params, score = autotsknn(X, k_values=[1, 2], lags_values=[3], h=2, test_size=2, metric="mae")
    assert params["k"] in [1, 2]
    assert params["lags"] == 3
    X_pred = X[-3:]
    preds = model.predict(X_pred)
    assert len(preds) == 2
