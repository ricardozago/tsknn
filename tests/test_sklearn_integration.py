import pandas as pd
from sklearn.pipeline import Pipeline

from tsknn import tsknn

df = pd.read_csv("data/AirPassengers.csv")
df["Month"] = pd.to_datetime(df["Month"])
df.set_index("Month", inplace=True)
df = df.asfreq("MS")
df.columns = ["passengers"]
X = df["passengers"].values


def test_pipeline_predict():
    pipe = Pipeline([("model", tsknn(lags=3, h=2))])
    pipe.fit(X)
    preds = pipe.predict(X[-3:])
    assert len(preds) == 2
