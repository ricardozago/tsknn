from sklearn.pipeline import Pipeline
import pandas as pd
from tsknn import tsknn


df = pd.read_csv("data/AirPassengers.csv")
df["Month"] = pd.to_datetime(df["Month"])
df.set_index("Month", inplace=True)
df = df.asfreq("MS")
X = df["passengers"].values

pipeline = Pipeline([("model", tsknn(lags=3, h=2))])
pipeline.fit(X)
print(pipeline.predict(X[-3:]))
