"""Example script demonstrating the usage of :class:`tsknn`."""

from tsknn.tsknn import tsknn
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
lags = 5
X_pred = df.values.T[0][-lags:]

# model = tsknn(cf = "mean", h=12, transform = "multiplicative", lags = 3, k=4)
# model = tsknn(cf = "mean", h=120, transform = None, lags = lags, k=4, msas = "recursive")
model = tsknn(cf="mean", h=5, transform="multiplicative", lags=lags, k=6, msas="mimo")

model.fit(X)
x_pred = model.predict(X_pred)
print(x_pred)
