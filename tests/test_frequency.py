import pandas as pd

from tsknn import autotsknn

df = pd.read_csv("data/AirPassengers.csv")
df["Month"] = pd.to_datetime(df["Month"])
df.set_index("Month", inplace=True)
df = df.asfreq("MS")
df.columns = ["passengers"]

weekly = df.resample("W").interpolate()
daily = df.resample("D").interpolate()


def test_autotsknn_weekly_freq():
    series = weekly["passengers"]
    model, params, score = autotsknn(series, freq="W")
    assert params["lags"] == 4
    assert model.h == 4
    preds = model.predict(series.values[-4:])
    assert len(preds) == 4


def test_autotsknn_daily_freq():
    series = daily["passengers"]
    model, params, score = autotsknn(series, freq="D")
    assert params["lags"] == 7
    assert model.h == 7
    preds = model.predict(series.values[-7:])
    assert len(preds) == 7
