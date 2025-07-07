import pandas as pd

from tsknn import freq_params, infer_freq

df = pd.read_csv("data/AirPassengers.csv")
df["Month"] = pd.to_datetime(df["Month"])
df.set_index("Month", inplace=True)
df = df.asfreq("MS")


def test_infer_freq_detects_ms():
    freq = infer_freq(df["#Passengers"])
    assert freq == "MS"


def test_freq_params_defaults():
    assert freq_params("D") == (7, 7)
    assert freq_params("W") == (4, 4)
    assert freq_params("unknown") == (3, 12)
