import numpy as np
import pandas as pd
from tsknn import cross_validate_params, mae

# Load sample data
DF_PATH = "data/AirPassengers.csv"
df = pd.read_csv(DF_PATH)
X = df["#Passengers"].values

# Define a custom metric (mean absolute scaled error)
def mase(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    scale = np.mean(np.abs(np.diff(y_true)))
    return np.mean(np.abs(y_true - y_pred)) / scale

# Search best parameters using cross-validation with the custom metric
params, score = cross_validate_params(
    X,
    {"k": range(1, 4), "lags": range(1, 4), "h": [2]},
    n_splits=3,
    metric=mase,
)
print("Best params:", params)
print("Score:", score)
