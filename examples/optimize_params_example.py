import pandas as pd
from tsknn import optimize_params

# Load the sample time series
DF_PATH = "data/AirPassengers.csv"
df = pd.read_csv(DF_PATH)
df["Month"] = pd.to_datetime(df["Month"])
df.set_index("Month", inplace=True)
df = df.asfreq("MS")
X = df["passengers"].values

# Parameter grid to test
param_grid = {
    "k": [2, 3, 4],
    "lags": [3, 5],
    "h": [12],
}

# Random search for the best parameters using MAE
best_params, score = optimize_params(
    X,
    param_grid,
    metric="mae",
    method="random",
    n_iter=5,
)
print("Best parameters:", best_params)
print("Score:", score)
