import pandas as pd
from tsknn import autotsknn

# Load the sample time series
DF_PATH = "data/AirPassengers.csv"
df = pd.read_csv(DF_PATH)
df["Month"] = pd.to_datetime(df["Month"])
df.set_index("Month", inplace=True)
df = df.asfreq("MS")
X = df["passengers"].values

# Automated search of k and lags using Bayesian optimisation
model, params, score = autotsknn(
    X,
    h=12,
    search_method="bayes",
    n_iter=10,
    random_state=0,
)

print("Best parameters:", params)
print("Score:", score)

# Generate forecasts with the resulting model
X_pred = X[-params["lags"]:]
print("Forecast:", model.predict(X_pred))
