import pandas as pd
from tsknn import optimize_params

# Carrega a série de exemplo
DF_PATH = "data/AirPassengers.csv"
df = pd.read_csv(DF_PATH)
df["Month"] = pd.to_datetime(df["Month"])
df.set_index("Month", inplace=True)
df = df.asfreq("MS")
X = df["passengers"].values

# Define o grid de parâmetros a ser testado
param_grid = {
    "k": [2, 3, 4],
    "lags": [3, 5],
    "h": [12],
}

# Busca aleatória pelos melhores parâmetros usando MAE
best_params, score = optimize_params(
    X,
    param_grid,
    metric="mae",
    method="random",
    n_iter=5,
)
print("Melhores parâmetros:", best_params)
print("Score obtido:", score)
