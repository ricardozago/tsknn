import pandas as pd
from tsknn import autotsknn

# Carrega a série de exemplo
DF_PATH = "data/AirPassengers.csv"
df = pd.read_csv(DF_PATH)
df["Month"] = pd.to_datetime(df["Month"])
df.set_index("Month", inplace=True)
df = df.asfreq("MS")
X = df["passengers"].values

# Busca automatizada de k e lags usando otimização bayesiana
model, params, score = autotsknn(
    X,
    h=12,
    search_method="bayes",
    n_iter=10,
    random_state=0,
)

print("Melhores parâmetros:", params)
print("Score obtido:", score)

# Gera previsões com o modelo encontrado
X_pred = X[-params["lags"]:]
print("Previsão:", model.predict(X_pred))
